"""TransNet transcript scorer.

The network scores every candidate transcript of a splice graph in three parts:

1. **Graph encoder.** One GATv2 layer (two heads of width 32) embeds the splice
   graph nodes (partial exons) from nine node features and two edge features.
2. **Sequence encoders.** Ordered LSTMs read each candidate's node sequence and
   its splice-junction sequence. A junction token combines the attention-
   weighted endpoint embeddings, the raw edge features, a small edge MLP and
   nine BAM junction-quality fields. Phasing fragments (read paths through the
   graph) are encoded the same way.
3. **Fragment competition.** Over three rounds, each phasing fragment splits its
   read count between the candidates that fully contain it and a per-fragment
   null option (softmax, mass conserved). Candidates update a recurrent state
   from the evidence they receive; a final MLP outputs one logit per candidate.

Candidates with identical node/junction sequences (aliases) compete once and
share one score. Scores are returned in the original candidate order.

Parameter and buffer names match the published checkpoints in
``transnet/pretrained``. No genomic coordinate, assembler identity or
nucleotide sequence is a neural input.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax

# Pair geometry between a candidate path and a phasing fragment (see describe_pair).
PAIR_FEATURES = (
    "shared_path_nodes",
    "shared_fragment_nodes",
    "shared_path_edges",
    "shared_fragment_edges",
    "matched_run_path",
    "matched_run_fragment",
    "unmatched_fragment_prefix",
    "unmatched_fragment_suffix",
    "matched_path_start",
    "matched_path_end",
    "log_path_length",
    "log_fragment_length",
)
# BAM evidence per splice junction, produced by transnet.features.junction_quality.
QUALITY_FIELDS = (
    "support_count",
    "anchor_min_mean",
    "anchor_ge8_fraction",
    "anchor_ge20_fraction",
    "mapq_mean",
    "mapq_ge20_fraction",
    "mapq_valid_fraction",
    "nm_rate_mean",
    "near_indel_fraction",
)
# Relation of a phasing fragment to a candidate path.
FULL, OVERHANG, CONFLICT = range(3)

HIDDEN = 32  # width of each GAT head and of the fragment encoder
HEADS = 2
LSTM = 32
DROPOUT = 0.2
TOKEN_WIDTH = 2 * HIDDEN * HEADS + 2 + 16 + 16  # 162: two endpoints, raw edge, edge MLP, quality
ROUNDS = 3
SEQUENCE_TOKEN_BUDGET = 250000  # bounds padding memory for very long sequence batches
# Safeguards (models whose fit records constant features): standardised values
# are clipped to [-CLIP, CLIP] and features constant in training are zeroed.
CLIP = 8
CONSTANT_RANGE = 1e-8  # a feature whose training range is at most this is constant
SAFEGUARDED = ("node", "edge", "abundance", "pair", "quality")


def signed_log(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def transform_quality(values):
    """Log the two count-like quality fields; the seven fractions/means stay linear."""
    if values.shape[-1] != len(QUALITY_FIELDS):
        raise ValueError("Expected the nine BAM-quality fields")
    return torch.cat((signed_log(values[..., :2]), values[..., 2:]), dim=-1)


# ---------------------------------------------------------------------------
# Candidate/fragment pair construction (CPU, done once per batch)
# ---------------------------------------------------------------------------


def contains(path, fragment):
    return any(
        path[i : i + len(fragment)] == fragment for i in range(len(path) - len(fragment) + 1)
    )


def fragment_compatible(path, fragment):
    """True when a partially overlapping fragment only extends past a candidate end.

    A shared interior run followed by a different branch is a conflict.
    """
    if contains(path, fragment):
        return True
    shared = set(path).intersection(fragment)
    if not shared:
        return False
    p = [i for i, node in enumerate(path) if node in shared]
    f = [i for i, node in enumerate(fragment) if node in shared]
    if p[-1] - p[0] + 1 != len(p) or f[-1] - f[0] + 1 != len(f):
        return False
    if path[p[0] : p[-1] + 1] != fragment[f[0] : f[-1] + 1]:
        return False
    return (f[0] == 0 or p[0] == 0) and (f[-1] == len(fragment) - 1 or p[-1] == len(path) - 1)


def phase_relation(path, fragment):
    """FULL (contained), OVERHANG (compatible overlap), CONFLICT, or None (disjoint)."""
    if not path or not fragment:
        raise ValueError("Empty candidate or phasing sequence")
    if contains(path, fragment):
        return FULL
    if set(path).isdisjoint(fragment):
        return None
    return OVERHANG if fragment_compatible(path, fragment) else CONFLICT


def describe_pair(path, frag):
    """Twelve label-free structural descriptors of an overlapping path/fragment pair."""
    shared = set(path) & set(frag)
    if not shared:
        raise ValueError("Disjoint pair is not eligible")
    # Longest consecutive shared run (dynamic programming keeps repeated nodes).
    positions = {}
    for i, node in enumerate(path):
        positions.setdefault(node, []).append(i)
    best = (0, 0, 0)
    previous = {}
    for j, node in enumerate(frag):
        current = {i: previous.get(i - 1, 0) + 1 for i in positions.get(node, [])}
        for i, length in current.items():
            if length > best[0]:
                best = (length, i - length + 1, j - length + 1)
        previous = current
    length, pstart, fstart = best
    pe = set(zip(path, path[1:]))
    fe = set(zip(frag, frag[1:]))
    edges = len(pe & fe)
    return [
        len(shared) / len(set(path)),
        len(shared) / len(set(frag)),
        edges / max(len(pe), 1),
        edges / max(len(fe), 1),
        length / len(path),
        length / len(frag),
        fstart / len(frag),
        (len(frag) - fstart - length) / len(frag),
        pstart / len(path),
        (pstart + length) / len(path),
        math.log1p(len(path)),
        math.log1p(len(frag)),
    ]


def pad_sequences(sequences, device=None):
    """Rows are sequences in input order; -1 is padding (node 0 is a real node)."""
    width = max((len(seq) for seq in sequences), default=0)
    result = torch.full((len(sequences), width), -1, dtype=torch.long, device=device)
    for row, seq in enumerate(sequences):
        if seq:
            result[row, : len(seq)] = torch.as_tensor(seq, dtype=torch.long, device=device)
    return result


def collate(graphs, check_labels=False):
    """Batch graphs and attach the candidate/fragment competition structure.

    Node indices in every sequence are shifted by their graph's node offset.
    Identical candidates within a graph (aliases) share one competitor; their
    native abundances are averaged once. Labels are only read (to reject
    conflicting aliases) when ``check_labels`` is set during training.
    """
    b = Batch.from_data_list(graphs)
    paths, fragments, sources, targets = [], [], [], []
    for i in range(b.num_graphs):
        offset = int(b.ptr[i])
        n = int(b.ptr[i + 1]) - offset
        src, dst = b.splice_sources[i], b.splice_targets[i]
        if len(b.candidate_paths[i]) != len(src) or len(src) != len(dst):
            raise ValueError("Candidate and splice list counts differ")
        if any(len(s) != len(t) for s, t in zip(src, dst)):
            raise ValueError("Splice source/target lengths differ for a candidate")
        for output, sequences in (
            (paths, b.candidate_paths[i]),
            (fragments, b.fragments[i]),
            (sources, src),
            (targets, dst),
        ):
            if any(node < 0 or node >= n for seq in sequences for node in seq):
                raise ValueError("Sequence references a missing graph node")
            output.extend([[node + offset for node in seq] for seq in sequences])
    device = b.x.device
    b.candidate_paths = pad_sequences(paths, device)
    b.fragments = pad_sequences(fragments, device)
    b.splice_sources = pad_sequences(sources, device)
    b.splice_targets = pad_sequences(targets, device)
    b.candidate_graph = torch.tensor(
        [i for i, g in enumerate(graphs) for _ in g.candidate_paths], dtype=torch.long
    )
    b.fragment_graph = torch.tensor(
        [i for i, g in enumerate(graphs) for _ in g.fragments], dtype=torch.long
    )
    if len(b.candidate_graph) != len(b.y) or len(b.fragment_graph) != len(b.fragment_counts):
        raise ValueError("Candidate or fragment count mismatch")

    representatives, inverse, unique_graph, pairs, features, relations = [], [], [], [], [], []
    coverage = b.abundance.clone()
    po = fo = 0
    for gid, graph in enumerate(graphs):
        groups = {}
        for i, (path, src, dst) in enumerate(
            zip(graph.candidate_paths, graph.splice_sources, graph.splice_targets)
        ):
            if not path:
                raise ValueError("Empty candidate sequence")
            groups.setdefault((tuple(path), tuple(src), tuple(dst)), []).append(i)
        local_inverse = [None] * len(graph.candidate_paths)
        for key, aliases in groups.items():
            uid = len(representatives)
            representatives.append(po + aliases[0])
            unique_graph.append(gid)
            if check_labels and len(set(graph.y[aliases].tolist())) != 1:
                raise ValueError("Conflicting labels for duplicate candidate structure")
            value = graph.abundance[aliases].unique().double().mean().to(coverage.dtype)
            coverage[torch.tensor(aliases) + po] = value
            for i in aliases:
                local_inverse[i] = uid
            path = list(key[0])
            for j, fragment in enumerate(graph.fragments):
                relation = phase_relation(path, fragment)
                if relation is not None:
                    pairs.append((uid, fo + j))
                    features.append(describe_pair(path, fragment))
                    relations.append(relation)
        inverse.extend(local_inverse)
        po += len(graph.candidate_paths)
        fo += len(graph.fragments)
    b.abundance = coverage
    b.allocation_representatives = torch.tensor(representatives, dtype=torch.long)
    b.allocation_inverse = torch.tensor(inverse, dtype=torch.long)
    b.allocation_candidate_graph = torch.tensor(unique_graph, dtype=torch.long)
    b.allocation_pairs = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()
    b.allocation_features = torch.tensor(features, dtype=torch.float32).reshape(
        -1, len(PAIR_FEATURES)
    )
    b.allocation_relation = torch.tensor(relations, dtype=torch.long)
    return b


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class EvidenceGATv2(GATv2Conv):
    """Two-head GATv2 over the splice graph (edge features: weight, length)."""

    def __init__(self, in_channels=9):
        super().__init__(in_channels, HIDDEN, heads=HEADS, edge_dim=2, dropout=DROPOUT)

    def edge_update(self, x_j, x_i, edge_attr, index, ptr, dim_size):
        x = x_i + x_j
        if edge_attr is not None:
            x = x + self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        logits = (F.leaky_relu(x, self.negative_slope) * self.att).sum(-1)
        alpha = softmax(logits, index, ptr, dim_size)
        return F.dropout(alpha, p=self.dropout, training=self.training)


def allocate(pair_logits, pairs, null_logits, counts, num_candidates):
    """Per-fragment softmax over compatible candidates plus a null option.

    Returns pair probabilities, null probabilities and the allocated count mass
    (float64). A fragment without compatible candidates goes entirely to null.
    """
    if pairs.shape != (2, len(pair_logits)) or null_logits.shape != counts.shape:
        raise ValueError("Allocation dimensions differ")
    pi, fi = pairs
    if len(pi) and (
        int(pi.min()) < 0
        or int(pi.max()) >= num_candidates
        or int(fi.min()) < 0
        or int(fi.max()) >= len(counts)
    ):
        raise ValueError("Allocation index out of range")
    values = pair_logits.double()
    null = null_logits.double()
    maximum = null.detach().clone()
    maximum.scatter_reduce_(0, fi, values.detach(), reduce="amax", include_self=True)
    weights = (values - maximum[fi]).exp()
    null_weights = (null - maximum).exp()
    denominator = null_weights.index_add(0, fi, weights)
    probability = weights / denominator[fi]
    null_probability = null_weights / denominator
    return (
        probability,
        null_probability,
        counts[fi].double() * probability,
        counts.double() * null_probability,
    )


def mass_summary(values, weights, indices, n):
    """Count-weighted mean of ``values`` and log1p total mass per receiving candidate."""
    mass = weights.new_zeros(n).index_add(0, indices, weights)
    total = weights.new_zeros((n, values.shape[1])).index_add(
        0, indices, weights[:, None] * values.double()
    )
    context = total / torch.where(mass > 0, mass, torch.ones_like(mass))[:, None]
    return context.to(values.dtype), mass.log1p().to(values.dtype)


def _edge_keys(edge_index, num_nodes):
    keys = edge_index[0] * num_nodes + edge_index[1]
    sorted_keys, order = keys.sort()
    if len(keys) > 1 and (sorted_keys[1:] == sorted_keys[:-1]).any():
        raise ValueError("Duplicate directed edges are ambiguous")
    return sorted_keys, order


def _lookup(mapping, requested):
    """Row index of each requested edge key; a missing edge is an error, never a fallback."""
    keys, order = mapping
    if requested.numel() == 0:
        return requested
    if len(keys) == 0:
        raise ValueError("Requested splice edges are absent from the graph")
    positions = torch.searchsorted(keys, requested)
    valid = positions < len(keys)
    valid &= keys[positions.clamp_max(len(keys) - 1)] == requested
    if not valid.all():
        raise ValueError("Requested splice edge is absent from the graph")
    return order[positions]


def _encode(embeddings, lengths, lstm):
    """Last LSTM hidden state per row; empty rows give zeros."""
    result = embeddings.new_zeros((len(lengths), lstm.hidden_size))
    nonempty = lengths > 0
    if nonempty.any():
        packed = pack_padded_sequence(
            embeddings[nonempty], lengths[nonempty].cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = lstm(packed)
        result[nonempty] = hidden[-1]
    return result


def _blocks(lengths, width):
    """Split long padded batches into length-sorted blocks; None when unnecessary."""
    if len(lengths) < 2 or len(lengths) * width <= SEQUENCE_TOKEN_BUDGET:
        return None
    order = torch.argsort(lengths, descending=True, stable=True)
    ranked = lengths[order].tolist()
    blocks = []
    start = 0
    while start < len(ranked):
        local_width = int(ranked[start])
        stop = min(len(ranked), start + max(1, SEQUENCE_TOKEN_BUDGET // max(local_width, 1)))
        blocks.append((order[start:stop], local_width))
        start = stop
    return blocks


def _blockwise(function, lengths, width, *sequences):
    """Apply ``function`` per block and restore the original row order."""
    blocks = _blocks(lengths, width)
    if not blocks:
        return None
    values = [function(*(s[idx, :w] for s in sequences)) for idx, w in blocks]
    inverse = torch.argsort(torch.cat([idx for idx, _ in blocks]))
    if isinstance(values[0], tuple):
        return tuple(torch.cat([v[k] for v in values])[inverse] for k in range(len(values[0])))
    return torch.cat(values)[inverse]


class TranscriptScorer(nn.Module):
    """GATv2 + ordered LSTMs + three-round fragment competition.

    ``fit`` holds training-set statistics (see transnet.fit) used to
    standardise node, edge, abundance, pair and quality features.

    Safeguards. A fit made by current TransNet also stores, for every feature,
    whether it varied in the training data (``active``). Such a model
    (``self.safeguards``) zeroes features that were constant in training and
    clips every standardised feature to [-CLIP, CLIP], so an unusual value in a
    new dataset cannot push the network far outside its training range. Model
    files whose fit has no ``active`` entries (older releases) are scored
    exactly as before.
    """

    def __init__(self, fit):
        super().__init__()
        scaling = fit["scaling"]
        for field in ("node", "edge", "abundance"):
            for kind in ("mean", "std"):
                self.register_buffer(
                    field + "_" + kind, torch.tensor(scaling[field][kind], dtype=torch.float32)
                )
        for prefix in ("pair", "quality", "fragment_quality"):
            stats = fit["pair_scaling" if prefix == "pair" else "quality_scaling"]
            for kind in ("mean", "std"):
                self.register_buffer(
                    prefix + "_" + kind, torch.tensor(stats[kind], dtype=torch.float32)
                )
        groups = dict(scaling, pair=fit["pair_scaling"], quality=fit["quality_scaling"])
        masked = ["active" in groups[name] for name in SAFEGUARDED]
        if any(masked) and not all(masked):
            raise ValueError("Model fit has safeguard masks for only some feature groups")
        self.safeguards = all(masked)
        if self.safeguards:
            for name in SAFEGUARDED:
                self.register_buffer(name + "_active", torch.tensor(groups[name]["active"], dtype=torch.bool))

        self.gat_layers = nn.ModuleList([EvidenceGATv2()])
        width = HIDDEN * HEADS
        self.node_seq_lstm = nn.LSTM(width, LSTM, batch_first=True)
        self.fragment_lstm = nn.LSTM(width, HIDDEN, batch_first=True)
        self.edge_embedding = nn.Sequential(
            nn.Linear(2, 16), nn.ELU(), nn.Linear(16, 16), nn.ELU(), nn.Dropout(DROPOUT)
        )

        def quality_encoder():
            return nn.Sequential(
                nn.Linear(len(QUALITY_FIELDS), 16),
                nn.ELU(),
                nn.Linear(16, 16),
                nn.ELU(),
                nn.Dropout(DROPOUT),
            )

        self.quality_encoder = quality_encoder()
        self.splice_lstm = nn.LSTM(TOKEN_WIDTH, LSTM, batch_first=True)
        self.fragment_quality_encoder = quality_encoder()
        self.fragment_splice_lstm = nn.LSTM(TOKEN_WIDTH, LSTM, batch_first=True)
        self.fragment_projection = nn.Linear(LSTM, LSTM)
        nn.init.normal_(self.fragment_projection.weight, std=0.01)
        nn.init.zeros_(self.fragment_projection.bias)
        # Candidate vector: node LSTM (32) + junction LSTM (32) + native abundance (1).
        path_width = LSTM + LSTM + 1
        # Message: assigned/overhang/conflict fragment contexts (3x32) + seven log masses.
        message_width = HIDDEN * 3 + 7
        self.transcript_state = nn.Linear(path_width, LSTM)
        self.routing_path = nn.Linear(LSTM, 16)
        self.routing_fragment = nn.Linear(HIDDEN, 16)
        self.affinity = nn.Sequential(
            nn.Linear(64 + len(PAIR_FEATURES), 64), nn.ELU(), nn.Linear(64, 1)
        )
        self.support_intensity = nn.Linear(LSTM, 1)
        self.null_affinity = nn.Linear(HIDDEN, 1)
        self.transcript_update = nn.GRUCell(message_width, LSTM)
        self.correctness = nn.Sequential(
            nn.Linear(path_width + LSTM + message_width, LSTM), nn.ELU(), nn.Linear(LSTM, 1)
        )

    def guarded_raw(self, values, name):
        """Clip and mask features in their unstandardised units (safeguarded models only).

        The values are standardised with the ``name`` statistics, clipped to
        [-CLIP, CLIP], zeroed where the feature was constant in training, and
        mapped back to raw units. The caller then standardises them as usual,
        so the arithmetic matches the research implementation exactly.
        """
        if not self.safeguards:
            return values
        mean, std = getattr(self, name + "_mean"), getattr(self, name + "_std")
        z = ((values - mean) / std).clamp(-CLIP, CLIP) * getattr(self, name + "_active")
        return z * std + mean

    def guard_standardised(self, values, name):
        """Clip and mask already standardised node or edge features."""
        if not self.safeguards:
            return values
        return values.clamp(-CLIP, CLIP) * getattr(self, name + "_active")

    # -- sequence encoders --------------------------------------------------

    def _nodes(self, x, sequences, lstm):
        valid = sequences >= 0
        split = _blockwise(
            lambda s: self._nodes(x, s, lstm), valid.sum(1), sequences.shape[1], sequences
        )
        if split is not None:
            return split
        if valid.any() and sequences[valid].max() >= len(x):
            raise ValueError("Node sequence references a missing node")
        embeddings = x[sequences.clamp_min(0)] * valid.unsqueeze(-1)
        return _encode(embeddings, valid.sum(1), lstm)

    def _token(self, x, attention, edge_attr, quality, source, target, encoder, mean, std):
        """162-wide junction token for each (source, target) splice edge."""
        weights = attention.unsqueeze(-1)

        def weighted(nodes):
            return (x[nodes].reshape(-1, HEADS, HIDDEN) * weights).reshape(-1, HEADS * HIDDEN)

        quality = (self.guarded_raw(transform_quality(quality), "quality") - mean) / std
        return torch.cat(
            [weighted(source), weighted(target), edge_attr, self.edge_embedding(edge_attr), encoder(quality)],
            -1,
        )

    def _junctions(self, ctx, sources, targets):
        """Encode each candidate's ordered splice junctions."""
        if sources.shape != targets.shape or not torch.equal(sources >= 0, targets >= 0):
            raise ValueError("Splice source/target pairs do not align")
        valid = sources >= 0
        split = _blockwise(
            lambda s, t: self._junctions(ctx, s, t), valid.sum(1), sources.shape[1], sources, targets
        )
        if split is not None:
            return split
        x, edge_attr, quality, original, attention_map, attention = ctx
        src, dst = sources[valid], targets[valid]
        requested = src * len(x) + dst
        token = self._token(
            x,
            attention[_lookup(attention_map, requested)],
            edge_attr[_lookup(original, requested)],
            quality[_lookup(original, requested)],
            src,
            dst,
            self.quality_encoder,
            self.quality_mean,
            self.quality_std,
        )
        padded = x.new_zeros((*sources.shape, TOKEN_WIDTH))
        padded[valid] = token
        return _encode(padded, valid.sum(1), self.splice_lstm)

    def _fragment_junctions(self, ctx, fragments):
        """Encode the splice transitions inside each phasing fragment."""
        if ((fragments[:, :-1] < 0) & (fragments[:, 1:] >= 0)).any():
            raise ValueError("Fragment padding interrupts original node order")
        split = _blockwise(
            lambda f: self._fragment_junctions(ctx, f),
            (fragments >= 0).sum(1),
            fragments.shape[1],
            fragments,
        )
        if split is not None:
            return split
        x, edge_attr, quality, original, attention_map, attention = ctx
        is_splice = self._is_splice  # computed from raw edge lengths in forward()
        width = max(fragments.shape[1] - 1, 0)
        padded = x.new_zeros((len(fragments), width, TOKEN_WIDTH))
        lengths = torch.zeros(len(fragments), dtype=torch.long, device=x.device)
        if width == 0 or len(fragments) == 0:
            return _encode(padded, lengths, self.fragment_splice_lstm), lengths
        left, right = fragments[:, :-1], fragments[:, 1:]
        valid = (left >= 0) & (right >= 0)
        source, target = left[valid], right[valid]
        if source.numel() and torch.max(torch.stack((source, target))) >= len(x):
            raise ValueError("Fragment transition references a missing node")
        # Fragment transitions that are graph edges and splices (not contiguous exon parts).
        requested = source * len(x) + target
        keys, order = original
        positions = torch.searchsorted(keys, requested)
        exists = positions < len(keys)
        if len(keys):
            exists &= keys[positions.clamp_max(len(keys) - 1)] == requested
        selected = torch.zeros_like(exists)
        original_ids = order[positions[exists]] if exists.any() else positions[:0]
        selected[exists] = is_splice[original_ids]
        mask = torch.zeros_like(valid)
        mask[valid] = selected
        lengths = mask.sum(1)
        if selected.any():
            source, target = source[selected], target[selected]
            edge_ids = original_ids[is_splice[original_ids]]
            tokens = self._token(
                x,
                attention[_lookup(attention_map, source * len(x) + target)],
                edge_attr[edge_ids],
                quality[edge_ids],
                source,
                target,
                self.fragment_quality_encoder,
                self.fragment_quality_mean,
                self.fragment_quality_std,
            )
            rows = mask.nonzero(as_tuple=True)[0]
            positions = mask.long().cumsum(1)[mask] - 1
            padded[rows, positions] = tokens
        return _encode(padded, lengths, self.fragment_splice_lstm), lengths

    # -- forward ---------------------------------------------------------------

    def forward(self, b):
        """Return one logit per candidate of a ``collate`` batch, in input order."""
        if b.edge_attr.ndim != 2 or b.edge_attr.shape[1] != 2:
            raise ValueError("Expected two-column graph-edge features")
        if b.edge_quality.shape != (len(b.edge_attr), len(QUALITY_FIELDS)):
            raise ValueError("BAM quality features misaligned with graph edges")
        # Edge feature column 1 is genomic gap length; 1 means adjacent exon parts.
        self._is_splice = b.edge_attr[:, 1] != 1.0
        try:
            return self._forward(b)
        finally:
            self._is_splice = None

    def _forward(self, b):
        x, edge_index = b.x, b.edge_index
        original = _edge_keys(edge_index, len(x))
        x = (signed_log(x) - self.node_mean) / self.node_std
        edge_attr = (signed_log(b.edge_attr) - self.edge_mean) / self.edge_std
        abundance = b.abundance
        if self.safeguards:
            # Clip/mask in signed-log space, then return to native units so the
            # standardisation below is unchanged (same arithmetic as research).
            transformed = self.guarded_raw(signed_log(abundance), "abundance")
            restored = torch.expm1(transformed.abs().double()).clamp(max=torch.finfo(abundance.dtype).max)
            abundance = torch.sign(transformed) * restored.to(abundance.dtype)
        abundance = (signed_log(abundance) - self.abundance_mean) / self.abundance_std
        x = self.guard_standardised(x, "node")
        edge_attr = self.guard_standardised(edge_attr, "edge")
        # GATv2 adds self loops, so attention rows have their own edge index map.
        x, (attention_edges, attention) = self.gat_layers[0](
            x, edge_index, edge_attr, return_attention_weights=True
        )
        x = F.elu(x)
        attention_map = _edge_keys(attention_edges, len(x))
        sources, targets = b.splice_sources, b.splice_targets
        if sources.shape != targets.shape or not torch.equal(sources >= 0, targets >= 0):
            raise ValueError("Splice source/target pairs do not align")
        _lookup(original, sources[sources >= 0] * len(x) + targets[targets >= 0])
        ctx = (x, edge_attr, b.edge_quality, original, attention_map, attention)
        nodes = self._nodes(x, b.candidate_paths, self.node_seq_lstm)
        splice = self._junctions(ctx, sources, targets)
        paths = torch.cat([nodes, splice, abundance.unsqueeze(-1)], -1)
        frags = self._nodes(x, b.fragments, self.fragment_lstm)
        junctions, lengths = self._fragment_junctions(ctx, b.fragments)
        frags = frags + self.fragment_projection(junctions) * (lengths > 0).unsqueeze(-1)
        logits = self._compete(b, paths, frags)
        if logits.ndim != 1 or not torch.isfinite(logits).all():
            raise ValueError("Nonfinite candidate logits")
        return logits

    def _compete(self, b, paths, frags):
        counts = b.fragment_counts
        inverse = b.allocation_inverse
        paths = paths[b.allocation_representatives]  # one competitor per alias group
        pairs, relation = b.allocation_pairs, b.allocation_relation
        pi, fi = pairs
        if not torch.equal(b.allocation_candidate_graph[pi], b.fragment_graph[fi]):
            raise ValueError("Cross-graph allocation forbidden")
        if len(inverse) != len(b.candidate_graph) or not torch.equal(
            b.allocation_candidate_graph[inverse], b.candidate_graph
        ):
            raise ValueError("Candidate alias graph identity differs")
        if not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Invalid fragment counts")
        n = len(paths)
        relative = (self.guarded_raw(b.allocation_features, "pair") - self.pair_mean) / self.pair_std
        full = relation == FULL
        full_pairs = pairs[:, full]
        spi, sfi = full_pairs
        f = self.routing_fragment(frags)
        # Overhanging and conflicting fragments are fixed context; only fully
        # contained fragments are routed.
        static = []
        for kind in (OVERHANG, CONFLICT):
            mask = relation == kind
            static.extend(mass_summary(frags[fi[mask]], counts[fi[mask]].double(), pi[mask], n))
        over_context, over_mass, conflict_context, conflict_mass = static
        _, available = mass_summary(frags[sfi], counts[sfi].double(), spi, n)
        # Number of compatible candidates per fragment (unique vs ambiguous support).
        degree = counts.new_zeros(len(frags)).double().index_add(
            0, sfi, counts.new_ones(len(sfi)).double()
        )
        state = torch.tanh(self.transcript_state(paths))
        null_logits = self.null_affinity(frags).squeeze(-1)
        for step in range(ROUNDS):
            p = self.routing_path(state)[spi]
            fragment = f[sfi]
            affinity = self.affinity(
                torch.cat([p, fragment, p * fragment, (p - fragment).abs(), relative[full]], -1)
            ).squeeze(-1)
            log_intensity = F.logsigmoid(self.support_intensity(state).squeeze(-1))
            probability, _, allocated, _ = allocate(
                affinity + log_intensity[spi], full_pairs, null_logits, counts, n
            )
            context, assigned = mass_summary(frags[sfi], allocated, spi, n)
            _, unique = mass_summary(frags[sfi], allocated * (degree[sfi] == 1), spi, n)
            _, ambiguous = mass_summary(frags[sfi], allocated * (degree[sfi] > 1), spi, n)
            _, unavailable = mass_summary(
                frags[sfi], counts[sfi].double() * (1 - probability), spi, n
            )
            masses = torch.stack(
                [assigned, available, unique, ambiguous, unavailable, over_mass, conflict_mass], -1
            )
            message = torch.cat([context, over_context, conflict_context, masses], -1)
            if step + 1 < ROUNDS:
                state = self.transcript_update(message, state)
        logits = self.correctness(torch.cat([paths, state, message], -1)).squeeze(-1)
        return logits[inverse]  # expand alias groups back to original candidate rows


def score(model, graphs, device):
    """Logits for a list of graphs, in candidate order."""
    return model(collate(graphs).to(device))
