"""Build validated PyTorch Geometric graphs from the exporter's feature tables.

One graph per exported splice graph. Each graph carries:

* ``x``             nine node features (NODE_FEATURES), raw
* ``edge_index``    directed edges, graph-local node indices
* ``edge_attr``     two edge features (weight, genomic gap length), raw
* ``edge_quality``  nine BAM junction-quality fields per edge row
* ``candidate_paths``, ``splice_sources``, ``splice_targets``  per candidate
* ``fragments``, ``fragment_counts``  phasing fragments and read counts
* ``abundance``     the assembler's native abundance per candidate
* ``y``             reference label per candidate (0 when no reference given)
* ``transcript_ids`` internal candidate IDs (see transnet.gtf.candidates)
"""

from concurrent.futures import ThreadPoolExecutor
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from transnet.gtf import chain_key
from transnet.model import QUALITY_FIELDS

log = logging.getLogger("transnet")

NODE_FEATURES = [
    "weight",
    "length",
    "maxcov",
    "stddev",
    "indel_sum_cov",
    "indel_ratio",
    "left_indel",
    "right_indel",
    "newly_added_length",
]
EDGE_FEATURES = ["weight", "length"]
TABLES = ["node.csv", "edge.csv", "path.label.csv", "phasing.csv"]


def load_tables(folder):
    dtype = {c: str for c in ("chr", "graph_id", "path_id", "sample", "node_sequence",
                              "splice_source", "splice_target")}
    return [
        pd.read_csv(Path(folder) / f"features.{suffix}", dtype=dtype, keep_default_na=False)
        for suffix in TABLES
    ]


def parse_sequence(value, field, allow_empty=False):
    if pd.isna(value) or str(value).strip() == "":
        if allow_empty:
            return []
        raise ValueError(f"Empty {field}")
    return [int(token.strip()) for token in str(value).split(",")]


def label_paths(tables, records, truth):
    """Check every exported path against its GTF candidate, then label it.

    The exported node sequence must reproduce the candidate's exons and the
    splice pairs its intron chain. ``truth`` is a set of reference intron
    chains, or None (all labels 0) when no reference annotation was supplied.
    """
    nodes, _, paths, _ = tables
    lookup = {
        (gid, int(nid)): (chrom, int(start), int(end))
        for gid, nid, chrom, start, end in nodes[
            ["graph_id", "node_id", "chr", "start_pos", "end_pos"]
        ].itertuples(index=False, name=None)
    }
    labels = []
    for row in paths.itertuples(index=False):
        prefix = row.graph_id + "."
        if not row.path_id.startswith(prefix):
            raise ValueError("Candidate path lost its graph identity")
        record = records[row.path_id[len(prefix):]]
        merged = []
        for node in parse_sequence(row.node_sequence, "candidate"):
            chrom, start, end = lookup[(row.graph_id, node)]
            if chrom != record["chr"] or row.chr != chrom:
                raise ValueError("Candidate chromosome differs from GTF")
            if merged and start == merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        if merged != [tuple(exon) for exon in record["exons"]]:
            raise ValueError("Exported path does not reproduce the GTF exons")
        sources = parse_sequence(row.splice_source, "source", True)
        targets = parse_sequence(row.splice_target, "target", True)
        pairs = tuple(
            (lookup[(row.graph_id, s)][2], lookup[(row.graph_id, t)][1] - 1)
            for s, t in zip(sources, targets)
        )
        if len(sources) != len(targets) or pairs != chain_key(record)[2]:
            raise ValueError("Exported path does not reproduce the GTF intron chain")
        labels.append(int(truth is not None and chain_key(record) in truth))
    paths["label"] = labels


def _graph(nodes, edges, paths, phasing, graph_id):
    """Validate one graph's rows and convert them to a Data object."""
    nodes = nodes.sort_values("node_id", kind="stable")
    original_ids = nodes["node_id"].tolist()
    if len(set(original_ids)) != len(original_ids) or any(int(n) != n or n < 0 for n in original_ids):
        raise ValueError(f"Graph {graph_id} has duplicate or invalid node IDs")
    mapping = {int(node): i for i, node in enumerate(original_ids)}

    def sequences(frame, column, allow_empty=False):
        try:
            return [[mapping[n] for n in parse_sequence(v, column, allow_empty)] for v in frame[column]]
        except KeyError as e:
            raise ValueError(f"Graph {graph_id} references missing node {e.args[0]}") from e

    if paths["path_id"].duplicated().any():
        raise ValueError(f"Graph {graph_id} has duplicate candidate path IDs")
    candidates = sequences(paths, "node_sequence")
    sources = sequences(paths, "splice_source", True)
    targets = sequences(paths, "splice_target", True)
    fragments = sequences(phasing, "node_sequence")
    edge_pairs = [(mapping[int(s)], mapping[int(t)]) for s, t in edges[["source", "target"]].itertuples(index=False, name=None)]
    if len(set(edge_pairs)) != len(edge_pairs):
        raise ValueError(f"Graph {graph_id} has duplicate directed edges")
    edge_set = set(edge_pairs)
    starts, ends = nodes["start_pos"].tolist(), nodes["end_pos"].tolist()
    lengths = edges["length"].tolist()
    for (s, t), length in zip(edge_pairs, lengths):
        if length != starts[t] - ends[s]:
            raise ValueError(f"Graph {graph_id} edge length disagrees with node coordinates")
    splice_edges = {pair for pair, length in zip(edge_pairs, lengths) if length != 1}
    for path, src, dst in zip(candidates, sources, targets):
        traversed = list(zip(path, path[1:]))
        if any(pair not in edge_set for pair in traversed):
            raise ValueError(f"Graph {graph_id} candidate traverses an absent edge")
        # Splice pairs must equal all splice transitions of the path, in order.
        if len(src) != len(dst) or list(zip(src, dst)) != [p for p in traversed if p in splice_edges]:
            raise ValueError(f"Graph {graph_id} splice pairs differ from the candidate's splices")
    d = Data(
        x=torch.tensor(nodes[NODE_FEATURES].to_numpy(dtype=np.float32)),
        edge_index=torch.tensor(np.asarray(edge_pairs, dtype=np.int64).reshape(-1, 2).T),
        edge_attr=torch.tensor(edges[EDGE_FEATURES].to_numpy(dtype=np.float32)),
    )
    d.candidate_paths, d.splice_sources, d.splice_targets = candidates, sources, targets
    d.fragments = fragments
    d.fragment_counts = torch.tensor(phasing["count"].to_numpy(dtype=np.float32))
    d.y = torch.tensor(paths["label"].to_numpy(dtype=np.float32))
    d.abundance = torch.tensor(paths["abundance"].to_numpy(dtype=np.float32))
    d.graph_id = str(graph_id)
    d.sample_id = str(paths["sample"].iloc[0])
    d.transcript_ids = [p[len(graph_id) + 1:] for p in paths["path_id"].astype(str)]
    d.node_ids = original_ids
    d.locus_chr = str(nodes["chr"].iloc[0])
    d.locus_start = int(nodes["start_pos"].min())
    d.locus_end = int(nodes["end_pos"].max())
    for field in ("x", "edge_attr", "abundance", "fragment_counts"):
        if not torch.isfinite(getattr(d, field)).all():
            raise ValueError(f"Graph {graph_id} has nonfinite {field}")
    if (d.fragment_counts < 0).any() or (d.abundance < 0).any():
        raise ValueError("Fragment counts and abundance must be nonnegative")
    return d


def load_quality(path):
    rows = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["sample"], row["graph_id"], int(row["source"]), int(row["target"]))
            values = [float(row[f]) for f in QUALITY_FIELDS]
            if not np.isfinite(values).all() or min(values) < 0:
                raise ValueError(f"Invalid junction quality: {key}")
            if key in rows:
                raise ValueError(f"Duplicate junction quality identity: {key}")
            rows[key] = values
    return rows


def attach_quality(graph, quality):
    """Quality rows in the graph's edge order; a missing row is an error."""
    values = []
    for s, t in graph.edge_index.T.tolist():
        key = (graph.sample_id, graph.graph_id, int(graph.node_ids[s]), int(graph.node_ids[t]))
        if key not in quality:
            raise ValueError(f"Missing junction quality for edge {key}")
        values.append(quality[key])
    graph.edge_quality = torch.as_tensor(values, dtype=torch.float32).reshape(-1, len(QUALITY_FIELDS))


def build_graphs(folder, records, truth, max_nodes=10000, threads=4):
    """All graphs with at least one candidate and at most ``max_nodes`` nodes.

    Returns (graphs, summary, exported candidate IDs). Graph order follows sorted graph IDs; candidate
    order within a graph follows the exporter's path table.
    """
    tables = load_tables(folder)
    label_paths(tables, records, truth)
    for frame in tables:
        for gid, part in frame.groupby("graph_id", sort=False):
            if part["chr"].nunique() != 1 or part["sample"].nunique() != 1:
                raise ValueError(f"graph_id {gid} spans chromosomes or samples")
    identities = {}
    for frame in tables:
        for gid, part in frame.groupby("graph_id", sort=False):
            identity = (str(part["chr"].iloc[0]), str(part["sample"].iloc[0]))
            if gid in identities and identities[gid] != identity:
                raise ValueError(f"graph_id {gid} changes chromosome/sample across feature tables")
            identities[gid] = identity
    grouped = [{gid: f for gid, f in t.groupby("graph_id", sort=False)} for t in tables]
    if set(grouped[2]) - set(grouped[0]):
        raise ValueError("Path table contains graphs absent from the node table")
    keys = sorted(grouped[0])
    allowed = [g for g in keys if g in grouped[2] and (max_nodes == 0 or len(grouped[0][g]) <= max_nodes)]

    def build(gid):
        frames = [groups.get(gid, table.iloc[:0]) for groups, table in zip(grouped, tables)]
        return _graph(*frames, gid)

    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        graphs = list(pool.map(build, allowed))
    quality = load_quality(Path(folder) / "quality.csv")
    for g in graphs:
        attach_quality(g, quality)
    summary = dict(
        graphs=len(keys),
        graphs_with_candidates=sum(g in grouped[2] for g in keys),
        graphs_over_size_limit=sum(g in grouped[2] and g not in set(allowed) for g in keys),
        exported_paths=len(tables[2]),
    )
    log.info("Graphs: %s", summary)
    exported = {
        pid[len(gid) + 1:] for gid, pid in tables[2][["graph_id", "path_id"]].itertuples(index=False, name=None)
    }
    return graphs, summary, exported
