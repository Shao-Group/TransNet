"""Feature standardisation statistics, fitted on training graphs only.

The fit is stored inside every TransNet model file and reused unchanged at
prediction time. Labels are never read.

* ``scaling``          signed-log mean/std of node features, edge features and
                       native abundance
* ``pair_scaling``     mean/std of the twelve candidate/fragment pair descriptors
* ``quality_scaling``  mean/std of the transformed junction-quality fields, over
                       splice edges used by at least one candidate

Each group also records the training ``minimum`` and ``maximum`` of every
feature and whether it varied (``active``: range above
``model.CONSTANT_RANGE``). A feature that was constant in training carries no
information the model could learn from, so the model zeroes it, and its
stored std is set to 1. The presence of ``active`` switches on the model's
safeguards (see ``TranscriptScorer``).
"""

import torch

from transnet.model import (
    CONSTANT_RANGE,
    PAIR_FEATURES,
    QUALITY_FIELDS,
    collate,
    signed_log,
    transform_quality,
)


class _Stats:
    """Running sum, sum of squares, minimum and maximum of each feature (float64)."""

    def __init__(self, width):
        self.total = torch.zeros(width, dtype=torch.float64)
        self.squares = self.total.clone()
        self.minimum = torch.full_like(self.total, torch.inf)
        self.maximum = torch.full_like(self.total, -torch.inf)
        self.n = 0

    def add(self, values):
        values = values.double().reshape(-1, len(self.total))
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite training features")
        self.total += values.sum(0)
        self.squares += values.square().sum(0)
        self.n += len(values)
        if len(values):
            self.minimum = torch.minimum(self.minimum, values.min(0).values)
            self.maximum = torch.maximum(self.maximum, values.max(0).values)

    def finish(self, name):
        if self.n == 0:
            raise ValueError(f"No training rows to fit {name} feature scaling")
        mean = self.total / self.n
        std = (self.squares / self.n - mean.square()).clamp_min(0).sqrt().clamp_min(1e-06)
        active = self.maximum - self.minimum > CONSTANT_RANGE
        std = torch.where(active, std, torch.ones_like(std))
        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "fit_rows": self.n,
            "active": active.tolist(),
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
        }


def fit(graphs):
    if not graphs:
        raise ValueError("Cannot fit feature scaling without training graphs")
    stats = dict(
        node=_Stats(graphs[0].x.shape[1]),
        edge=_Stats(2),
        abundance=_Stats(1),
        pair=_Stats(len(PAIR_FEATURES)),
        quality=_Stats(len(QUALITY_FIELDS)),
    )
    for g in graphs:
        stats["node"].add(signed_log(g.x.double()))
        stats["edge"].add(signed_log(g.edge_attr.double()))
        stats["abundance"].add(signed_log(g.abundance.double()))
        # Descriptors of every candidate/fragment pair that shares a node.
        stats["pair"].add(collate([g]).allocation_features)
        # Quality of the splice edges used by at least one candidate.
        edges = {tuple(e): i for i, e in enumerate(g.edge_index.T.tolist())}
        used = sorted({edges[e] for s, t in zip(g.splice_sources, g.splice_targets) for e in zip(s, t)})
        stats["quality"].add(transform_quality(g.edge_quality[used].double()))
    result = {name: s.finish(name) for name, s in stats.items()}
    return dict(
        scaling={name: result[name] for name in ("node", "edge", "abundance")},
        pair_scaling=result["pair"],
        quality_scaling=result["quality"],
        preprocessing=dict(quality_transform="log of support_count and anchor_min_mean",
                           safeguards=f"constant mask (range <= {CONSTANT_RANGE}) and clip to [-8, 8]"),
    )
