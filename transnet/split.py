"""Training/validation split of graphs. Never reads labels.

Two modes:

* chromosome lists (default): graphs on ``--val-chroms`` validate, graphs on
  ``--train-chroms`` train.
* random loci (``--val-fraction``): graphs from all GTFs are grouped into
  genomic overlap components (same chromosome, transitively overlapping
  intervals), so the same locus never appears in both roles, even across
  GTFs. A seeded shuffle holds out that fraction of components.
"""

import random


def by_chromosome(graphs, train_chroms, val_chroms):
    if set(train_chroms) & set(val_chroms):
        raise ValueError("Training and validation chromosomes overlap")
    train = [g for g in graphs if g.locus_chr in train_chroms]
    validation = [g for g in graphs if g.locus_chr in val_chroms]
    return train, validation


def by_components(graphs, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    groups, group, chrom, right = [], [], None, -1
    for c, start, end, i in sorted((g.locus_chr, g.locus_start, g.locus_end, i) for i, g in enumerate(graphs)):
        if c != chrom or start > right:
            if group:
                groups.append(group)
            group, right, chrom = [], end, c
        group.append(i)
        right = max(right, end)
    if group:
        groups.append(group)
    if len(groups) < 2:
        raise ValueError("Need at least two independent loci to split")
    random.Random(seed).shuffle(groups)
    n = max(1, min(len(groups) - 1, round(fraction * len(groups))))
    held = {i for group in groups[:n] for i in group}
    train = [g for i, g in enumerate(graphs) if i not in held]
    validation = [g for i, g in enumerate(graphs) if i in held]
    return train, validation
