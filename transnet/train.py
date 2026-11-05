"""Training with equal per-GTF transcript-weighted binary cross-entropy.

Loss weighting: with N candidate GTFs, each GTF contributes 1/N of the total
loss, divided equally among its unique transcripts (a transcript appearing in
k graphs gets 1/k of its weight in each). Every GTF is a separate forward pass
through the same shared model; GTF identity is not a model input.

Each update samples one batch group (one batch per GTF, sampled with a seeded
generator) and minimises the weighted loss with Adam and a cosine learning-rate
schedule. Every ``validate_every`` updates the model is scored on the
validation loci; the checkpoint with the lowest validation weighted BCE is kept.
"""

from collections import Counter
import copy
import logging
import random

import numpy as np
import torch
from torch.nn import functional as F

from transnet.model import TranscriptScorer, collate
from transnet.run import graph_groups

log = logging.getLogger("transnet")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def weighted_batches(graphs_by_gtf, batch_size, pair_budget, check_labels):
    """Collated batches, grouped so that group i holds batch i of every GTF."""
    names = [n for n, gs in graphs_by_gtf.items() if gs]
    sequences = []
    for name in names:
        cohort = graphs_by_gtf[name]
        counts = Counter(t for g in cohort for t in g.transcript_ids)
        for g in cohort:
            g.loss_weight = torch.tensor(
                [1.0 / (len(names) * len(counts) * counts[t]) for t in g.transcript_ids],
                dtype=torch.float64,
            )
        sequences.append([
            dict(
                batch=collate(group, check_labels=check_labels),
                y=torch.cat([g.y for g in group]),
                weight=torch.cat([g.loss_weight for g in group]),
                ids=[t for g in group for t in g.transcript_ids],
                gtf=name,
            )
            for group in graph_groups(cohort, batch_size, pair_budget)
        ])
    groups = [
        [seq[i] for seq in sequences if i < len(seq)]
        for i in range(max((len(s) for s in sequences), default=0))
    ]
    total = sum(float(b["weight"].sum()) for g in groups for b in g)
    if groups and abs(total - 1.0) > 1e-8:
        raise ValueError("Loss weights do not sum to one")
    return groups


def weighted_bce(logits, b):
    y, w = b["y"].to(logits.device), b["weight"].to(logits.device)
    return (F.binary_cross_entropy_with_logits(logits, y, reduction="none").double() * w).sum()


@torch.no_grad()
def validate(model, groups, device):
    """Weighted BCE over all validation candidates (lower is better)."""
    model.eval()
    loss = mass = 0.0
    for group in groups:
        for b in group:
            z = model(b["batch"].clone().to(device))
            loss += float(weighted_bce(z, b))
            mass += float(b["weight"].sum())
    return loss / mass


def train(train_graphs, val_graphs, fit, args, device):
    """Return (best model, history rows, best step)."""
    seed_everything(args.seed)
    train_groups = weighted_batches(train_graphs, args.batch_size, args.pair_budget, True)
    val_groups = weighted_batches(val_graphs, args.batch_size, args.pair_budget, True)
    if not train_groups or not val_groups:
        raise ValueError("Empty training or validation set")
    log.info("Training batch groups: %d; validation batch groups: %d", len(train_groups), len(val_groups))
    model = TranscriptScorer(fit).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.updates, eta_min=1e-06)
    sampler = random.Random(args.seed)
    best, best_step, best_state, history = float("inf"), 0, None, []
    running, interval = 0.0, 0
    for step in range(1, args.updates + 1):
        model.train()
        group = train_groups[sampler.randrange(len(train_groups))]
        optimizer.zero_grad(set_to_none=True)
        for b in group:
            # Scale by the number of groups so one update estimates the full loss.
            loss = weighted_bce(model(b["batch"].clone().to(device)), b) * len(train_groups)
            loss.backward()
            running += float(loss.detach())
        interval += 1
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError("Nonfinite gradients")
        optimizer.step()
        scheduler.step()
        if step % args.validate_every == 0 or step == args.updates:
            val = validate(model, val_groups, device)
            if val < best:
                best, best_step = val, step
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            history.append(dict(step=step, train_loss=running / interval, validation_bce=val, best_step=best_step))
            log.info("update %d/%d  train loss %.4f  validation BCE %.4f  (best %.4f at %d)",
                     step, args.updates, running / interval, val, best, best_step)
            running, interval = 0.0, 0
    model.load_state_dict(best_state)
    return model.eval(), history, best_step, best
