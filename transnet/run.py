"""Steps shared by ``transnet train`` and ``transnet predict``."""

import logging
from pathlib import Path

import pandas as pd
import torch

from transnet import features, gtf, graphs as graph_io
from transnet.model import TranscriptScorer, score

log = logging.getLogger("transnet")


def prepare(args, gtfs, chromosomes, reference_chroms=None):
    """Extract graphs for each candidate GTF independently.

    Returns {name: dict(gtf, records, graphs, exported, assignments)}.
    Each GTF is its own guide, graph set and phasing; nothing is pooled.
    """
    work = Path(args.work)
    logfile = work / "commands.log"
    work.mkdir(parents=True, exist_ok=True)
    truth = None
    if args.reference:
        truth = gtf.reference_chains(args.reference, set(reference_chroms or chromosomes))
        log.info("Reference: %d multi-exon intron chains on selected chromosomes", len(truth))
    features.check_library_type(args.reads, args.library_type)
    log.info("Filtering BAM to %d chromosomes", len(chromosomes))
    bam = features.filter_bam(args.bam, chromosomes, work / "bam", args.threads, logfile)
    args.insert_size_estimate = None
    if args.reads == "long":
        log.info("Long reads: library unstranded, insert size 0/0/0 (no estimation)")
    elif args.insert_size is not None:
        log.info("Insert size given: median %d, low %d, high %d", *args.insert_size)
    else:
        # One estimate from the filtered BAM, used for every GTF of this run.
        estimate = features.estimate_insert_size(bam, args.library_type, logfile)
        args.insert_size = [estimate["median"], estimate["low"], estimate["high"]]
        args.insert_size_estimate = estimate
        log.info(
            "Insert size estimated by the exporter preview from %d read pairs: median %d, low %d, high %d "
            "(mean %.2f +/- %.2f); used for all GTFs (equivalent to --insert-size %d,%d,%d)",
            estimate["pairs"], estimate["median"], estimate["low"], estimate["high"],
            estimate["mean"], estimate["std"], *args.insert_size,
        )
    library = features.library_arguments(args.reads, args.library_type, args.insert_size)
    result = {}
    for name, path in gtfs.items():
        records = gtf.candidates(path, set(chromosomes))
        log.info("[%s] %d eligible multi-exon candidates", name, len(records))
        if not records:
            raise ValueError(f"No eligible multi-exon candidates in {path} on selected chromosomes")
        folder = work / name
        folder.mkdir(parents=True, exist_ok=True)
        guide = folder / "candidates.gtf"
        gtf.write_guide(guide, records)
        log.info("[%s] Exporting splice graphs", name)
        features.export_graphs(bam, guide, folder, name, library, logfile)
        log.info("[%s] Measuring junction quality", name)
        features.junction_quality(bam, folder)
        graphs, summary, exported = graph_io.build_graphs(folder, records, truth, args.max_nodes, args.threads)
        for g in graphs:
            g.source_gtf = name
        log.info(
            "[%s] %d graphs, %d candidate paths exported, %d graphs over %d nodes skipped",
            name, len(graphs), summary["exported_paths"], summary["graphs_over_size_limit"], args.max_nodes,
        )
        result[name] = dict(
            gtf=path, records=records, graphs=graphs, exported=exported,
            assignments=features.assignment_counts(folder), truth=truth,
        )
    return result


def graph_groups(graphs, batch_size=32, pair_budget=2000000):
    """Consecutive graphs, at most ``batch_size`` per batch and bounded candidate x fragment pairs."""
    group, paths, fragments = [], 0, 0
    for g in graphs:
        p, f = len(g.y), len(g.fragments)
        if group and (len(group) >= batch_size or (paths + p) * (fragments + f) > pair_budget):
            yield group
            group, paths, fragments = [], 0, 0
        group.append(g)
        paths += p
        fragments += f
    if group:
        yield group


@torch.no_grad()
def predict_graphs(model, graphs, device, batch_size=32, pair_budget=2000000):
    """Mean sigmoid score per internal candidate ID over all graphs that contain it."""
    model.eval()
    rows = []
    for group in graph_groups(graphs, batch_size, pair_budget):
        z = score(model, group, device).sigmoid().cpu().tolist()
        ids = [t for g in group for t in g.transcript_ids]
        if len(ids) != len(z):
            raise ValueError("Scores do not align with candidates")
        rows.extend(zip(ids, z))
    frame = pd.DataFrame(rows, columns=["candidate", "score"])
    return frame.groupby("candidate", sort=False).score.mean().to_dict()


def transcript_table(item, scores):
    """One row per eligible candidate of a GTF, in GTF order.

    Candidates without a score (not exported to a graph, or in a graph above the
    size limit) get score 0 and a status explaining why, matching the paper's
    evaluation convention.
    """
    rows = []
    for cid, r in item["records"].items():
        if cid in scores:
            value, status = scores[cid], "scored"
        else:
            value = 0.0
            if cid in item["exported"]:
                status = "graph_too_large"
            elif cid in item["assignments"]:
                candidate, assigned = item["assignments"][cid]
                status = (
                    "no_read_bundle" if candidate == 0
                    else "bundle_assignment_rejected" if assigned == 0
                    else "assigned_without_exported_path"
                )
            else:
                status = "not_exported"
        row = dict(
            transcript_id=r["original_id"],
            chr=r["chr"],
            strand=r["strand"],
            start=r["exons"][0][0],
            end=r["exons"][-1][1],
            exons=len(r["exons"]),
            native_abundance=r["cov"],
            score=value,
            status=status,
        )
        if item["truth"] is not None:
            row["label"] = int(gtf.chain_key(r) in item["truth"])
        rows.append(row)
    columns = ["transcript_id", "chr", "strand", "start", "end", "exons",
               "native_abundance", "score", "status"]
    if item["truth"] is not None:
        columns.append("label")
    return pd.DataFrame(rows, columns=columns)


def save_model(path, model, fit, reads, metadata):
    torch.save(
        dict(format="transnet-model-v1", reads=reads, fit=fit, state_dict=model.state_dict(), metadata=metadata),
        path,
    )


PRETRAINED = Path(__file__).resolve().parent / "pretrained"


def load_model(name_or_path, device):
    path = Path(name_or_path)
    if not path.exists():
        path = PRETRAINED / f"{name_or_path}.pt"
    if not path.exists():
        names = sorted(p.stem for p in PRETRAINED.glob("*.pt"))
        raise FileNotFoundError(f"No model file {name_or_path!r}; pretrained models: {', '.join(names)}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("format") != "transnet-model-v1":
        raise ValueError(f"{path} is not a TransNet model file")
    model = TranscriptScorer(data["fit"])
    model.load_state_dict(data["state_dict"], strict=True)
    return model.to(device).eval(), data, path
