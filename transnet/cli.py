"""TransNet command line: ``transnet predict`` and ``transnet train``.

Run ``transnet predict -h`` or ``transnet train -h`` for all options; the
README explains inputs and outputs.
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import torch

from transnet import __version__, features, gtf
from transnet.run import load_model, predict_graphs, prepare, save_model, transcript_table

log = logging.getLogger("transnet")


def chromosome_list(value):
    return [c.strip() for c in value.split(",") if c.strip()]


def insert_size(value):
    parts = [int(v) for v in value.split(",")]
    if len(parts) != 3 or not parts[1] <= parts[0] <= parts[2]:
        raise argparse.ArgumentTypeError("expected MEDIAN,LOW,HIGH with LOW <= MEDIAN <= HIGH")
    return parts


def gtf_inputs(values):
    """NAME=PATH or PATH (NAME = file name without .gtf/.gz)."""
    result = {}
    for value in values:
        name, _, path = value.rpartition("=") if "=" in value else ("", "", value)
        name = name or Path(path).name.removesuffix(".gz").removesuffix(".gtf")
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"GTF name {name!r} may only use letters, digits, '_' and '-'")
        if name in result:
            raise ValueError(f"Duplicate GTF name {name!r}; use NAME=PATH")
        if not Path(path).exists():
            raise FileNotFoundError(path)
        result[name] = path
    return result


def common(p):
    p.add_argument("--bam", required=True, help="coordinate-sorted RNA-seq alignments")
    p.add_argument("--gtf", required=True, action="append", metavar="[NAME=]PATH",
                   help="candidate transcripts from one assembler (repeat for several); "
                        "transcript lines need the native abundance attribute 'cov'")
    p.add_argument("--out", required=True, help="output directory (must not already contain a run)")
    p.add_argument("--library-type", choices=features.LIBRARY_TYPES,
                   help="short reads only: strandedness (first = fr-firststrand / dUTP)")
    p.add_argument("--insert-size", type=insert_size, metavar="MEDIAN,LOW,HIGH",
                   help="short reads only: fragment insert size median and range "
                        "[estimated from the BAM by the exporter's Scallop2 preview]")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    p.add_argument("--max-nodes", type=int, default=10000,
                   help="skip splice graphs with more nodes (0 = no limit) [10000]")
    p.add_argument("--batch-size", type=int, default=32, help="graphs per batch [32]")
    p.add_argument("--pair-budget", type=int, default=2000000,
                   help="max candidates x fragments per batch [2000000]")
    p.add_argument("--keep-work", action="store_true",
                   help="keep intermediate files (filtered BAM, guides, graph tables) in OUT/work")


def parser():
    p = argparse.ArgumentParser(prog="transnet", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="version", version="TransNet " + __version__)
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("predict", help="score candidate transcripts with a trained model")
    common(pr)
    pr.add_argument("--model", required=True,
                    help="pretrained name (sr, pacbio, ont_cdna, ont_drna, simulation) or a model.pt path")
    pr.add_argument("--reference", help="optional reference annotation GTF: adds labels, AP and PR curves")
    pr.add_argument("--chromosomes", type=chromosome_list,
                    help="comma-separated chromosomes to score [all chromosomes of the GTFs found in the BAM]")

    tr = sub.add_parser("train", help="train a new model")
    common(tr)
    tr.add_argument("--reference", required=True, help="reference annotation GTF that defines correct candidates")
    tr.add_argument("--reads", required=True, choices=("short", "long"))
    tr.add_argument("--train-chroms", type=chromosome_list, help="chromosomes used for training")
    tr.add_argument("--val-chroms", type=chromosome_list, help="chromosomes used for validation")
    tr.add_argument("--chromosomes", type=chromosome_list,
                    help="with --val-fraction: chromosomes to use for both roles")
    tr.add_argument("--val-fraction", type=float,
                    help="hold out this fraction of genomic loci for validation instead of chromosome lists")
    tr.add_argument("--split-seed", type=int, default=42, help="seed for --val-fraction [42]")
    tr.add_argument("--updates", type=int, default=10400, help="optimizer updates [10400]")
    tr.add_argument("--validate-every", type=int, default=208, help="updates between validations [208]")
    tr.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate [3e-4]")
    tr.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay [1e-4]")
    tr.add_argument("--seed", type=int, default=42, help="training seed [42]")
    iq = sub.add_parser("isoquant-cov", help="add IsoQuant TPM as the 'cov' attribute of an IsoQuant GTF")
    iq.add_argument("--gtf", required=True, help="IsoQuant *.transcript_models.gtf")
    iq.add_argument("--tpm", required=True, help="IsoQuant *.discovered_transcript_tpm.tsv")
    iq.add_argument("--out", required=True, help="output GTF")
    return p


def setup(args):
    out = Path(args.out)
    if (out / "transnet.log").exists():
        raise SystemExit(f"{out} already contains a TransNet run; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    args.work = str(out / "work")
    handlers = [logging.FileHandler(out / "transnet.log"), logging.StreamHandler(sys.stderr)]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers)
    log.info("TransNet %s: %s", __version__, " ".join(sys.argv))
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", args.device)
    torch.set_num_threads(max(1, args.threads))
    return out


def finish(args, out, started):
    if not args.keep_work:
        shutil.rmtree(out / "work", ignore_errors=True)
    log.info("Done in %.1f min. Outputs in %s", (time.time() - started) / 60, out)


def write_scores(out, name, item, table, model_name):
    table.to_csv(out / f"{name}.scores.tsv", sep="\t", index=False)
    scores = {r.transcript_id: (r.score, r.status) for r in table.itertuples()}
    gtf.write_scored_gtf(item["gtf"], out / f"{name}.scored.gtf", scores, model_name)


def predict(args):
    started = time.time()
    out = setup(args)
    gtfs = gtf_inputs(args.gtf)
    model, data, path = load_model(args.model, args.device)
    args.reads = data["reads"]
    log.info("Model %s (%s reads)", path, args.reads)
    chromosomes = args.chromosomes
    if not chromosomes:
        present = set(features.bam_chromosomes(args.bam))
        seen = {r["chr"] for p in gtfs.values() for r in gtf.read_gtf(p).values()}
        chromosomes = sorted(seen & present)
    prepared = prepare(args, gtfs, chromosomes)
    tables = {}
    for name, item in prepared.items():
        scores = predict_graphs(model, item["graphs"], args.device, args.batch_size, args.pair_budget)
        tables[name] = transcript_table(item, scores)
        write_scores(out, name, item, tables[name], path.stem)
        log.info("[%s] %d candidates scored, %d without a graph path (score 0)", name,
                 int((tables[name].status == "scored").sum()), int((tables[name].status != "scored").sum()))
    if args.reference:
        from transnet.evaluate import evaluate

        log.info("\n%s", evaluate(tables, out, Path(args.bam).name).to_string(index=False))
    finish(args, out, started)


def train(args):
    from transnet.evaluate import evaluate
    from transnet.fit import fit as fit_scaling
    from transnet import split
    from transnet.train import train as run_training

    started = time.time()
    out = setup(args)
    gtfs = gtf_inputs(args.gtf)
    if args.val_fraction is not None:
        if not args.chromosomes or args.train_chroms or args.val_chroms:
            raise SystemExit("--val-fraction needs --chromosomes and no --train/--val-chroms")
        chromosomes = args.chromosomes
    else:
        if not args.train_chroms or not args.val_chroms:
            raise SystemExit("Give --train-chroms and --val-chroms (or --chromosomes with --val-fraction)")
        chromosomes = args.train_chroms + args.val_chroms
    prepared = prepare(args, gtfs, chromosomes)
    everything = [g for item in prepared.values() for g in item["graphs"]]
    if args.val_fraction is not None:
        train_set, val_set = split.by_components(everything, args.val_fraction, args.split_seed)
    else:
        train_set, val_set = split.by_chromosome(everything, args.train_chroms, args.val_chroms)
    by_gtf = lambda graphs: {n: [g for g in graphs if g.source_gtf == n] for n in prepared}
    log.info("Split: %d training graphs, %d validation graphs", len(train_set), len(val_set))
    fit = fit_scaling(train_set)
    model, history, best_step, best = run_training(by_gtf(train_set), by_gtf(val_set), fit, args, args.device)
    import pandas as pd

    pd.DataFrame(history).to_csv(out / "training_history.tsv", sep="\t", index=False)
    metadata = dict(
        transnet_version=__version__, bam=str(args.bam), gtfs=gtfs, reference=args.reference,
        best_step=best_step, validation_bce=best,
        split=dict(train_chroms=args.train_chroms, val_chroms=args.val_chroms, chromosomes=args.chromosomes,
                   val_fraction=args.val_fraction, split_seed=args.split_seed),
        settings={k: getattr(args, k) for k in ("updates", "validate_every", "lr", "weight_decay", "seed",
                                               "batch_size", "pair_budget", "max_nodes", "library_type", "insert_size")},
        insert_size_estimate=args.insert_size_estimate,
    )
    save_model(out / "model.pt", model, fit, args.reads, metadata)
    log.info("Saved %s (best validation BCE %.4f at update %d)", out / "model.pt", best, best_step)
    # Validation report: score each GTF's validation candidates with the best model.
    folder = out / "validation"
    folder.mkdir()
    val_ids = {id(g) for g in val_set}
    tables = {}
    for name, item in prepared.items():
        graphs = [g for g in item["graphs"] if id(g) in val_ids]
        scores = predict_graphs(model, graphs, args.device, args.batch_size, args.pair_budget)
        keep = {t for g in graphs for t in g.transcript_ids}
        table = transcript_table(dict(item, records={k: v for k, v in item["records"].items() if k in keep}), scores)
        table.to_csv(folder / f"{name}.scores.tsv", sep="\t", index=False)
        tables[name] = table
    log.info("\n%s", evaluate(tables, folder, "validation").to_string(index=False))
    log.info("Model metadata: %s", json.dumps(metadata, default=str))
    finish(args, out, started)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "isoquant-cov":
        gtf.add_isoquant_cov(args.gtf, args.tpm, args.out)
        return
    try:
        (predict if args.command == "predict" else train)(args)
    except Exception:
        log.exception("TransNet failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
