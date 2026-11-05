"""Feature extraction: BAM filtering, splice-graph export and junction quality.

For each candidate GTF the modified Scallop2 exporter (``exporter/``) builds
splice graphs from the BAM, guided only by that GTF's candidates, and writes
four tables with prefix ``features``:

* ``features.node.csv``       graph nodes (partial exons) and nine node features
* ``features.edge.csv``       graph edges with weight and genomic gap length
* ``features.path.label.csv`` one row per candidate path through a graph
* ``features.phasing.csv``    phasing fragments (read paths) with read counts

``junction_quality`` then streams the BAM once more and adds nine read-level
quality fields for every splice edge (``quality.csv``).
"""

from collections import Counter
import csv
import logging
from pathlib import Path
import re
import shutil
import subprocess

from transnet.model import QUALITY_FIELDS

log = logging.getLogger("transnet")
PACKAGE = Path(__file__).resolve().parent.parent
LIBRARY_TYPES = ("first", "second", "unstranded")


def _run(command, logfile):
    log.info("Running: %s", " ".join(map(str, command)))
    with open(logfile, "a") as stream:
        result = subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {logfile}")


def samtools():
    path = shutil.which("samtools")
    if not path:
        raise RuntimeError("samtools not found on PATH (install the TransNet conda environment)")
    return path


def exporter():
    path = PACKAGE / "bin" / "transnet-exporter"
    if not path.exists():
        raise RuntimeError(f"{path} is missing; build it once with: bash exporter/build.sh")
    return path


def bam_chromosomes(bam):
    header = subprocess.check_output([samtools(), "view", "-H", str(bam)], text=True)
    return [
        dict(part.split(":", 1) for part in line.split("\t")[1:])["SN"]
        for line in header.splitlines()
        if line.startswith("@SQ\t")
    ]


def filter_bam(bam, chromosomes, work, threads, logfile):
    """Primary, mapped alignments (-F 2308) on the requested chromosomes, indexed."""
    references = bam_chromosomes(bam)
    missing = set(chromosomes) - set(references)
    if missing:
        raise ValueError("Chromosomes absent from the BAM header: " + ", ".join(sorted(missing)))
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    # Index a private symlink so a stale or missing user index is never used.
    local = work / "input.bam"
    if local.is_symlink() or local.exists():
        local.unlink()
    local.symlink_to(Path(bam).resolve())
    t = str(max(1, threads - 1))
    _run([samtools(), "index", "-@", t, "-o", work / "input.bam.bai", local], logfile)
    output = work / "primary.bam"
    regions = [c for c in references if c in set(chromosomes)]
    _run([samtools(), "view", "-@", t, "-b", "-F", "2308", "-o", output, local, *regions], logfile)
    _run([samtools(), "index", "-@", t, output], logfile)
    return output


def check_library_type(reads, library_type):
    if reads == "short" and library_type not in LIBRARY_TYPES:
        raise ValueError("Short reads require --library-type {first,second,unstranded}")


def library_arguments(reads, library_type=None, insert_size=None):
    """Exporter library flags. Long reads are always unstranded with insert size
    0/0/0; short reads get explicit bounds (given by the user or estimated)."""
    if reads == "long":
        return ["--library_type", "unstranded", "--insertsize_median", "0",
                "--insertsize_low", "0", "--insertsize_high", "0"]
    check_library_type(reads, library_type)
    if insert_size is None:
        raise ValueError("Short reads need an insert size: --insert-size MEDIAN,LOW,HIGH or an estimate")
    median, low, high = insert_size
    return ["--library_type", library_type, "--insertsize_median", str(median),
            "--insertsize_low", str(low), "--insertsize_high", str(high)]


PREVIEW = re.compile(
    r"preview insertsize: sampled reads = (\d+), isize = (\S+) \+/- (\S+), "
    r"median = (\d+), low = (\d+), high = (\d+)"
)
MIN_PREVIEW_PAIRS = 100


def parse_preview(text):
    """The exporter's (Scallop2's) ``preview insertsize`` line as a dict, or None."""
    rows = PREVIEW.findall(text)
    if len(rows) != 1:
        return None
    pairs, mean, std, median, low, high = rows[0]
    return dict(pairs=int(pairs), mean=float(mean), std=float(std),
                median=int(median), low=int(low), high=int(high))


def estimate_insert_size(bam, library_type, logfile):
    """Paired-end insert size from the exporter's built-in (Scallop2) preview.

    The preview samples up to about 1M read pairs of ``bam`` and reports the
    median and the 0.5% / 99.8% percentiles of the insert size. Raises when it
    finds fewer than 100 pairs (for example single-end reads).
    """
    command = [exporter(), "-i", bam, "--preview", "--library_type", library_type, "--verbose", "1"]
    log.info("Running: %s", " ".join(map(str, command)))
    result = subprocess.run(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(logfile, "a") as stream:
        stream.write(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Insert-size preview failed ({result.returncode}); see {logfile}")
    estimate = parse_preview(result.stdout)
    short = re.search(r"not enough paired-end reads to create the profile \((\d+) collected\)", result.stdout)
    if estimate is None or estimate["pairs"] < MIN_PREVIEW_PAIRS:
        found = int(short.group(1)) if short else (estimate or {}).get("pairs", 0)
        raise ValueError(
            f"Could not estimate the paired-end insert size from the BAM: {found} usable read pairs "
            f"(at least {MIN_PREVIEW_PAIRS} needed; single-end reads have none). "
            "Pass --insert-size MEDIAN,LOW,HIGH explicitly."
        )
    if not 0 < estimate["low"] <= estimate["median"] <= estimate["high"]:
        raise ValueError(f"Insert-size preview gave invalid bounds {estimate}; "
                         "pass --insert-size MEDIAN,LOW,HIGH explicitly.")
    return estimate


def export_graphs(bam, guide, folder, sample, library, logfile):
    """Run the exporter for one candidate guide; tables land in ``folder``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _run(
        [
            exporter(), "-i", bam, "-b", guide,
            "--feature_prefix", folder / "features",
            "--feature_sample", sample,
            "--transcript_bundle_count_file", folder / "assignment_counts.tsv",
            *library,
            "--use_second_alignment", "false",
            "--max_num_cigar", "10000",
            "--verbose", "1",
        ],
        logfile,
    )


# ---------------------------------------------------------------------------
# BAM junction quality
# ---------------------------------------------------------------------------

CIGAR = re.compile(r"(\d+)([MIDNSHP=X])")
REF_CONSUME = set("MDN=X")


def _splice_keys(edges):
    """(chr, donor, acceptor) for every splice edge. Exporter coordinates are 0-based;
    an edge's end_pos - 1 is the acceptor position."""
    keys = set()
    with open(edges, newline="") as f:
        for row in csv.DictReader(f):
            donor, acceptor = int(row["start_pos"]), int(row["end_pos"]) - 1
            if acceptor > donor:
                keys.add((row["chr"], donor, acceptor))
    return keys


def _stream(bam, keys):
    """Accumulate evidence for spliced reads whose N operation matches a graph splice.

    Only N operations flanked by M operations count, matching Scallop2's hit.cc.
    """
    stats = {}
    proc = subprocess.Popen(
        [samtools(), "view", "-F", "2308", str(bam)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1024 * 1024,
    )
    try:
        for line in proc.stdout:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11:
                raise ValueError("Malformed SAM record")
            chromosome, position = parts[2], int(parts[3]) - 1
            if chromosome == "*":
                continue
            cigar = parts[5]
            ops = [(int(n), op) for n, op in CIGAR.findall(cigar)]
            if not ops or "".join(str(n) + op for n, op in ops) != cigar:
                raise ValueError("Malformed CIGAR: " + cigar)
            nm = next((int(t[5:]) for t in parts[11:] if t.startswith("NM:i:")), None)
            aligned = sum(n for n, op in ops if op in "M=XI")
            nm_rate = nm / aligned if nm is not None and aligned else None
            mapq = int(parts[4])
            valid_mapq = mapq != 255  # 255 = MAPQ unavailable
            for i, (n, op) in enumerate(ops):
                if op == "N":
                    key = (chromosome, position, position + n)
                    if 0 < i < len(ops) - 1 and ops[i - 1][1] == "M" and ops[i + 1][1] == "M" and key in keys:
                        left, right = ops[i - 1][0], ops[i + 1][0]
                        anchor = min(left, right)
                        near = (left < 10 and i >= 2 and ops[i - 2][1] in "ID") or (
                            right < 10 and i + 2 < len(ops) and ops[i + 2][1] in "ID"
                        )
                        v = stats.setdefault(key, [0.0] * 10)
                        v[0] += 1
                        v[1] += anchor
                        v[2] += anchor >= 8
                        v[3] += anchor >= 20
                        if valid_mapq:
                            v[4] += mapq
                            v[5] += mapq >= 20
                            v[6] += 1
                        if nm_rate is not None:
                            v[7] += nm_rate
                            v[8] += 1
                        v[9] += near
                    position += n
                elif op in REF_CONSUME:
                    position += n
        _, err = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError("samtools failed: " + err[-1000:])
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.communicate()
    return stats


def _metrics(v):
    if v is None or v[0] == 0:
        return [0.0] * len(QUALITY_FIELDS)
    n, valid = v[0], v[6]
    return [
        int(n),                              # support_count
        v[1] / n,                            # anchor_min_mean
        v[2] / n,                            # anchor_ge8_fraction
        v[3] / n,                            # anchor_ge20_fraction
        v[4] / valid if valid else 0.0,      # mapq_mean
        v[5] / valid if valid else 0.0,      # mapq_ge20_fraction
        valid / n,                           # mapq_valid_fraction
        v[7] / v[8] if v[8] else 0.0,        # nm_rate_mean
        v[9] / n,                            # near_indel_fraction
    ]


def junction_quality(bam, folder):
    """Write ``quality.csv``: nine BAM evidence fields for every exported edge row.

    Edges without matching spliced reads (including non-splice edges) get zeros.
    """
    folder = Path(folder)
    edges = folder / "features.edge.csv"
    stats = _stream(bam, _splice_keys(edges))
    names = ("sample", "chr", "graph_id", "source", "target", "start_pos", "end_pos", *QUALITY_FIELDS)
    output = folder / "quality.csv"
    seen = set()
    with edges.open(newline="") as source, output.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=names)
        writer.writeheader()
        for row in csv.DictReader(source):
            identity = (row["sample"], row["graph_id"], row["source"], row["target"])
            if identity in seen:
                raise ValueError("Duplicate graph edge: " + str(identity))
            seen.add(identity)
            key = (row["chr"], int(row["start_pos"]), int(row["end_pos"]) - 1)
            writer.writerow(
                {**{k: row[k] for k in names[:7]}, **dict(zip(QUALITY_FIELDS, _metrics(stats.get(key))))}
            )
    return output


def assignment_counts(folder):
    """{internal transcript id: (candidate bundles, assigned bundles)} from the exporter."""
    path = Path(folder) / "assignment_counts.tsv"
    if not path.exists():
        return {}
    with path.open() as stream:
        return {
            r["transcript_id"]: (int(r["candidate_bundle_count"]), int(r["assigned_bundle_count"]))
            for r in csv.DictReader(stream, delimiter="\t")
        }
