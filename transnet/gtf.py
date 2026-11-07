"""GTF reading, candidate inventory and score-annotated GTF output."""

import gzip
import math
import os
from pathlib import Path
import re
import tempfile

ATTRIBUTE = re.compile(r'(\w+)\s+"([^"\n]*)"')


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def read_gtf(path, chromosomes=None):
    """Transcripts keyed by transcript_id: chr, strand, merged exons and ``cov``.

    ``cov`` is the assembler's native abundance from the transcript record
    (StringTie/Scallop2 coverage, IsoQuant TPM), or None when absent.
    """
    records = {}
    with _open(path) as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) != 9:
                raise ValueError(f"Malformed GTF line in {path}")
            if f[2] not in ("exon", "transcript") or (
                chromosomes is not None and f[0] not in chromosomes
            ):
                continue
            attrs = dict(ATTRIBUTE.findall(f[8]))
            tid = attrs.get("transcript_id")
            if not tid:
                raise ValueError(f"Missing transcript_id in {path}")
            start, end = int(f[3]), int(f[4])
            if start < 1 or end < start:
                raise ValueError(f"Invalid GTF coordinates for {tid}")
            d = records.setdefault(tid, {"chr": f[0], "strand": f[6], "exons": [], "cov": None})
            if (d["chr"], d["strand"]) != (f[0], f[6]):
                raise ValueError(f"Transcript spans chromosomes/strands: {tid}")
            if f[2] == "exon":
                d["exons"].append((start, end))
            elif "cov" in attrs:
                value = float(attrs["cov"])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid cov for {tid}")
                d["cov"] = value
    for d in records.values():
        merged = []
        for start, end in sorted(set(d["exons"])):
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        d["exons"] = merged
    return records


def chain_key(d):
    """Stranded intron chain; a candidate is correct when it matches a reference chain."""
    return (
        d["chr"],
        d["strand"],
        tuple((a[1], b[0] - 1) for a, b in zip(d["exons"], d["exons"][1:])),
    )


def eligible(d):
    """The model scores stranded multi-exon transcripts only."""
    return len(d["exons"]) > 1 and d["strand"] in ("+", "-")


def reference_chains(path, chromosomes):
    return {chain_key(r) for r in read_gtf(path, chromosomes).values() if len(r["exons"]) > 1}


def candidates(path, chromosomes):
    """Eligible candidates of one assembler GTF, in file order.

    Returns {internal_id: record}; each record keeps ``original_id``. Internal
    IDs (candidate_00000000, ...) avoid any character issues in the exporter.
    """
    result = {}
    for tid, d in read_gtf(path, chromosomes).items():
        if not eligible(d):
            continue
        if d["cov"] is None:
            raise ValueError(
                f"Transcript {tid} in {path} has no 'cov' attribute; TransNet needs the "
                "assembler's native abundance on each transcript line"
            )
        result[f"candidate_{len(result):08d}"] = {**d, "original_id": tid}
    return result


def write_guide(path, records):
    """Write the candidate guide GTF read by the feature exporter (-b)."""
    with Path(path).open("w") as stream:
        for tid, d in records.items():
            attrs = f'gene_id "{tid}"; transcript_id "{tid}";'
            stream.write(
                f"{d['chr']}\tguide\ttranscript\t{d['exons'][0][0]}\t{d['exons'][-1][1]}\t.\t"
                f"{d['strand']}\t.\t{attrs} cov \"{d['cov']:.8f}\";\n"
            )
            for i, (start, end) in enumerate(d["exons"], 1):
                stream.write(
                    f"{d['chr']}\tguide\texon\t{start}\t{end}\t.\t{d['strand']}\t.\t"
                    f'{attrs} exon_number "{i}";\n'
                )


def write_scored_gtf(original, destination, scores, model_name):
    """Copy ``original`` and append TransNet attributes to every transcript/exon line.

    scores: {original transcript_id: (score or None, status)}. Transcripts not in
    ``scores`` are outside the model scope (single-exon, unstranded, or on
    chromosomes that were not scored) and receive only a status.
    """
    destination = Path(destination)
    if Path(original).resolve() == destination.resolve():
        raise ValueError("Refusing to overwrite the input GTF")
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=destination.name + ".")
    try:
        with _open(original) as src, os.fdopen(fd, "w") as dst:
            for line in src:
                fields = line.rstrip("\r\n").split("\t")
                if line.startswith("#") or len(fields) != 9:
                    dst.write(line)
                    continue
                tid = dict(ATTRIBUTE.findall(fields[8])).get("transcript_id")
                if tid is None:
                    dst.write(line)
                    continue
                score, status = scores.get(tid, (None, "outside_model_scope"))
                extra = f'transnet_score "{score:.6g}"; ' if score is not None else ""
                extra += f'transnet_status "{status}"; transnet_model "{model_name}";'
                attrs = fields[8].rstrip()
                fields[8] = attrs + ("" if attrs.endswith(";") else ";") + " " + extra
                dst.write("\t".join(fields) + "\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def add_isoquant_cov(gtf_path, tpm_path, output):
    """Copy an IsoQuant GTF, setting each transcript's ``cov`` to its reported TPM.

    The pretrained long-read models were trained with IsoQuant TPM as the
    native abundance of IsoQuant candidates.
    """
    output = Path(output)
    if output.resolve() in (Path(gtf_path).resolve(), Path(tpm_path).resolve()):
        raise ValueError("Refusing to overwrite an IsoQuant input file")
    abundance = {}
    for line in Path(tpm_path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if parts[0] in ("feature_id", "transcript_id"):
            continue
        if len(parts) != 2:
            raise ValueError("Expected a single-sample IsoQuant TPM table (id, TPM)")
        value = float(parts[1])
        if parts[0] in abundance or not math.isfinite(value) or value < 0:
            raise ValueError("Duplicate or invalid TPM for " + parts[0])
        abundance[parts[0]] = value
    fd, temporary = tempfile.mkstemp(dir=output.parent, prefix=output.name + ".")
    try:
        with _open(gtf_path) as src, os.fdopen(fd, "w") as dst:
            for line in src:
                f = line.rstrip("\n").split("\t")
                if line.startswith("#") or len(f) != 9 or f[2] != "transcript":
                    dst.write(line)
                    continue
                match = re.search(r'transcript_id "([^"]+)"', f[8])
                if not match or match[1] not in abundance:
                    raise ValueError("No TPM for IsoQuant transcript " + (match[1] if match else "?"))
                f[8] = re.sub(r'\bcov\s+"[^"]*"\s*;?', "", f[8]).rstrip() + ' cov "' + str(abundance[match[1]]) + '";'
                dst.write("\t".join(f) + "\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
