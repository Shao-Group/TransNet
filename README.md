# TransNet

TransNet scores candidate transcripts assembled from RNA-seq. You give it
an aligned BAM file and one or more candidate GTFs (for example from
StringTie3, Scallop2 or IsoQuant). It returns, for every candidate, the
probability that the candidate's intron chain is correct. Scores can be used
to rank or filter an assembly.

TransNet does not assemble transcripts. It rebuilds each assembler's splice
graphs from the BAM, using only that assembler's candidates as a guide, and
scores those candidates with a graph neural network trained against a
reference annotation.

```
BAM + candidate GTF ──► splice graphs, phasing fragments, junction read quality
                    ──► GATv2 graph encoder + ordered LSTMs + fragment competition
                    ──► score per candidate  ──► TSV + score-annotated GTF (+ AP / PR curves)
```

Contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [Inputs](#3-inputs)
4. [Pretrained models](#4-pretrained-models)
5. [`transnet predict`](#5-transnet-predict)
6. [`transnet train`](#6-transnet-train)
7. [Outputs](#7-outputs)
8. [How the model works](#8-how-the-model-works)
9. [Running on a cluster](#9-running-on-a-cluster)
10. [Repository layout](#10-repository-layout)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Installation

Requirements: Linux x86-64 and conda (or mamba). A CUDA GPU is optional;
prediction works on a CPU, but training is much faster on a GPU.

```bash
git clone <this repository> TransNet && cd TransNet
conda env create -f environment.yml        # PyTorch 2.12, PyG 2.8.1, samtools 1.24, C++ toolchain
conda activate transnet
pip install -e .                           # provides the `transnet` command
bash exporter/build.sh                     # compiles bin/transnet-exporter (about 2 minutes)
transnet --version
```

On a machine with an NVIDIA driver, conda installs the CUDA build of PyTorch
automatically. To force it, add `pytorch-gpu=2.12.0` to `environment.yml`.
`python -c "import torch; print(torch.cuda.is_available())"` shows whether a
GPU is usable.

The exporter in `exporter/` is a modified
[Scallop2](https://github.com/Shao-Group/scallop2) (BSD-3-Clause, see
`exporter/LICENSE.scallop2`). It builds splice graphs guided by candidate
transcripts and writes them as feature tables.

## 2. Quick start

Score StringTie3 and Scallop2 candidates of a short-read sample with the
pretrained short-read model:

```bash
transnet predict \
  --bam sample.bam \
  --gtf stringtie=stringtie.gtf \
  --gtf scallop2=scallop2.gtf \
  --model sr \
  --library-type first \
  --out results/
```

The insert size is estimated from the BAM (§3); add
`--insert-size MEDIAN,LOW,HIGH` to set it yourself.

Add `--reference annotation.gtf` to also get average precision and
precision-recall curves.

Score StringTie3 and IsoQuant candidates of an Oxford Nanopore cDNA sample:

```bash
transnet isoquant-cov --gtf isoquant/OUT.transcript_models.gtf \
  --tpm isoquant/OUT.discovered_transcript_tpm.tsv --out isoquant.gtf
transnet predict --bam sample.bam --gtf stringtie=stringtie.gtf --gtf isoquant=isoquant.gtf \
  --model ont_cdna --out results/
```

Train your own model (chromosomes 1–8 for training, 9–10 for validation):

```bash
transnet train \
  --bam sample.bam --gtf stringtie=stringtie.gtf --gtf scallop2=scallop2.gtf \
  --reference annotation.gtf --reads short \
  --library-type first \
  --train-chroms chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8 --val-chroms chr9,chr10 \
  --out my_model/
transnet predict --model my_model/model.pt ...
```

## 3. Inputs

| Input | Required for | Notes |
|---|---|---|
| BAM (`--bam`) | always | Coordinate-sorted alignments of one sample. Spliced aligners (STAR, HISAT2, minimap2 `-ax splice`) work. TransNet indexes a private copy, so an index is not needed. Only primary mapped alignments are used (`samtools view -F 2308`). |
| Candidate GTF (`--gtf [NAME=]PATH`, repeatable) | always | One file per assembler. Each `transcript` line must carry the assembler's abundance as a `cov "..."` attribute. StringTie and Scallop2 write it already; for IsoQuant use `transnet isoquant-cov` (it stores IsoQuant's TPM as `cov`, the unit the pretrained models were trained with). `NAME` (letters, digits, `_`, `-`) names the output files; by default it is the file name. |
| Reference annotation (`--reference`) | `train`; optional for `predict` | Defines correct candidates (§8). Used only for labels and evaluation, never as a model input. |
| Library type (`--library-type`) | short reads | `first` = fr-firststrand (dUTP, StringTie `--rf`), `second` = fr-secondstrand (StringTie `--fr`), `unstranded`. |
| Insert size (`--insert-size MEDIAN,LOW,HIGH`) | optional, short reads | Fragment length median and plausible range. By default TransNet estimates it once per run with the exporter's built-in Scallop2 preview (`--preview`) on the filtered BAM: it samples up to about one million read pairs and takes the median and the 0.5% / 99.8% percentiles. The estimate and the number of read pairs are written to `transnet.log`, and the same values are used for every GTF of the run. If fewer than 100 read pairs are found (for example single-end reads), TransNet stops and asks for `--insert-size`. Long reads always use 0/0/0 and are never previewed. |

Chromosome names must agree between the BAM, the GTFs and the reference.

Scope: TransNet scores **stranded multi-exon** candidates. Single-exon and
unstranded transcripts are copied to the output GTF with status
`outside_model_scope` and get no score.

### Producing candidate GTFs

The pretrained models were trained on candidates made like this (StringTie
3.0.3, Scallop2 1.1.2, IsoQuant 4.0.0, each run on the same BAM, without a
reference annotation):

```bash
# short reads (library: first -> --rf, second -> --fr, unstranded -> nothing)
stringtie sample.bam -p 8 --rf -o stringtie.gtf
scallop2 -i sample.bam -o scallop2.gtf --library_type first \
  --insertsize_median 278 --insertsize_low 72 --insertsize_high 374 \
  --use_second_alignment false --max_num_cigar 10000
# The --insertsize_* flags may be left out: Scallop2 then estimates them from
# the BAM with its preview, the code TransNet's exporter inherits. To give
# Scallop2 and TransNet identical values (as TransNet-Test does), pass the
# values TransNet logs, or run the exporter preview once
# (bin/transnet-exporter -i sample.bam --preview --library_type first --verbose 1)
# and pass its median/low/high to both.
# long reads
stringtie sample.bam -p 8 -L -o stringtie.gtf
isoquant.py --bam sample.bam --reference genome.fa --data_type nanopore \
  --threads 8 --output isoquant --prefix OUT        # pacbio_ccs for PacBio
```

## 4. Pretrained models

Use `--model NAME` with one of these models. They are stored in
`transnet/pretrained/`.

| Name | Reads | Training sample | Candidate GTFs used in training | Validation AP (per GTF) |
|---|---|---|---|---|
| `sr` | short (Illumina) | SRR307903 | StringTie3, Scallop2 | 0.896, 0.885 |
| `simulation` | short (Polyester simulation) | dataset-test1 sim_1 | StringTie3, Scallop2 | 0.925, 0.947 |
| `pacbio` | long (PacBio) | ENCFF694DIE | StringTie3, IsoQuant | 0.827, 0.853 |
| `ont_cdna` | long (ONT cDNA) | ENCFF263YFG | StringTie3, IsoQuant | 0.803, 0.897 |
| `ont_drna` | long (ONT direct RNA) | ENCFF771DIX | StringTie3, IsoQuant | 0.872, 0.941 |

All models were trained on chromosomes 1–10, with held-out genomic loci on
those chromosomes used for validation. The reference was the GENCODE v49
basic annotation for real data and the simulated (RefSeq-derived) transcripts
for `simulation`. Use the model that matches your
sequencing platform. Validation AP is development data, not a test result.

**Feature safeguards.** The models store, for every input feature, whether it
varied in the training data. When scoring, features that were constant in
training are set to zero, and every standardised feature is clipped to
[-8, 8], so an unusual value in a new dataset cannot push the network far
outside the range it was trained on. `transnet train` records the same
information, so retrained models use the same safeguards.

Each file stores its training provenance:
`python -c "import torch; print(torch.load('transnet/pretrained/sr.pt', weights_only=False)['metadata'])"`.

## 5. `transnet predict`

```
transnet predict --bam BAM --gtf [NAME=]GTF [--gtf ...] --model MODEL --out DIR
                 [--reference GTF] [--chromosomes chr1,chr2,...]
                 [--library-type first|second|unstranded [--insert-size MEDIAN,LOW,HIGH]]
                 [--threads 4] [--device auto] [--max-nodes 10000]
                 [--batch-size 32] [--pair-budget 2000000] [--keep-work]
```

* `--model`: a pretrained name (§4) or the path of a `model.pt` from `transnet train`.
  The read type (short/long) comes from the model. Short-read models need
  `--library-type`; `--insert-size` is optional (estimated from the BAM, §3).
* `--chromosomes`: chromosomes to score. By default, every chromosome that
  appears in a candidate GTF and in the BAM header.
* `--max-nodes`: graphs with more nodes are skipped; their candidates get
  score 0 with status `graph_too_large`. `0` means no limit. The default
  (10,000) excluded no graph in any dataset of the paper.
* `--batch-size`, `--pair-budget`: memory controls. Lower them if the GPU runs
  out of memory; scores do not depend on them (up to floating-point rounding).
* `--keep-work`: keep the intermediate files (§7).

Each GTF is processed independently. Several GTFs in one call share the BAM
filtering step.

## 6. `transnet train`

```
transnet train --bam BAM --gtf [NAME=]GTF [--gtf ...] --reference GTF --reads short|long --out DIR
               (--train-chroms LIST --val-chroms LIST | --chromosomes LIST --val-fraction F [--split-seed 42])
               [--library-type ... [--insert-size ...]]
               [--updates 10400] [--validate-every 208] [--lr 3e-4] [--weight-decay 1e-4] [--seed 42]
               [--threads 4] [--device auto] [--max-nodes 10000] [--batch-size 32] [--pair-budget 2000000]
               [--keep-work]
```

**Labels.** A candidate is positive when its stranded intron chain equals
the intron chain of a reference transcript (§8).

**Split.** Validation selects the checkpoint, so it must not overlap the
training loci. Choose one of:

* `--train-chroms` / `--val-chroms` (default): whole chromosomes per role.
* `--chromosomes` with `--val-fraction F`: genomic loci are grouped into
  overlap components (overlapping graphs from any GTF stay together), and a
  seeded random fraction `F` of the components is held out. The paper models
  used this kind of locus hold-out within chromosomes 1–10.

Labels are never used to make the split. Keep chromosomes you want to
report results on out of training and validation.

**Objective.** Weighted binary cross-entropy. With N candidate GTFs, each GTF
contributes 1/N of the loss, shared equally by its unique transcripts. Each
GTF is a separate forward pass through the same shared model.

**Optimisation.** Adam (learning rate 3e-4, weight decay 1e-4) with cosine
decay to 1e-6 over `--updates` updates. Each update draws one batch per GTF.
Every `--validate-every` updates the validation loss is computed; the
weights with the lowest validation loss are saved. The feature
standardisation and safeguards (§4) are fitted on the training graphs only
and saved inside `model.pt`. The defaults are the settings of the paper models. On one A100
GPU, a training run took 20–70 minutes after feature extraction.

## 7. Outputs

`transnet predict --out DIR`:

| File | Content |
|---|---|
| `NAME.scores.tsv` | One row per eligible candidate of GTF `NAME`, in GTF order (columns below). |
| `NAME.scored.gtf` | The input GTF, unchanged except for the attributes `transnet_score`, `transnet_status` and `transnet_model` on every transcript and exon line. |
| `evaluation.tsv` | With `--reference`: average precision of the TransNet score and of the native abundance, per GTF. |
| `pr_curves.pdf` | With `--reference`: precision-recall curves for both rankings. |
| `transnet.log` | Full log: command, versions, counts, timing. |
| `work/` | Only with `--keep-work`: `bam/primary.bam` (filtered BAM) and per GTF `NAME/candidates.gtf` (guide), `features.{node,edge,path.label,phasing}.csv` (splice graphs), `quality.csv` (junction read quality), `assignment_counts.tsv`. |

`NAME.scores.tsv` columns:

| Column | Meaning |
|---|---|
| `transcript_id` | ID from the input GTF |
| `chr`, `strand`, `start`, `end`, `exons` | Candidate coordinates (1-based, inclusive) and exon count |
| `native_abundance` | The candidate's `cov` from the input GTF |
| `score` | TransNet probability in [0, 1] |
| `status` | `scored`, or why the candidate has no score (score is then 0): `no_read_bundle` (no read cluster overlaps it), `bundle_assignment_rejected` (the read cluster does not support it), `assigned_without_exported_path`, `not_exported`, or `graph_too_large` |
| `label` | With `--reference`: 1 if the intron chain matches the reference |

A candidate that appears in several graphs gets the mean of its scores.
Unscored candidates keep score 0 so that rankings and AP cover every
candidate of the GTF. This is how the paper evaluated them.

`transnet train --out DIR` writes `model.pt` (weights, feature
standardisation, read type and training metadata), `training_history.tsv`
(training loss and validation loss at each validation), `transnet.log`, and
`validation/` with `NAME.scores.tsv`, `evaluation.tsv` and `pr_curves.pdf`
for the validation loci.

## 8. How the model works

**Splice graphs.** For each candidate GTF, the exporter groups reads into
bundles and builds a splice graph per bundle. Nodes are partial exons; edges
join adjacent partial exons or span introns. Each candidate assigned to a
bundle is a path through its graph. Phasing fragments are the node paths
supported by single reads or read pairs, with read counts. Node features
(9): read weight, length, maximum coverage, coverage standard deviation,
indel coverage, indel ratio, left/right indel, guide-added length. Edge
features (2): read weight and genomic gap length. Junction quality (9, per
splice edge, from the BAM): supporting reads, mean minimum anchor and the
fractions of anchors ≥ 8 and ≥ 20 bp, mean MAPQ and the fractions with MAPQ
≥ 20 and with valid MAPQ, mean NM edit rate, and the fraction with an indel
near the junction.

**Network.** A two-head GATv2 layer embeds the nodes. One LSTM reads each
candidate's node sequence. Another reads its splice junctions; each junction
token combines the attention-weighted endpoint embeddings, the edge features
and an encoding of the junction quality. Phasing fragments are encoded the
same way. Over three rounds, each fragment's read count is divided by a
softmax among the candidates that fully contain it, plus a "none of them"
option. Each candidate updates its state (GRU) from the evidence it
received, from fragments that only overhang it, and from conflicting ones.
A final MLP outputs the logit. Identical candidates compete once and share
a score. The native abundance is one input feature. Genomic coordinates,
sequence and assembler identity are not inputs.

**Correctness.** A candidate is correct when its strand and complete intron
chain equal those of a reference transcript; transcript ends may differ.
Recall in the evaluation is measured over the correct candidates present in
each GTF.

## 9. Running on a cluster

Example Slurm script (adapt account, partition and paths):

```bash
#!/bin/bash
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=06:00:00
source activate transnet
transnet predict --bam sample.bam --gtf stringtie=stringtie.gtf --gtf scallop2=scallop2.gtf \
  --model sr --library-type first --threads 8 --out results/
```

Resources: feature extraction streams the BAM a few times and uses one CPU
core per GTF step. A 5 GB short-read BAM restricted to 14 chromosomes needs
roughly 30–60 minutes. Memory: 16–64 GB depending on the BAM and graph sizes.

## 10. Repository layout

```
TransNet/
├── README.md              this file
├── environment.yml        conda environment (runtime + exporter build)
├── pyproject.toml         Python package (`pip install -e .`)
├── transnet/
│   ├── cli.py             `transnet predict | train | isoquant-cov`
│   ├── run.py             shared steps: extraction, scoring, score tables, model files
│   ├── features.py        BAM filtering, exporter call, junction-quality extraction
│   ├── graphs.py          feature tables -> validated PyTorch Geometric graphs, labels
│   ├── gtf.py             GTF reading, guide writing, scored-GTF writing
│   ├── model.py           the TransNet network
│   ├── fit.py             training-only feature standardisation
│   ├── split.py           training/validation split
│   ├── train.py           training loop and loss weighting
│   ├── evaluate.py        average precision and PR curves
│   └── pretrained/        sr, simulation, pacbio, ont_cdna, ont_drna (*.pt)
├── exporter/              modified Scallop2 source + build.sh
├── bin/                   transnet-exporter (created by exporter/build.sh)
└── dev/                   how the pretrained files were made and verified (not needed to run TransNet)
```

### Reproducibility

Prediction with a given model and given feature tables is deterministic up
to GPU floating-point rounding. The published weights give bit-identical
logits to the research checkpoints (`dev/`). The splice-graph exporter
inherits from Scallop2 a graph library that orders edges by memory address.
As a result, about 0.5% of graph edges (mostly edges without read-path
support) can differ between builds of the exporter, or even with unrelated
input changes such as file paths. On the SRR307911 test set this changed AP
by at most 0.0001; about 1% of candidates' scores changed by more than 0.01.
One build run twice on the same inputs gives identical results.

## 11. Troubleshooting

* **`bin/transnet-exporter is missing`**: run `bash exporter/build.sh` inside
  the activated environment.
* **`... has no 'cov' attribute`**: the GTF lacks abundances. For IsoQuant use
  `transnet isoquant-cov`. For other tools, add the abundance the tool
  reports as `cov "..."` on transcript lines.
* **`Chromosomes absent from the BAM header`**: GTF and BAM chromosome names
  differ (for example `chr1` and `1`). Rename one of them, or restrict with
  `--chromosomes`.
* **CUDA out of memory**: lower `--batch-size` and/or `--pair-budget`, or use
  `--max-nodes`.
* **Many `no_read_bundle` candidates**: the BAM and the GTF may come from
  different samples, or the GTF was assembled on another BAM.
