# TransNet

TransNet scores candidate transcripts from RNA-seq assemblies using an aligned
BAM file and one or more candidate GTFs. It produces a score table and a
score-annotated GTF for each input assembly. Higher scores indicate stronger
support for a correct transcript intron chain.

TransNet scores already assemled transcripts, it does not assemble
transcripts. Pretrained models are provided for Illumina, PacBio and Oxford
Nanopore data.

## Installation

Requirements: Anaconda or mamba. Prediction supports CPU and
CUDA GPU; a GPU is recommended for training.

```bash
git clone https://github.com/Shao-Group/TransNet.git
cd TransNet
conda env create -f environment.yml
conda activate transnet
pip install -e .
bash exporter/build.sh
transnet --version
```

Run subsequent commands in the activated `transnet` environment. For GPU use,
check availability with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Inputs

| Input | Requirement |
|---|---|
| `--bam sample.bam` | Coordinate-sorted RNA-seq alignments for one sample. A BAM index is created automatically without modifying the input. |
| `--gtf NAME=candidates.gtf` | Candidate transcripts assembled from that BAM. Repeat for multiple assemblies, keeping each assembler's GTF separate. Each transcript record must have its native abundance in a `cov "..."` attribute. |
| `--reference annotation.gtf` | Required for training; optional for evaluating predictions. |
| `--library-type` | Required for short reads: `first` (fr-firststrand/dUTP), `second` (fr-secondstrand), or `unstranded`. |
| `--insert-size MEDIAN,LOW,HIGH` | Optional for short reads; estimated from the BAM by default. Supply explicitly if estimation fails, including for single-end reads. |

Chromosome names must match across the BAM, candidate GTFs and reference
(for example, all use `chr1`, rather than mixing `chr1` and `1`). Each `--gtf`
can also be given as a path alone; `NAME` controls the output filenames.
Output directories must be absent or empty.

StringTie and Scallop2 GTFs already contain `cov`. For IsoQuant, first convert
its transcript TPM values into this attribute:

```bash
transnet isoquant-cov \
  --gtf isoquant/OUT.transcript_models.gtf \
  --tpm isoquant/OUT.discovered_transcript_tpm.tsv \
  --out isoquant.gtf
```

## Choose a pretrained model

Use the model matching your sequencing platform:

| `--model` | Data type |
|---|---|
| `sr` | Illumina short reads |
| `pacbio` | PacBio long reads |
| `ont_cdna` | Oxford Nanopore cDNA |
| `ont_drna` | Oxford Nanopore direct RNA |
| `simulation` | Simulated short reads (Polyester) |

You can also provide the path to a `model.pt` produced by `transnet train`.

## Score transcripts

For short reads, score one or more candidate assemblies:

```bash
transnet predict \
  --bam sample.bam \
  --gtf stringtie=stringtie.gtf \
  --gtf scallop2=scallop2.gtf \
  --model sr \
  --library-type first \
  --out results/
```

For Oxford Nanopore cDNA, using the converted IsoQuant GTF:

```bash
transnet predict \
  --bam sample.bam \
  --gtf stringtie=stringtie.gtf \
  --gtf isoquant=isoquant.gtf \
  --model ont_cdna \
  --out results/
```

Long-read models do not require library-type or insert-size settings. Add
`--reference annotation.gtf` to either command to produce average precision
(AP) and precision-recall curves alongside the scores.

Common options:

| Option | Purpose |
|---|---|
| `--chromosomes chr11,chr12` | Restrict prediction to selected chromosomes. By default, score chromosomes shared by the candidate GTFs and BAM. |
| `--threads 8` | Set CPU threads (default: 4). |
| `--device cpu` or `--device cuda` | Choose a device; the default selects a GPU when available. |
| `--batch-size 16` | Reduce from the default of 32 to lower prediction memory use. |
| `--pair-budget 1000000` | Reduce from the default of 2000000 to further limit batch memory. |
| `--max-nodes 10000` | Skip larger graphs and assign their candidates score 0; use `0` for no limit. |
| `--keep-work` | Retain intermediate files for inspection. |

## Train a model

Provide a reference annotation and separate training and validation
chromosomes. For example:

```bash
transnet train \
  --bam sample.bam \
  --gtf stringtie=stringtie.gtf \
  --gtf scallop2=scallop2.gtf \
  --reference annotation.gtf \
  --reads short \
  --library-type first \
  --train-chroms chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8 \
  --val-chroms chr9,chr10 \
  --out my_model/
```

For long reads, use `--reads long` and omit `--library-type`. Alternatively,
replace the chromosome split options with
`--chromosomes chr1,chr2,chr3 --val-fraction 0.2` to hold out 20% of genomic
loci for validation. Overlapping loci from different GTFs stay in the same
split. Keep any chromosomes intended for test evaluation out of both splits.

Training saves the model with the lowest validation loss. Defaults are
10,400 updates with validation every 208 updates; use `--updates` and
`--validate-every` to change them.

Use the trained model for prediction:

```bash
transnet predict \
  --bam sample.bam \
  --gtf stringtie=stringtie.gtf \
  --model my_model/model.pt \
  --library-type first \
  --chromosomes chr11,chr12 \
  --out predictions/
```

## Outputs

Prediction writes the following files to `--out`:

| File | Contents |
|---|---|
| `NAME.scores.tsv` | One row per eligible candidate on the selected chromosomes, in input GTF order. |
| `NAME.scored.gtf` | Input GTF annotated with `transnet_score`, `transnet_status` and `transnet_model`. |
| `evaluation.tsv` | With `--reference`: AP for TransNet scores and native abundance, per assembly. |
| `pr_curves.pdf` | With `--reference`: precision-recall curves for both rankings. |
| `transnet.log` | Run settings, progress and diagnostic messages. |
| `work/` | Intermediate BAM and feature files, retained with `--keep-work`. |

The score table contains:

| Column | Meaning |
|---|---|
| `transcript_id` | Original transcript ID. |
| `chr`, `strand`, `start`, `end`, `exons` | Coordinates (1-based, inclusive) and exon count. |
| `native_abundance` | The candidate's input `cov` value. |
| `score` | Model score between 0 and 1, suitable for ranking candidates. |
| `status` | `scored`, or the reason for a zero score: `no_read_bundle`, `bundle_assignment_rejected`, `assigned_without_exported_path`, `not_exported`, or `graph_too_large`. |
| `label` | With `--reference`: 1 when the candidate's strand and complete intron chain match a reference transcript; transcript ends may differ. |

Eligible candidates that cannot be scored remain in the table with score 0.
Single-exon, unstranded and unselected-chromosome transcripts remain in the
output GTF with status `outside_model_scope` and no score. Evaluation covers
eligible candidates on the selected chromosomes; recall is measured over
reference-matching candidates present in the input assembly.

Training writes `model.pt`, `training_history.tsv`, `transnet.log`, and a
`validation/` folder containing scores, AP and precision-recall curves.
Training validation reports cover candidates in retained validation graphs;
use `predict` on held-out chromosomes for evaluation including unmapped
candidates.

## Troubleshooting

- **Exporter missing:** activate the environment and run `bash exporter/build.sh` from the repository root.
- **Missing `cov`:** use `transnet isoquant-cov` for IsoQuant, or add the assembler's native abundance to each transcript record.
- **Chromosome mismatch:** check naming in the BAM, GTFs and reference, or select matching chromosomes with `--chromosomes`.
- **Insert-size estimation failed:** provide `--insert-size MEDIAN,LOW,HIGH` for your library.
- **GPU out of memory:** lower `--batch-size` and `--pair-budget`, or use `--device cpu`.

For the full option list:

```bash
transnet predict --help
transnet train --help
transnet isoquant-cov --help
```
