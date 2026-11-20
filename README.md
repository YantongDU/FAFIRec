# FAFIRec: diagnostic-guided alignment and fusion

This repository implements the revised, zero-initialized **linear diagnostic
controller** for sparse-interaction recommendation. It includes train-only
teacher diagnostics, detached user-weighted MK-MMD, conflict-conditioned fusion,
matched controls, exact chunked catalog ranking, checkpoint replay, and paired
multi-seed analysis. Historical code is preserved under `legacy/`.

This is an implementation of the revised method, **not a reproduction of the
historical paper tables**. No benchmark performance claim follows from the
included smoke configuration. See [method correspondence](docs/METHOD_IMPLEMENTATION.md)
and [中文运行说明](docs/RUNNING_zh.md).

## Literature-guided independent residual experiments

The new, separate `research_ml1m.py` workflow freezes the validation-selected
strong CF model, adds an independent ID/text expert, and combines their scores.
It supports a zero-residual fallback, training-seen centered text adaptation,
optional same-ID contrastive alignment, masked full-catalog CE, and a second,
frozen-expert relative router. It is an experimental redesign, not a faithful
reproduction of PAD, LLM-ESR, RLMRec or IISAN. See
[verified sources and design (中文)](docs/LITERATURE_GUIDED_DESIGN_zh.md).

`configs/ml1m_research_pilot.json` declares six validation-only candidates.
`evaluate_research.py` separately replays a chosen checkpoint on validation/test;
`analyze_research.py` checks artifact identities and independently recomputes
metrics from the complete saved rankings. The local pilot references the frozen
previous main/CF-search run directories, including their data and feature
manifests. These private artifacts are not bundled; preserve them with the run.
The previously observed test split is an iterative benchmark, not an untouched
confirmation set. No performance claim follows from code or smoke tests alone.
See [运行与核验步骤](docs/RESEARCH_RUNNING_zh.md) for the isolated five-seed
workflow, prerequisite artifacts and common validation selection.

The [completed literature-guided results (中文)](docs/ML1M_RESEARCH_RESULTS_zh.md)
include all controls, an additional single-branch CF-CE baseline, seed uncertainty
and negative findings. The validation-selected text/relative model reached
test Recall@10/NDCG@10 of 0.162811/0.161769, compared with 0.135447/0.135793
for the previous strong CF and 0.161237/0.159959 for single-branch CF-CE.
Most of the improvement comes from the classification training recipe;
the router adds little, and some cold-user NDCG metrics remain below CF-CE.
All 35 new selected checkpoints passed independent ranking replays and audits.
An [Overleaf-ready table](docs/ML1M_RESEARCH_TABLE.tex) preserves every control.

## ACE-SASRec (SIGIR 2026) external baseline

The separate `baselines/ace2026_model.py` implementation preserves ACE's
uncentered spectral initialization, global scaling, trainable item table and
raw-dot SASRec objective. `tools/ace2026_experiment.py` reuses the frozen ML-1M
inputs and evaluation; its parameters are initialized independently of CF weights.
Validation-only tuning, five isolated seeds and independent ranking replay are
described in the [running guide (中文)](docs/ACE2026_RUNNING_zh.md).
Architecture, loss and tuning-budget differences are explicit in the
[method correspondence](docs/ACE2026_BASELINE_zh.md). The
[paper-reported baseline reference](docs/ACE_PAPER_BASELINE_REFERENCE_zh.md)
keeps the original ML-20M and other datasets separate from local ML-1M results.

The [completed five-seed comparison (中文)](docs/ACE2026_RESULTS_zh.md) reports
ACE Recall@10/NDCG@10/NDCG@50 of 0.169937/0.169444/0.252323, versus
0.162811/0.161769/0.241545 for the validation-selected FAFIRec text residual.
ACE has higher means in this experiment. See the
[machine-readable seed results](docs/ACE2026_RESULTS.json) and
[Overleaf table](docs/ACE2026_TABLE.tex) with its
[bibliography entry](docs/ACE2026_REFERENCE.bib).

## Diagnostic-guided integration: coverage and calibration controls

The isolated `experiments/diagnostic_v4/` workflow revisits the integration
motivation with all-transition training, an independent collaborative relation
branch, train-internal diagnostic proxies, fixed score calibration and optional
identity-conditioned relation alignment. ACE and EASE remain credited existing
components. This redesign has different equations from the manuscript's MK-MMD
model; improvements must be separated from diagnostic novelty.

The [design and sources (中文)](docs/DIAGNOSTIC_V4_DESIGN_zh.md) describe the
coverage issue, causal label boundaries, seven fixed recipe controls and an
additional predeclared constant-scale control. The final
[five-seed results](docs/DIAGNOSTIC_V4_RESULTS_zh.md),
[machine-readable scores](docs/DIAGNOSTIC_V4_RESULTS.json) and
[Overleaf table](docs/DIAGNOSTIC_V4_TABLE.tex) preserve the common validation
choice and all negative findings. Test does not select the recipe.

Run `run_final_components.py run` to prepare the five coverage backbones and
fixed-budget internal teachers, then `complete_five_seed.py` for fixed-recipe
training, a five-seed validation lock and independent ranking replays. Paths
and private prerequisites follow the existing ACE running guide. The small
configuration files contain the actual local artifact paths; use fresh output
directories for a new experiment and keep frozen inputs with its manifests.

## MovieLens-1M: chronological 80/10/10, multiple held-out positives

**Use this workflow for the current ML-1M experiment.** Read the complete
[ML-1M paper-protocol record (中文)](docs/ML1M_PAPER_PROTOCOL_zh.md), including
the declared choices and differences from the historical manuscript.

For the new validation-only transfer pilot, initialization options and replay
commands, see [迁移初始化与验证运行说明](docs/TRANSFER_RUNNING_zh.md) and
[原因审计与固定验证方案](docs/ML1M_IMPROVEMENT_DIAGNOSIS_zh.md).

`configs/ml1m_stable_transfer.json` records the selected initialization and
fine-tuning settings. `verify_ml1m_replays.py` independently checks the complete
selected-checkpoint grid. The separate `configs/ml1m_cf_validation.json` and
`analyze_cf_baseline.py` compare CF learning rates using validation before
reporting the stronger CF baseline; see the running guide for artifact inputs.

The [completed transfer results (中文)](docs/ML1M_TRANSFER_RESULTS_zh.md) report
five seeds, all matched controls and the validation-selected stronger CF.
The updated protocol raised the full model's overall Recall@10/NDCG@10 means
by 2.40%/2.50% relative to the earlier full model, while remaining below the
stronger CF baseline. No overall diagnostic advantage over the matched uniform
control is established. An [Overleaf table](docs/ML1M_TRANSFER_TABLE.tex)
includes both CF learning-rate settings.

This entry point retains each user's complete validation and test blocks.
Recall@K divides by that user's number of relevant items; it is not Hit@K.
NDCG@K uses the corresponding multi-positive ideal ranking. Evaluation excludes
the observed history from the full movie catalog, without restoring targets.
Validation observes training events only; test observes training plus the
complete validation block.

| Workflow | Training / evaluation / analysis | Data and metrics |
|---|---|---|
| Current ML-1M ratio protocol | `train_ml1m.py` / `evaluate_ml1m.py` / `analyze_ml1m.py` | Version2, chronological80/10/10, multi-positive user-macro Recall/NDCG |
| Existing leave-two-out workflow | `training.py` / `evaluate.py` / `analyze_results.py` | Version1, one validation and one test target per user |

**Do not mix these data formats, checkpoints, or analysis commands.** In
particular, `configs/ml1m.json` belongs to the older single-target workflow;
the current ratio experiment uses `configs/ml1m_paper_qwen3.json`.

### Install the ML-1M experiment environment

The pinned ML-1M dependencies require Python 3.11 or newer; the local run uses
Python 3.12.14. Create and activate a dedicated Python environment. Install the appropriate
official PyTorch wheel first; the local GPU experiment uses CUDA12.6:

```bash
python -m venv .venv
# Activate .venv using your operating system's normal command.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements_ml1m.txt
python run_tests.py
```

`requirements_ml1m.txt` pins the training and Qwen encoding dependencies. The
supplied paper-protocol configuration requests CUDA. A CPU experiment needs a
CPU PyTorch installation and an explicit device choice; feature encoding on CPU
also requires `--dtype float32`.

### Prepare data and real semantic features

Obtain and extract the raw [MovieLens-1M release from GroupLens](https://grouplens.org/datasets/movielens/1m/)
before preparation. The input directory must contain `ratings.dat` and
`movies.dat`; `users.dat` availability is recorded, but demographic fields are
not consumed by the revised item-text model.

```bash
python prepare_ml1m_paper.py --input data/raw/ml-1m --output data/ml1m_paper_ratio
```

The adapter retains ratings4/5, orders each user by timestamp with source-order
ties, and splits events as `floor(0.8n)`, `floor(0.1n)`, and the remainder. It
records eligibility exclusions and file hashes. No undocumented subsampling is
introduced to force the historical table's population counts.

Before generating features, obtain the fixed
[official Qwen3-Embedding-0.6B model](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3)
in a local model directory. The explicit download command below retrieves that
official revision and creates the verified asset manifest required by the
generator; it can also verify already downloaded matching assets. Training does
not download model weights.

```bash
python generate_ml1m_text_features.py --model-dir data/models/qwen3_embedding_06b --download-only
python generate_ml1m_text_features.py --model-dir data/models/qwen3_embedding_06b --data-dir data/ml1m_paper_ratio --output data/ml1m_qwen3_06b.pt --device cuda --dtype float16 --batch-size 16 --max-length 128
```

The generator encodes static movie titles/year and genres only, and writes
the feature tensor plus its ordered MovieID/hash manifest. It records the
fixed model revision, pooling, tokenization, truncation and software versions.
This is an explicit Qwen encoder choice, not PAD's original LLM2Vec artifact.
Raw MovieLens files, generated features and LLM weights are local inputs under
Git-ignored `data/`; they are not bundled or submitted to the repository.

### Train five seeds, resume, analyze and reload

```bash
# Seeds2025--2029; CF, uniform fusion and diagnostic control.
python train_ml1m.py --config configs/ml1m_paper_qwen3.json

# Continue the same interrupted experiment.
python train_ml1m.py --config configs/ml1m_paper_qwen3.json --resume

# Run after the complete configured seed/control grid has finished.
python analyze_ml1m.py runs/ml1m_paper_qwen3_v1

# Independently reload one selected checkpoint into a new evaluation directory.
python evaluate_ml1m.py --run-dir runs/ml1m_paper_qwen3_v1 --seed 2025 --control diagnostic --split test --output-dir runs/ml1m_paper_qwen3_reloaded_2025
```

The configuration uses d64, batch2048, Adam/lr0.001, alignment coefficient0.3,
coupled weight decay1e-5, validation NDCG@50, patience10 and at most150 student
epochs. Teacher budget30, sequence length50 and encoder details are declared
completion choices; see the protocol record. Keep the source, input features,
configuration and runtime unchanged when resuming. Use a fresh output directory
for a different experiment.

Results include `analysis/paper_table.csv`, per-user target sets and ranked
recommendation lists, selected/last checkpoints and training/diagnostic logs.
The [completed local five-seed result record (中文)](docs/ML1M_ACTUAL_RESULTS_zh.md)
reports the actual metrics and limitations: the full controller's overall and
cold-user Recall@10/NDCG@10 means were below the CF control under these settings.
All 15 selected checkpoints were independently replayed with identical per-user
outputs. An [Overleaf-ready controlled-comparison table](docs/ML1M_control_table.tex)
is provided for this declared protocol.
The analyzer requires all configured jobs and verifies their identities and
hashes before reporting seed means, sample standard deviations and paired
comparisons. This workflow does **not** claim reproduction of the historical
table, superiority of the current model, or completion of all competing
baselines from the paper.

## Existing single-target workflow: installation

Python 3.10 or newer is required. Local CPU checks use Python 3.12; the CI
workflow is configured for Python 3.11 and the declared dependency versions.

```bash
python -m venv .venv
# Activate .venv using your operating system's normal command.
python -m pip install -r requirements.txt
python run_tests.py
```

For GPU experiments, install a CUDA-enabled PyTorch build appropriate to the
machine before running the project. Explicit `--device cuda` fails if CUDA is
unavailable; `auto` selects an available CUDA device or CPU. CPU and GPU results
are not claimed to be bitwise equal.

## Existing single-target workflow: run the included data

The public Pantry histories have no timestamps. Source order must be explicitly
accepted and is recorded as chronology-unverified.

```bash
python prepare_data.py --source prime-pantry --input dataset --output data/prime_pantry --allow-unverified-order
python training.py --config configs/pantry_lexical_smoke.json
python training.py --config configs/pantry_lexical_smoke.json --resume
python analyze_results.py runs/pantry_lexical_smoke
python evaluate.py --run-dir runs/pantry_lexical_smoke --seed 2025 --control diagnostic --output-dir runs/reloaded_smoke --audit-users 16
```

This configuration intentionally uses signed-hash metadata features, 128 users,
two seeds, and two teacher/student epochs. Its nine controls verify computation
and artifacts. These are **not LLM embeddings or publication-quality benchmark
results**. Output directories are protected from accidental overwrites; choose
`--output-dir` for a new run. `--resume` accepts identical source, data,
configuration and recorded runtime and restores epoch-boundary RNG state.

## Existing single-target workflow: alternative benchmark adapters

The following commands prepare leave-two-out data. For the current ML-1M
80/10/10 protocol, use `prepare_ml1m_paper.py` above instead.

```bash
python prepare_data.py --source movielens1m --input /path/to/ml-1m --output data/ml1m --min-rating 4
python prepare_data.py --source netflix --input /path/to/netflix --metadata /path/to/movie_titles.txt --output data/netflix --min-rating 4
python prepare_data.py --source book-crossing --input /path/to/BX-CSV-Dump --output data/book_crossing --min-rating 6 --allow-unverified-order
```

The rating thresholds above are explicit **examples**, not assertions about the
original paper's preprocessing. Each adapter records filtering counts, source
hashes, ordering, metadata scope and exact leave-two-out protocol. The item
catalog includes supplied static metadata, including zero-training-feedback
items. Book-Crossing lacks timestamps, so this adapter does not establish a
chronological sequential benchmark. It requires a cleaned/complete item catalog
if ratings refer to ISBNs missing from the supplied metadata. See
[data format and adapters](docs/DATA_FORMAT.md) for all input conventions and
generic CSV support. Raw benchmark data are not redistributed or automatically
downloaded.

## Existing single-target workflow: optional LLM2Vec inputs

Training defaults to external embeddings and requires both a tensor and its
`.manifest.json` sidecar with ordered raw item IDs and a matching SHA-256.
Missing or mismatched features fail explicitly; they never silently become
random or lexical features.

If model access and hardware are available, install the official LLM2Vec
dependencies in a suitable feature-generation environment and run:

```bash
python generate_text_features.py --data-dir data/ml1m --model /path/to/base-model --adapter /path/to/llm2vec-adapter --output data/ml1m_llm2vec.pt --device cuda
python training.py --config configs/ml1m.json
python analyze_results.py runs/ml1m
```

The generator uses only the prepared item metadata and writes padding row0 plus
rows in the exact prepared item order. It does not add model access or download
gated weights on the user's behalf. The optional LLM2Vec path is separate from
the CPU-tested experiment dependency set.

Configs for ML-1M, Netflix, Book-Crossing and Pantry use five training seeds and
the same controls. Defaults are starting values; tune using validation only.
Neither the original three benchmark inputs nor their actual LLM features are
bundled with this code.

## Matched controls

| Control | Change from the full method |
|---|---|
| `diagnostic` | Full zero-initialized linear diagnostic controller |
| `uniform` | Constant zero control, uniform alignment contribution |
| `frequency` | Declared causal prefix-length control; **not PAD** |
| `cf` | Collaborative-only continued teacher baseline |
| `diagnostic_uniform_alignment` | Uniform alignment weights; diagnostic fusion retained |
| `diagnostic_no_alignment` | No MK-MMD term; diagnostic fusion retained |
| `diagnostic_no_conflict` | Remove disagreement correction; retain score and dimension bias |
| `diagnostic_loss_only` | Mask gradient feature |
| `diagnostic_gradient_only` | Mask loss feature |

All multimodal controls share the encoder, text features, trainable projection,
item gate, teacher initialization, batch order, negative streams and validation
selection rule. The teacher is trained and selected once per seed.

## Existing single-target workflow: evaluation and artifacts

For `training.py`, the default candidate policy is `full`: all nonpadding catalog items, with
target-independent membership. Two explicitly named history-filtered policies
are available; repeated-target handling is never implicit. Ties prefer smaller
integer item IDs. One relevant held-out item per user means Hit@K equals
Recall@K. Hit/Recall, NDCG and MRR are exported at5,10,20 by default.

Each run saves configuration/source/runtime/data hashes, selected teacher,
fixed diagnostics and sampling metadata, selected/last checkpoints, epoch logs,
per-user ranks and control parameters. Checkpoints bind seed, control, stage,
teacher and diagnostic identity. Reload verifies the original feature matrix,
raw item mapping, user split, cache and source. Text features remain external;
keep the prepared data and feature sidecar with the run.

Validation aggregation and item scoring are chunked. Catalog item vectors are
cached once per evaluation; there is no user-by-entire-catalog score matrix.
The ranking audit reports sufficient set/order preservation certificates
against the same checkpoint's collaborative branch, with the sampled users and
candidate scope recorded.

Analysis exports `aggregate_metrics.csv`, `paper_table.csv`, and `analysis.json`.
Means and sample standard deviations use training seeds as units. Paired
seed-bootstrap intervals and sign-flip tests condition on the fixed evaluation
users; repeated users across seeds are not treated as independent. Five seeds
give coarse inference, and the analysis applies Holm adjustment to its reported
family of paired tests. Incomplete, mixed-protocol or modified recorded runs
are rejected.
