# Validation record

Validation date: 2026-11-16. This record describes correctness and pipeline
checks, not evidence of improved recommendation accuracy.

## Automated checks

54 CPU regression/integration tests passed locally. Coverage includes:

- Causal and padding invariance; identical training/retrieval representations.
- Exact neutral initialization and signed diagnostic control.
- Gate derivatives, raw item mixing and externally fixed text features.
- Detached MK-MMD weights and the no-alignment/no-conflict/feature ablations.
- Unit-scale teacher query gradients and independence from held-out targets.
- Timestamp ordering, dataset adapters, strict feature/data mapping and hashes.
- Exact dense/chunked ranks, deterministic ties, repeats and cutoff boundaries.
- Randomized ranking-bound checks and distinct set/order preservation tests.
- Epoch-boundary resumed weights equal uninterrupted weights exactly.
- Checkpoint reload reproduces ranks; wrong-control checkpoints, corrupt caches,
  changed raw item mappings and changed source are rejected.
- Complete paired experiment grids, artifact integrity and seed-level inference.

Local compatibility environment: Windows, Python3.12.14, preinstalled
PyTorch2.5.1+cpu and NumPy2.3.5. The distributed requirements use PyTorch>=2.6;
the repository's CPU CI installs the declared dependency range on Python3.11.
CUDA execution and LLM2Vec model generation were not exercised locally.

## Real-data pipeline check

The public Pantry source contains12,871 users,8,347 items and115,004 events.
The prepared source catalog and all events are audited; the smoke run selects
the first128 users explicitly and keeps the full8,347-item candidate catalog.

Executed configuration: `configs/pantry_lexical_smoke.json`, output override
`runs/pantry_method_aligned_v4`.

- Seeds2025 and2026.
- Two selected teacher runs and18 selected student runs across all nine controls.
- Two teacher and two student epochs per run, CPU, width16, one encoder layer.
- Explicit signed-hash metadata features; **NOT LLM embeddings**.
- Full target-independent catalog, item0 excluded, smaller-ID tie breaking.
- Completed-job resume verified; independent selected-checkpoint reload produced
  byte-identical per-user rank output for seed2025 diagnostic control.
- Sixteen sampled users passed through the full-catalog ranking-audit pipeline.
- Analysis checked all18 configured jobs,18 per-user hashes,18 checkpoint hashes
  and18 diagnostic-cache hashes before producing tables/comparisons.

All checkpoints, logs, per-user outputs and analysis remain in the local run
directory; they are excluded from source control. The smoke results cannot
support a benchmark performance comparison. The three formal datasets and their
actual mapped language-model embeddings must be provided for formal experiments.
