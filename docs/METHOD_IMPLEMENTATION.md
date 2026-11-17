# Method-to-code correspondence

This implementation follows the non-experimental revision titled *From Alignment
to Fusion: Diagnostic-Guided Integration for Sparse-Interaction User
Recommendation*. It is a new method-aligned implementation, not evidence that
historical paper tables were reproduced.

| Method component | Implementation | Invariant |
|---|---|---|
| Shared ID/text causal encoder | `fafirec/model.py`, `encode_branches` | Shared Transformer parameters; raw input vectors; valid-event positions; causal and padding masks |
| Unit-scale teacher diagnostics | `fafirec/diagnostics.py`, `query_diagnostics` | Loss is softplus(-q·(v_pos-v_neg)); gradient is with respect to an independent detached normalized query, without sqrt(d) |
| Fixed user diagnostic | `compute_diagnostics` | Final training transition; fixed once-per-user negative excluding full training history; evaluation-mode teacher; population standardization over training users |
| Shared linear control | `FAFIRecModel._control_scores` | Zero scalar intercept and two zero coefficients; both diagnostic signs unrestricted; no old nonlinear reliability estimator |
| Weighted MK-MMD | `alignment_weights`, `_weighted_mmd` | Batch softmax at temperature1; detached weights; five fixed uniform Gaussian kernels; one final supervised state per user; diagonal terms retained |
| Conflict-conditioned gate | `_fusion_gate` | sigmoid(s+beta+softplus(eta_raw)(2sigmoid(s)-1)abs(c-t)); neutral start equals matched uniform control |
| Item representation | `item_vectors` | User-independent dimension gate mixes raw ID/projected text before normalization |
| Pairwise recommendation | `forward` | sqrt(d)-scaled normalized dot products; softplus BPR at valid shifted next-item positions only |
| Retrieval | `user_vectors`, `item_vectors`, runner `evaluate` | Same representation/scoring functions; dropout disabled; item0 excluded |
| Ranking preservation analysis | `fafirec/certificates.py` | Post-normalization user and item drift; fixed checkpoint CF reference; separate set/order certificates |

The cached diagnostics characterize in-sample teacher fit and contrast geometry.
They are dependent statistics, not two independent reliability estimates or
calibrated correctness probabilities. A larger gradient is not assigned a fixed
positive interpretation.

The encoder is causal conditional on the cached control. That control uses the
user's full observed **training** history and is broadcast across training
prefixes. This is a transductive observed-user formulation, not a strict
prefix-online or zero-history protocol. Never add validation/test targets to
the diagnostic cache.

`cf` continues the same selected teacher for the same student epoch budget.
Other controls use the same selected teacher's collaborative initialization and
the same seed, epoch permutations, negative-sampling streams, optimizer and
validation selection rule. The `frequency` comparison uses causal prefix
length, not PAD's target-item-frequency routing, and must not be labeled PAD.

The fixed kernel widths are `(0.5,1,2,3.5,5)`. Default alignment weight, encoder
sizes and training budgets are starting configurations, not validated optimal
hyperparameters. Tune only using validation; record each chosen configuration.

Prediction candidate policies are explicit:

- `full` (default): every nonpadding metadata-catalog item, independent of target.
- `exclude_seen`: remove observed history identities; repeated targets cause an error.
- `exclude_seen_restore_target`: historical offline convention that restores the
  true target after filtering; it is target-conditioned and must be reported.

Single-positive Hit@K equals Recall@K. NDCG and MRR use the same one-based rank.
Equal scores prefer smaller integer item IDs. The runner obtains target scores
from an identical first chunk-scoring pass, then accumulates exact ranks in a
second pass. It caches item vectors once per evaluation and never materializes
the complete user-by-catalog score matrix.

The ranking audit compares the fused and CF branches of the **same** checkpoint.
It does not compare against a separately trained baseline, enforce certificates
during training, or equate top-K set preservation with NDCG preservation.
