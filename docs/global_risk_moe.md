# Fixed-feature global-risk MoE for FSCIL

## Problem and implementation boundary

The previous `retained` classifier subtracts each session's maximum expert
logit, so the winning session depends entirely on the frozen anchor and its
refitted task offsets. Expert improvements cannot repair that decision. Its
candidate objective only ranks classes within the new session. We replace both
objectives together, rather than adjusting their thresholds.

This branch's `bimc.yaml` selects `GlobalRiskRunner`. `bimc_legacy.yaml` is the
unaltered previous configuration and selects the original `Runner`; its source
is retained for reproducible ablations. No legacy task gate is invoked by the
new runner. Original CLIP weights stay frozen. Base MoE training is retained;
after S0 **the entire image encoder, including its base adapters, is frozen**.
Unused reconstruction descriptors are not trained in the new runner.

Incremental adapters operate on the final normalized feature z=f0(x), not on
changing transformer-layer activations. This sacrifices some internal-layer
adaptation capacity in return for a consistent replay space and one backbone
forward per test batch. It is a testable design tradeoff, not an optimality claim.

## One classifier for training, validation, and inference

Each class bank has a frozen set of expert IDs and an input-dependent router.
A bank corresponds to a cohort of classes for parameter sharing, **not a task
selection gate**. All banks are evaluated on every query, with no test task ID.
Experts take the same fixed z. Routing is a dense softmax over identity and the
bank's selected parallel residual experts:

    h_b(z) = normalize(z + sum_e w_b,e(z) A_e(z))
    score_c(z) = scale * h_b(z)^T p_c

The new router can reuse every previously accepted frozen expert. A candidate
may additionally include one zero-up-initialized bottleneck expert. No growing
softmax modifies any old bank. The dense mixture is bounded in its mixing
weights; this is not top-k sparse execution. Experts are shared, routers are
bank-specific. We do not claim that shared temperature alone makes all banks
calibrated: their scores are optimized against the same old-class competitors.

Text calibration blends template and description embeddings. Visual prototypes
are means of transformed support features. Visual calibration recomputes base
reference prototypes from base cached samples **through the same candidate
route**, then applies BiMC's similarity-weighted shrinkage. Text/visual fusion
is the same function for training, validation, final binding, and deployment.
The nonlinear adapter is applied to samples before averaging; mapping a stored
base mean would not be equivalent. No cross-space subtraction/transport is used.
Prototypes remain differentiable during candidate training. For a training
query, every support view of its original source is excluded from its own
class prototype. Other-class prototypes retain their available fitting rows.

Old class scores are computed by frozen banks. New scores compete with all of
them in a single softmax cross-entropy. The fitting objective is class-balanced
all-seen-class CE plus a residual displacement penalty. Old cached samples
therefore backpropagate into **new** scores through the softmax denominator,
without modifying any old scoring function. Gradient chunks accumulate one
full class-balanced objective per epoch; `BATCH_SIZE` controls memory, not
class-frequency weighting or optimizer step count.

## Memory protocol

This is **feature replay**, not exemplar-free learning. It stores float32
normalized S0-encoder features, labels, and original source IDs for all base
training examples and each incremental support example. No historical raw
images are used for incremental adaptation. Text prototypes are also stored.
The feature cache, head parameters, and wall/evaluation times are reported.
Raw base images are needed once when importing a legacy S0 checkpoint, to
re-extract consistent float32 features. Base images are not required again
for subsequent schema-10 continuation beyond normal dataset initialization.

Do not compare against strict no-memory methods without reporting this protocol
and memory budget. The implementation does not synthesize unobserved old data.

## Expansion is conditional empirical model selection

For every session, stratified source-disjoint folds compare:

1. Identity (no new router or expert).
2. Reuse: train a new bank router over identity and frozen experts, if available.
3. Expand: train a new bank router and one low-rank expert, within the budget.

All candidate parameter fitting, novel prototype construction, and base
calibration references exclude the fold's held-out sources. Historical and
current validation features are both scored against **all seen classes**.
Each candidate starts from the same initialization in every fold and final
refit; trial RNG does not perturb future sessions. Source folds are independent
of test data. The reference identity classifier includes the same new classes,
so old-class regret is measured against a meaningful no-adaptation alternative.

Candidate risk is mean fold class-balanced CE plus parameter cost (NLL per
million newly stored parameters). A candidate is admissible only if its gain
exceeds `MIN_GAIN` and its mean historical-class NLL increase does not exceed
`OLD_RISK_TOLERANCE`. Defaults are zero for both. **Filter before ranking**;
a rejected best unconstrained candidate cannot hide another admissible one.
Among admissible candidates choose lowest penalized risk, breaking ties by
stored parameter count and name. With no admissible candidate bind identity.
If the budget is exhausted, reuse remains possible. `EXPAND=False` from S0
produces the frozen-feature BiMC comparison without incremental experts.

No naive fold standard error is interpreted as a confidence guarantee. Folds
share fitting data and model selection is adaptive. Historical validation is
**conditional on fixed old classifiers**: old classifiers may have seen these
rows when originally trained, but new candidate fitting/calibration does not.
It is not an independent end-to-end generalization estimate. Final full-support
refit is a new fit from the same initialization, not a validated frozen model;
its test performance is unknown. Candidate failures raise before binding, so
formal model state remains unchanged. This is transactional rejection, not an
accuracy guarantee or automatic recovery from arbitrary training exceptions.

## What is guaranteed and what is not

For old class c, its expert set, expert parameters, router, prototype, scale,
and fixed input encoder do not change. Hence its raw score function is
unchanged after any later expansion. Tests check exact old logits, including
multiple banks and reuse. Class probabilities and accuracy can still change
because new classes compete. Zero forgetting in accuracy is **not** guaranteed.

No hard task gate limits expert contributions to within-session ranking.
Parameter freezing alone would not establish this property; immutable complete
class functions do. New classes cannot update old functions to correct old
errors. A final-feature adapter cannot recover visual information absent from
f0. These are intentional limitations, to be measured, not hidden.

Defaults (rank 16, 30 full-gradient epochs, learning rate 3e-4, residual penalty
0.1, parameter cost 0.01 per million parameters, budget 10) are starting values,
not values fitted on CUB test sessions. Tune them only on base-class held-out
pseudo-incremental episodes, then lock them before evaluating the real class
stream. No automatic meta-validation/tuning pipeline is claimed here.

## Checkpoints and metrics

Schema 10 stores the fixed image encoder, immutable head topology/weights,
feature memory, support manifest, calibration/selection protocol, text-file
hash, class ordering, RNG state, and unrounded metrics. Changes to the saved
protocol are rejected on resume; changing end session, output path, and device
is allowed. Saves use atomic replacement. `latest.pth` is always saved;
`SAVE_EVERY_SESSION` controls numbered snapshots.

A base-only schema 7/8/9 checkpoint can be imported into a **different output
directory**. S0 features/text/prototypes and accuracy are recomputed with the
requested new configuration. Legacy incremental checkpoints are rejected;
their task gate and adapted internal representations cannot be reinterpreted
as this model. Migration preserves the original checkpoint.

AA is the arithmetic mean of unrounded overall session accuracies. PD is S0
minus the latest session's overall accuracy. Both are reported every session,
together with per-cohort accuracy and error flows. PD includes increasing
classification difficulty, not just old-class forgetting. Novel accuracy is
the mean of incremental-cohort accuracies, matching the legacy reporting
convention; it is not the pooled novel-sample micro average.

## Reproducible comparison

Use the same dataset, class order, descriptions, manifest, and S0 checkpoint
for the legacy model, identity-only new model, and full new model. Use separate
checkpoint directories. Report multiple paired seeds (e.g. 1, 2, 3, 4, 5), full
S0-S10 trajectories, AA/PD, cohort-at-arrival versus final accuracy, accepted
experts, feature bytes, head parameters, and evaluation time. Assess whether
the paired gain is consistent, not just whether one seed rises by 0.1 points.
Do not select thresholds or seeds using the incremental test accuracies.

The previous supplied CUB log gives AA approximately 80.13%, final accuracy
76.34%, and PD 11.54 points. It is a historical reference, not a paired result
for this implementation. No CUB performance improvement is established by
unit tests or synthetic fixtures.

## Basis

- CLIP-Adapter, https://arxiv.org/abs/2110.04544: residual bottleneck adaptation
  on frozen VLM features supports the output-adapter design choice.
- SEMA, https://arxiv.org/abs/2403.18886: expandable adapters and learned mixture
  routing motivate capacity reuse; they do not certify five-shot risk selection.
- BiMC, CVPR 2025, https://github.com/yychen016/BiMC: intra-/inter-modality
  calibration supplies the classifier prior. Its training-free results do not
  prove gains for our trainable extension.

The global-risk objective, frozen bank bindings, and conditional feature-replay
selection here are a concrete synthesis; they are not a theorem copied from
these papers. Real-data ablations remain necessary.
