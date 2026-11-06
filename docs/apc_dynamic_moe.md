# Unified version-consistent APC and dynamic MoE architecture

## Status and scope

This branch is an architecture refactor, not a performance claim. A full CUB200 experiment is gated on passing the unit, integration, checkpoint round-trip, and multi-session acceptance tests. No AA or PD improvement is claimed until those experiments are run.

The frozen CLIP ViT backbone, visual MoE adapters, zero-up incremental expert initialization, historical expert freezing, and BiMC visual/text calibration are retained.

## One version definition

An encoder version is a monotonically increasing **MoE topology version**, not a session number. A session may commit more than one expert; every committed topology transition receives a new version. Expert metadata therefore stores both:

- `birth_session`: which FSCIL session introduced the expert (audit metadata).
- `birth_version`: the topology version at which the expert becomes active (execution semantics).

A class prototype stores its `encoder_version`. Its image feature and its classifier prototype are always evaluated with the same version. Legacy topology manifests without `birth_version` are migrated in expert order to unique versions.

## APC representation

APC uses the post-residual Transformer block hidden state: the exact CLS state passed to the next Transformer block. The Adapter's normalized input and adapter-local output remain separately exposed for diagnostics, but are not substituted for the actual block output. APC calibration schema is version 4. Its calibration state is bound to the exact MoE topology signature/version, historical global class IDs, and support-manifest hash; stale calibration is rebuilt before a decision.

## Candidate validation and deployment classifier

A candidate is no longer accepted from novel-only class loss. Each source-disjoint fold builds the current task's support prototypes under the tested topology, merges them with immutable historical task states, and evaluates the same retained classifier used at deployment:

1. Each global class column is scored using that class's creation topology version.
2. Task evidence is computed in the frozen base anchor space.
3. Task offsets are fitted from source-exclusive support anchors for the tested candidate/null state.
4. Within-task class evidence is combined with task evidence using the deployment scoring rule.
5. The held-out novel query is evaluated against **all seen classes**, not only the current session's classes.
6. The objective also includes class-balanced source-exclusive anchor task-gate loss, so a candidate cannot be selected solely for improving new-class discrimination while ignoring the historical task gate.

The candidate and null model use paired, source-grouped folds and identical scoring semantics. Every view of a source is held out together, and a source ID is forbidden from crossing class labels. Candidate parameters are reset between folds. The best candidate must exceed a family-wise paired threshold (Bonferroni-adjusted normal critical value) for the number of candidates tested in that round. The best-performing fold's trained expert/router state initializes full-support retraining, rather than restarting from an unrelated random initialization. Greedy selection is marginal: after a committed layer changes the topology, APC is recalibrated and all remaining layer representations are recomputed.

## Historical knowledge and state

Historical experts and router columns are frozen. Historical class prototypes remain immutable and retain their creation topology version. Task-gate calibration uses source-exclusive anchors. Checkpoints persist topology metadata, task states, gate offsets, APC calibration, support manifest, diagnostics and RNG state. The current checkpoint schema is 11; schema 10 and 11 APC payloads are recognized, but APC v2/v3 calibration is never reused by the v4 controller. It is recalibrated from the historical support manifest at the completed session, without requiring the entire FSCIL stream to restart.

## Efficiency accounting

Every candidate trial records validation/search time. Every committed expert records training time and incremental parameter count. Session diagnostics also retain expert counts, routing summaries and deployment evaluation time. These are required alongside AA and PD; expert expansion efficiency is not inferred from accuracy alone.

## Synthetic multi-session integration coverage

`tests/test_apc_multisession_integration.py` drives the production `APCRunner.run()`
through Session 0, Session 1 and Session 2 using deterministic in-memory data and
the production `VisualMoEAdapter`, candidate-fold trainer, APC controller,
versioned global classifier, anchor task gate, topology exporter/rebuilder and
checkpoint serializer/restorer. Only the image/text backbone and data source are
replaced with small deterministic fixtures; the expansion and classifier
control flow remains production code.

The integration test verifies that real candidates are trained and committed,
multiple layers can receive distinct topology versions in one session, old
version-0 class scores remain invariant after later experts are introduced,
historical parameters are frozen, and a newly constructed runner restores the
full checkpoint with identical topology, model tensors, class/version/task
metadata, task-gate offsets, versioned logits, retained logits, AA and PD
diagnostics. The synthetic fixture is a software integration test, not a
benchmark and not evidence of CUB200 accuracy improvement.

## Required acceptance gates before CUB200

- Adapter input, adapter-local output, and actual post-residual block output have distinct definitions and tests.
- Expert birth versions are monotonic and topology save/load round-trips exactly.
- Historical class logits are invariant under the addition of a later-version expert when evaluated at their creation version.
- Candidate/null validation uses the exact global versioned BiMC classifier, all currently accumulated historical task states, and source-exclusive support prototypes.
- APC state is rejected when its topology signature/version or historical class set is stale.
- Cross-validation holds out complete source groups, not individual rows/views.
- Candidate selection controls family-wise false expansion across layers and reuses the selected trained fold state for final support retraining.
- Task-gate offsets and all prototype/version metadata survive checkpoint round-trip with identical logits.
- Multi-layer expansion commits zero, one, or multiple experts without duplicate versions or stale downstream APC features.
- Tests cover candidate rollback, frozen historical parameters, source leakage, and checkpoint recovery.
- Metrics report class-balanced AA, task-balanced AA, PD, expert counts, incremental parameter growth, candidate search time, training time, and inference latency.

## Run

```bash
python -m pytest -q
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc.yaml
```

The command above is not an authorization to launch the full experiment. First satisfy all acceptance gates in automated tests and a small multi-session integration run.
