# Changelog

## 2.0.0-rc1 — 2026-09-09

- Separated the fixed four-expert base softmax from incremental expert gates.
- Added descriptor-aware NULL/Top-1 hard routing with deterministic older-expert tie-breaking.
- Added query-exclusive BiMC LOO expansion necessity checks and hard-source, multi-view layer selection.
- Limited automatic growth to one proposed expert in one layer per incremental session.
- Added zero-initialised provisional experts and commit-on-improvement rollback.
- Added historical support accuracy checks, class-centroid route anchors and strict route-preservation acceptance.
- Split descriptor fitting and calibration across disjoint deterministic support views.
- Bumped checkpoints to schema 7/topology v2; older unified-router checkpoints require a Session 0 retrain.
