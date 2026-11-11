# Unified Dynamic-MoE FSCIL

This branch implements one shared visual feature space for few-shot class-incremental learning (FSCIL), using a frozen CLIP ViT and incrementally expandable residual experts.

## Architecture

- The pretrained CLIP visual backbone is frozen.
- Each configured middle/high ViT block owns one expandable pool of bottleneck residual experts.
- Every expert in a block competes in one router. Inference computes only the selected Top-K experts (Top-2 by default); there is no Session-ID router, independent Sigmoid gate, or versioned encoder path.
- A newly appended expert has a zero output projection and a dormant router column. Appending it therefore preserves the current function until the new route is trained.
- After expansion, historical experts and router columns are frozen but remain eligible for routing. Only the newest expert and its router column are trainable.
- The expansion controller compares per-class features from the frozen ViT (all adapters disabled) against fixed base-session nearest-neighbor similarity thresholds at each inserted layer. Only layers whose current classes exceed their calibrated base threshold become expansion candidates.
- Each nominated layer must also pass source-exclusive, cross-fitted LOO risk evaluation against the complete bank of historical and current class prototypes. Candidate layers are evaluated individually and as conditional combinations; acceptance requires a positive one-sided lower confidence bound above a configured minimum risk reduction.
- Historical class-level inputs to the actual routers are stored separately from frozen-backbone expansion prototypes. New routes are initialized from the current-vs-history feature shift, kept dormant at initialization, and trained with a dedicated router learning rate. A diagonal-variance margin objective constrains new routes over historical support regions; this is an empirical support-statistic constraint, not a distribution-free guarantee.
- The final classifier uses one current-model visual feature space and one cumulative class-prototype bank. Historical prototypes are not recomputed by passing a mean token through a changed encoder.

## Training flow

1. Train the base-session MoE against the frozen text/description classifier and distill toward frozen CLIP visual features; recompute visual prototypes only after training.
2. Freeze the base experts and routers; store enhanced base visual prototypes, actual-router support statistics, frozen-backbone layer prototypes, and fixed per-layer expansion thresholds.
3. For each incremental session, compare raw frozen-backbone layer prototypes with all historical classes and nominate only layers exceeding their base-calibrated confusion threshold.
4. Create source-exclusive support folds and evaluate temporary experts with the same all-seen-class LOO prototype classifier.
5. Initialize each new route from current-vs-history layer features, then train only the newest expert and router column with separate learning rates. Refresh LOO support prototypes after every optimizer update.
6. Commit only experts that are active on current support, preserve historical routing statistics, and reduce cross-fitted all-class risk; otherwise roll back the temporary topology.
7. Freeze accepted experts, compute current-session visual/text prototypes, append raw and adapted layer statistics, and evaluate every seen class through the same dynamic encoder.
8. Save the model topology, weights, cumulative prototype states, fixed expansion thresholds, support manifest, metrics, and RNG state.

## Run

Set the dataset root and text-description JSON path in the dataset configuration, then run:

~~~bash
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml
~~~

For reproducible comparisons, reuse the same support manifest:

~~~bash
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml --support_manifest checkpoints/unified_dynamic_moe/cub_seed1_support.json
~~~

A staged run can stop after a selected session:

~~~bash
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml --end_session 3
~~~

Unified checkpoints use schema 12. Earlier APC, descriptor-router, and versioned-classifier checkpoints are intentionally rejected: their routing and prototype semantics are not equivalent and cannot be safely migrated.

## Validation

Run the CPU architecture tests:

~~~bash
python -m unittest -v tests.test_unified_dynamic_moe
~~~

The tests cover function-preserving expansion, one sparse Top-K competition, actual skipping of unselected experts, incremental freeze policy, route-activation gradients, covariance-aware historical route constraints, topology round trips, and source-exclusive LOO prototypes.

A passing unit suite is necessary but not sufficient. Full benchmark acceptance still requires the final trained checkpoint to report session-wise accuracy, AA, PD, base/novel accuracy, expert counts, per-expert route selection, and historical route-preservation values on the same support manifest and evaluation protocol.
