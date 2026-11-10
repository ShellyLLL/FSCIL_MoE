# Unified Dynamic-MoE FSCIL

This branch implements one shared visual feature space for few-shot class-incremental learning (FSCIL), using a frozen CLIP ViT and incrementally expandable residual experts.

## Architecture

- The pretrained CLIP visual backbone is frozen.
- Each configured middle/high ViT block owns one expandable pool of bottleneck residual experts.
- Every expert in a block competes in one router. Inference computes only the selected Top-K experts (Top-2 by default); there is no Session-ID router, independent Sigmoid gate, or versioned encoder path.
- A newly appended expert has a zero output projection and a dormant router column. Appending it therefore preserves the current function until the new route is trained.
- After expansion, historical experts and router columns are frozen but remain eligible for routing. Only the newest expert and its router column are trainable.
- Candidate expansion is evaluated by support-source leave-one-out (LOO) risk against the complete bank of historical and current class prototypes. Candidate layers are evaluated individually and then as conditional combinations; an expansion is accepted only when its measured gain exceeds its fold standard error.
- Historical class-level layer-input means and diagonal variances are retained as compact statistics. A covariance-aware margin loss constrains each new router column to remain below the previous Top-2 routes over historical class regions, while a complementary activation loss makes the new route useful for current classes.
- The final classifier uses one current-model visual feature space and one cumulative class-prototype bank. Historical prototypes are not recomputed by passing a mean token through a changed encoder.

## Training flow

1. Train the base-session MoE against the base classes, with frozen CLIP features as a distillation target.
2. Freeze the base experts and routers; store base visual prototypes and class-local layer statistics.
3. For each incremental session, create source-exclusive support folds.
4. Evaluate the unchanged model and temporary candidate expert sets using the same all-seen-class prototype classifier.
5. Train accepted experts and route columns using all-seen-class LOO classification plus historical route-preservation objectives.
6. Freeze the accepted experts, compute current-session visual/text prototypes, append class-local statistics, and evaluate every seen class through the same dynamic encoder.
7. Save the model topology, weights, cumulative prototype states, support manifest, metrics, and RNG state.

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
