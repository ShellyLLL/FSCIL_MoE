# APC-controlled dynamic MoE for FSCIL

## Architecture

1. **Frozen visual encoder.** The pretrained CLIP visual backbone and text encoder remain frozen. Base adapters are trained in Session 0 and then frozen.
2. **Expandable MoE adapter.** Shared base experts use Softmax routing. Incremental Sigmoid gates form a separate bounded residual branch with total gate mass strictly below one; zero-up initialization preserves the old function at expert creation.
3. **Adaptation-aware prototype confusion controller (APC).** APC calibrates a K-shot pairwise prototype-similarity distribution on base pseudo-incremental episodes. It expands when maximum new-to-old similarity in the current adapted representation exceeds an empirical, session-size-adjusted threshold.
4. **Dynamic routing and activation/freezing.** Only the newly appended expert and its router column are trainable. ViT, shared experts, old incremental experts and old router columns remain frozen.
5. **Incremental learning and old-knowledge protection.** Source-exclusive all-seen-class prototype CE is combined with visual-text alignment and penalties on the newest expert's effective gate and residual at stored old-class anchors. Failed fitting rolls back the new expert.
6. **Dual-level modality-calibrated classifier.** BiMC visual prototype calibration and text-name/description calibration are followed by visual-text fusion and one global all-seen-class prototype classifier. Task identity is not used at inference.

## Stable APC comparison locus

An adapter at an earlier transformer block changes inputs to every later block. Historical downstream block-input prototypes therefore cannot be compared directly with current downstream features without replaying historical images or retaining full token-level activations.

To avoid comparing incompatible feature spaces, APC growth is restricted to the **earliest configured MoE block**. Its input is invariant because no preceding MoE adapter exists. The system stores pre-adapter CLS features for each support source, reapplies the current adapter pool to old and new anchors, and computes confusion risk in the same adapted space. Other configured blocks retain shared base experts but do not receive incremental experts in this implementation.

This is a deliberate correctness boundary. Independently expanding downstream blocks requires token-level historical feature replay or another mathematically consistent representation transport.

## Calibration and limitations

APC uses deterministic pseudo-episodes from base classes, with K distinct source IDs per pseudo-class. Its empirical threshold is a family-size-adjusted upper quantile. Similarity pairs are dependent and future session distributions may shift; this is an empirical decision rule, not a formal false-expansion guarantee. Defaults require base validation before performance claims.

The old-class protection term constrains the new expert on stored historical CLS anchors. It reduces interference but does not prove every old test image is invariant. This is feature replay, not exemplar-free learning.

## Checkpoints

New checkpoints use schema 10 and persist APC calibration, dynamic MoE topology, task prototype states, stable-block feature anchors, support manifest, RNG state and diagnostics. Schema 6–9 base-only checkpoints can be imported and recalibrated. Legacy incremental checkpoints are rejected because they use a different classifier objective.

## Run

```bash
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc.yaml
python -m pytest -q
```

No claim of improved AA or reduced PD is made until full dataset runs complete on the target environment.