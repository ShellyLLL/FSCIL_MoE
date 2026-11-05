# Multi-layer APC-controlled dynamic MoE for FSCIL

## Shared encoder and historical versions

The method has one frozen pretrained CLIP ViT backbone. It does not create a separate ViT per session. Each configured Transformer block owns a VisualMoEAdapter with a shared base expert pool and append-only incremental experts.

A prototype stores the session version at which it was created. At inference, each class column is scored using the same expert activation boundary that produced that class prototype. The implementation uses the existing `max_session` argument to activate only experts whose birth session is not later than the requested version. This is versioned execution of one shared backbone, not multiple independent encoders.

## Multi-layer expansion controller

Every configured MoE block is independently calibrated during Session 0. The controller stores two empirical base-episode distributions per block:

1. Maximum novel-to-old prototype cosine similarity (prototype-confusion risk).
2. Source-exclusive K-shot novel-query prototype classification loss (existing-expert adaptation capacity).

During an incremental session, features are re-extracted for all fixed seen-class support examples under the current deployed topology. Each layer is a candidate if either its confusion risk or its adaptation loss exceeds its own calibrated threshold. This is a candidate-generation rule, not permission to deploy an expert.

The APC calibration is an empirical reference distribution, not a formal guarantee under distribution shift. The Bonferroni-adjusted upper quantile controls the calibration family across layers and the two monitored metrics.

## New-expert gain verification and multi-layer selection

Candidate layers are evaluated with source-disjoint folds from the current session's fixed K-shot support set. For each candidate layer:

- Append a zero-up expert and a new router column.
- Train only that expert and router column; all historical experts, router columns and ViT parameters remain frozen.
- Reset candidate parameters between folds.
- Compare held-out novel-class prototype loss against the no-new-expert baseline on the same folds.
- Accept a layer only when its paired mean loss reduction exceeds its estimated standard error (one-standard-error rule).

Selection is greedy and marginal. After a layer is accepted and trained on all current support examples, the full seen-class support set is re-encoded. The remaining downstream layers are re-evaluated because an earlier-layer expert changes their input distribution. The process repeats until no remaining candidate provides a supported gain. A session can therefore add no experts, one expert, or experts in multiple configured blocks. The method does not hard-code the earliest MoE block.

The final layer set is selected sequentially, not by exhaustive global subset optimization. The one-standard-error rule is a conservative finite-support heuristic; full CUB200 runs and repeated seeds are required before making performance claims.

## Historical knowledge and classifier

The pretrained ViT and historical experts remain frozen. The new expert is zero-up initialized, so adding it initially preserves the previous function. The parent Runner's versioned classifier is used by APCRunner: each class column is scored with its creation-session encoder version; task-level evidence is calibrated using source-exclusive historical anchors; within-task class logits are retained from the versioned classifier.

This avoids comparing a query embedding from the latest expert topology directly with a prototype created by an older topology. It does not mean that multiple ViT backbones are stored.

## Modality calibration and support protocol

Visual calibration transports the paired support shift from the base visual space. Text calibration fuses class-name and description features. The classifier then fuses calibrated visual and text prototypes. The support manifest fixes the K-shot sources across training, calibration and evaluation.

## Checkpoints and compatibility

Checkpoints persist the model weights, dynamic MoE topology, per-task prototype states, APC calibration state, support manifest, diagnostics and RNG state. APC calibration schema version 1 used a single stable block and is not compatible with the multi-layer version 2 controller. A legacy incremental checkpoint must not be silently resumed as a multi-layer experiment; restart from Session 0 with the new branch. A Session-0 checkpoint can be recalibrated for all configured blocks.

## Run and validation

```bash
python -m pytest -q
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc.yaml
```

No claim of higher AA or lower PD is made until the full CUB200 experiment completes. Report per-session Acc, class-balanced and task-balanced AA separately, PD/forgetting, per-block expert counts, risk, adaptation loss, paired validation gains, and routing utilization.
