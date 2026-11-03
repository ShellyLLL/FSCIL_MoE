# BiMC

This repository is a research fork based on the implementation of **Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration (CVPR 2025)**.

## Abstract

Few-shot Class-Incremental Learning (FSCIL) challenges models to adapt to new classes with limited samples, presenting greater difficulties than traditional class-incremental learning. While existing approaches rely heavily on visual models and require additional training during base or incremental phases, we propose a training-free framework that leverages pre-trained visual-language models like CLIP. At the core of our approach is a novel Bi-level Modality Calibration (BiMC) strategy. Our framework initially performs intra-modal calibration, combining LLM-generated fine-grained category descriptions with visual prototypes from the base session to achieve precise classifier estimation. This is further complemented by inter-modal calibration that fuses pre-trained linguistic knowledge with task-specific visual priors to mitigate modality-specific biases. To enhance prediction robustness, we introduce additional metrics and strategies that maximize the utilization of limited data. Extensive experimental results demonstrate that our approach significantly outperforms existing methods.

## Installation

### Dataset

Please follow [CEC](https://github.com/icoz69/CEC-CVPR2021) to download *mini*-ImageNet, CUB-200 and CIFAR-100.

### Requirement

- `torch==1.13.1`
- `torchvision==0.14.1`
- `yacs==0.1.8` 
- `tqdm==4.66.1`
- `ftfy==6.1.1`
- `regex==2023.10.3`
- `scikit-learn==1.3.2`

## FSCIL MoE implementation

This fork extends BiMC with frozen-backbone visual MoE adapters and demand-driven
expansion. It is a research candidate, not a verified improvement over BiMC.
Read [the design, proof scope, limitations and experiment protocol](docs/fscil_design.md)
before comparing results. The original BiMC implementation is available at
https://github.com/yychen016/BiMC.

Update `DATASET.ROOT` in the dataset YAML first. The repository's description JSONs
and pretrained CLIP weights are also required for real experiments.

```bash
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc.yaml --expansion_mode auto
```

For a paired comparison, reuse the same Session 0 checkpoint and support manifest,
with separate output directories so runs do not overwrite one another:

```bash
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc.yaml --resume checkpoints/fscil_moe/session_00.pth --expansion_mode none TRAINER.BiMC.CHECKPOINT.DIR checkpoints/no_expansion
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc.yaml --resume checkpoints/fscil_moe/session_00.pth --expansion_mode auto TRAINER.BiMC.CHECKPOINT.DIR checkpoints/auto_expansion
```

`--support_manifest PATH` selects an explicit support manifest. The checkpoint
also contains the fixed manifest. `--end_session 3` supports a partial development
run; its AA/PD must not be reported as the complete benchmark result.

### Implemented contract

- Base training freezes CLIP and fits only the adapters; all historical parameters
  remain frozen during subsequent expansion.
- Every class keeps the expert-prefix path used to build its prototype. Inference
  scores **all** seen classes using their respective paths, without query task IDs.
  This preserves old score functions, not old accuracy against newly added classes.
- Soft routing is used in both training and inference. There is no NULL gate or
  hard Top-1 deployment path. Different feature paths require additional forwards.
- Descriptor coverage is evaluated in each descriptor's birth path. Expansion also
  requires a negative all-seen, source-exclusive BiMC classification margin.
  At most one candidate layer is trained per session.
- Incremental training uses visual/text calibration and fusion consistent with
  deployment, with refreshed stop-gradient support prototypes each epoch.
- Candidate retention requires lower deterministic support LOO CE without lower
  support accuracy. This is a fit diagnostic, **not held-out generalization evidence**.
  Rejection or exceptions restore the prior topology, weights and candidate RNG.
- Visual and fused prototypes are stored separately. Legacy prototype refinement
  and conflict repulsion are unsupported under the multi-path protocol.
- Checkpoint schema **7** stores feature paths, topology, prototypes, manifest,
  diagnostics, RNG and unrounded accuracies. Older schemas require retraining.
- AA includes Session 0; PD is `A_0 - A_B` in percentage points. No achieved
  AA/PD improvement is claimed without real-data experiments.

The old `bimc_ensemble.yaml` is an unsupported legacy configuration; its keys are
not implemented by this runner. Use `bimc.yaml`. For a frozen-CLIP baseline, append
`TRAINER.BiMC.VISUAL_MOE.ENABLE False` to the command and use a separate output directory.

### Numerical and integration tests

```bash
python -m pip install pytest
python -m pytest -q
```

The tests use a tiny real ViT and synthetic data; no tokenizer, CLIP download or
benchmark dataset is required. They cover old-score invariance, prototype
consistency, rollback, checkpoint recovery, and a three-session continuous versus
resumed run. This is not a benchmark performance evaluation.

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)




