# BiMC

This is the official implementation of paper **Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration (CVPR 2025)**.

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

## Experiments

First, remember to modify the data path `ROOT` in the `dataset` configuration file.

~~~BASH
# CIFAR BIMC
python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc.yaml

# CIFAR BIMC_Ensemble
python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml

# MiniImagenet BIMC
python main.py --data_cfg ./configs/datasets/miniimagenet.yaml --train_cfg ./configs/trainers/bimc.yaml

# MiniImagenet BIMC_Ensemble
python main.py --data_cfg ./configs/datasets/miniimagenet.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml

# CUB200 BIMC
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml

# CUB200 BIMC_Ensemble
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml
~~~

### Unified AE-driven dynamic FSCIL MoE

This branch implements a single current visual feature space. It intentionally
does not support resuming checkpoints produced by the earlier versioned/task-
gated architecture. Unified checkpoints use schema 12.

The incremental procedure is:

1. Run a frozen-reference ViT stream alongside the adapted stream. The reference stream never receives adapter residuals and supplies stable per-layer CLS features for routing, AE descriptors, expansion decisions, and old-class anchors.
2. Score each layer with the reconstruction-error Z-scores of every deployed
   linear AE descriptor. Each descriptor is a centered, tied-weight, low-rank linear AE and keeps a rolling 500-error reference.
3. Expand a layer only when the class-balanced mean absolute Z-score is above
   the configured threshold for **every** existing descriptor. Scan blocks in
   ascending order and recollect deeper-layer features after an upstream layer
   has been expanded and trained.
4. Add at most one residual expert per layer per session. Its up-projection is
   zero-initialized. Freeze the CLIP backbone, all historical experts and all
   historical router columns; train only the new expert and its independent
   sigmoid router column.
5. Suppress the new expert's router response and residual on stored old-class,
   per-layer Gaussian anchors. No historical images or historical encoder
   versions are used.
6. Fit the new expert's paired AE on its final local representation distribution,
   then freeze it and its 500-error reference buffer.
7. Build current-session visual prototypes with source-exclusive leave-one-out
   support means. Use calibrated text semantics to retrieve related old visual
   prototype directions and align the new visual prototypes. Fuse calibrated
   visual and textual prototypes into one global classifier over every seen
   class.
8. Evaluate all seen classes using one current CLIP+MoE forward path and that
   single global prototype bank. No task gate, version-specific feature
   extraction, or task-specific classifier branch participates in prediction.

The SEMA implementation is the reference for the linear AE, MSE
reconstruction objective, rolling reconstruction-error records and absolute
Z-score outlier rule. The BiMC paper supplies the textual intra-modal
calibration and visual/textual prototype fusion motivation. The semantic
similarity transfer from old visual prototypes is the FSCIL-specific alignment
design used in this branch; it is not claimed to be a verbatim reproduction of
BiMC's visual calibration equation.

Run tests:

~~~bash
python -m unittest tests.test_risk_guided_moe -v
~~~

Run a full experiment using the standard configuration:

~~~bash
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml
~~~

The support manifest remains fixed across comparisons. AA is the mean of the
per-session overall accuracies observed so far. PD is average forgetting in
percentage points: for each task, take its best observed task accuracy from
the task's introduction through the current session, subtract its current
accuracy, then average across tasks. Checkpoints save the complete expert
topology, AE reference buffers, per-class/per-layer old-class anchors, global
prototype states, diagnostics and RNG state. Do not resume schema 6-9
checkpoints: their versioned feature paths and task gates are incompatible
with the unified feature space.

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)
- [SEMA-CL](https://github.com/huiyiwang01/SEMA-CL), for the linear AE descriptor and reconstruction-error outlier detection
- [MoE-Adapters4CL](https://github.com/JiazuoYu/MoE-Adapters4CL), for dynamic expert routing design

