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

This branch implements one current visual feature space with a frozen CLIP ViT
reference stream and an adapted stream. Unified checkpoints use schema 17;
older schemas are rejected because they do not contain the visual-only classifier
state and parent-initialized descriptor state required by this architecture.

The incremental lifecycle is:

1. The reference stream never receives adapter residuals. Its per-layer CLS
   features, after a frozen adapter LayerNorm, are the stable input for AE
   descriptors, router columns, expansion decisions and old-class route
   anchors. Expert functions consume the adapted stream.
2. Each expert owns a centered, tied-weight, low-rank linear AE. Base AEs are
   fitted with Base-router responsibility weights and independently held-out
   source identities. Their per-layer known-class scores calibrate expansion
   thresholds using the same class-wise statistic used at incremental time.
3. For each class and layer, compute the median signed reconstruction-error
   Z-score separately for every existing descriptor, then take the minimum
   across descriptors. Expand only if at least one class exceeds the calibrated
   threshold for every existing descriptor. Select the child AE parent by the
   lowest class-balanced mean descriptor score over uncovered classes.
4. Add at most one zero-up-initialized residual expert per layer per session.
   Train only that expert and its independent sigmoid router column. Base
   experts retain their softmax mixture; historical incremental columns remain
   independent and frozen. Layer decisions share one reference-feature scan
   because adapter residuals cannot change the reference stream; functional
   experts are still trained shallow-to-deep.
5. Store old-class Gaussian anchors separately in reference-route space and
   adapted expert-input space. Suppress the new router on reference anchors and
   suppress its gated residual using the corresponding adapted-space anchors.
6. Initialize each new AE basis from the selected historical parent, recenter
   it on only the uncovered classes assigned to the new expert, and fit it in
   the stable reference CLS space. Calibrate reconstruction errors on independent
   held-out source identities using that expert's router responsibilities;
   shrink weighted moments toward the selected parent descriptor's error
   statistics using effective sample size. Never use in-sample AE calibration.
7. Build source-exclusive leave-one-out visual prototypes. Calibrate new visual
   prototypes using a mixture of visual-neighbor and text-semantic-neighbor
   distributions over Base classes, transport the adapted/reference feature
   shift, and retain the calibrated visual prototype as the classifier target.
   Text semantics guide the calibration weights but are not fused into the
   final classifier.
8. Evaluate every seen class with one current CLIP+MoE forward path and the
   global calibrated prototype bank. No task gate, version-specific feature
   extraction, task-specific classifier, or task ID participates in prediction.

The linear AE, MSE reconstruction objective and reconstruction-error outlier
logic follow the SEMA reference. Text-description semantics guide visual
prototype calibration, while the final FSCIL classifier remains visual-only.
The independent incremental sigmoid columns, dynamic expert lifecycle and
global visual prototype bank are the unified design implemented in this branch.

Run module-level architecture tests:

~~~bash
python -m unittest discover -s tests -p "test_risk_guided_moe.py" -v
~~~

Do not start real-dataset acceptance until the architecture-level test suite,
checkpoint roundtrip, multi-session lifecycle, freeze-boundary checks and
reference/adapted feature-space checks all pass.

The support manifest remains fixed across comparisons. AA is the mean of the
per-session overall accuracies observed so far. PD is average forgetting in
percentage points: for each task, take its best observed task accuracy from
the task's introduction through the current session, subtract its current
accuracy, then average across tasks. Checkpoints save the complete expert
topology, AE reference buffers, dual-space per-class/per-layer old-class
anchors, global visual prototype states, diagnostics and RNG state; historical
per-sample feature embeddings are not retained in task state.

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)
- [SEMA-CL](https://github.com/huiyiwang01/SEMA-CL), for the linear AE descriptor and reconstruction-error outlier detection
- [MoE-Adapters4CL](https://github.com/JiazuoYu/MoE-Adapters4CL), for dynamic expert routing design

