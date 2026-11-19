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

### Unified AE-driven dynamic FSCIL MoE (method model v1.0)

The branch implements the unified FSCIL method model:
task-level AE distribution-shift detection, on-demand residual expert expansion,
Base-Softmax plus incremental-Sigmoid routing, selective training/freeze,
semantic-guided novel visual prototype calibration, and one global visual
prototype classifier.

Lifecycle:

1. The frozen CLIP ViT reference stream provides stable per-layer CLS features
   for AE coverage detection and router inputs. The adapted stream receives the
   residual expert adapters. Historical experts and routers remain frozen but
   participate in every later forward pass.
2. Base experts and the Base router are trained on Base classes. After Base
   training, frozen-reference CLS sources are partitioned into disjoint AE-fit
   and AE-error-reference sets. Each expert's centered low-rank linear AE is
   fitted using its deployed Base-Softmax responsibility. The independent
   holdout estimates that descriptor's reconstruction-error mean and standard
   deviation. It does not calibrate the task-level expansion threshold.
3. At each incremental Session, collect frozen-reference CLS input
   features for every support source. Each descriptor converts its independent,
   unweighted holdout errors into finite-sample upper-tail conformal p-values.
   The maximum p-value across deployed experts tests whether at least one
   existing expert covers each sample; this remains valid as the expert pool
   grows. The lower-median sample p-value is converted into an exact binomial
   order-statistic p-value. Expand one expert when this task p-value is below
   the configured false-alarm rate (default 0.05); otherwise reuse the pool.
   No pseudo-Sessions or task-threshold calibration are used.
4. Train the new expert and its router column on all current-Session samples.
   Every current-Session sample is positive router supervision and participates
   in the global seen-class visual-prototype classification objective. Old-class
   route anchors suppress new-expert interference. All historical experts,
   router columns and the ViT remain frozen.
5. Fit the new expert's paired AE on all current-Session frozen-reference
   CLS features. A source-exclusive holdout estimates its reconstruction-error
   mean and standard deviation. The AE is used for future task-level expansion
   decisions, not classification.
6. After expert training and freezing, compute each novel class's adapted
   visual support prototype. Encode its class description with the frozen
   CLIP text encoder. Description-semantic similarity retrieves related Base
   description semantics and their corresponding Base **adapted visual**
   prototypes. Blend the novel visual prototype with this retrieved visual
   direction using a calibration strength controlled by semantic confidence
   and support-prototype reliability. Only novel visual prototypes are
   calibrated; historical prototypes remain unchanged.
7. Merge all seen-class visual prototypes into one global visual classifier.
   Training uses source-exclusive leave-one-out novel visual prototypes to
   prevent support-query leakage. Inference uses one current CLIP+MoE feature
   path and the global visual prototype bank. AE is not used at inference;
   there is no task ID, task gate, task-specific classifier or text classifier.

The semantic calibration does not use visual-neighbor weighting, prompt/
description text intra-modal fusion, or BiMC inter-modal classifier fusion.
Text descriptions only retrieve Base visual knowledge; the final classifier
remains visual-only.

Run architecture tests:

~~~bash
python -m unittest discover -s tests -p "test_risk_guided_moe.py" -v
~~~

Do not start real-dataset acceptance until architecture tests, checkpoint
roundtrip, multi-session lifecycle, freeze-boundary checks, and reference/
adapted feature-space checks all pass.

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)
- [SEMA-CL](https://github.com/huiyiwang01/SEMA-CL), for the linear AE descriptor and reconstruction-error outlier detection
- [MoE-Adapters4CL](https://github.com/JiazuoYu/MoE-Adapters4CL), for dynamic expert routing design

