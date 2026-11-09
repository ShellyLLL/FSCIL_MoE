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

### Dynamic FSCIL MoE runs

The `bimc.yaml` configuration now auto-saves a dataset/seed-specific fixed
support manifest alongside session checkpoints. Use the same manifest for
comparisons. The CUB200 run is:

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml --support_manifest checkpoints/fscil_moe/cub_seed1_support.json
~~~

At each incremental session, each eligible block receives a temporary candidate
expert. Five support folds compare its held-out, task-local fused classifier
loss with the unchanged encoder; at most one candidate is retained if its
gain exceeds one standard error. Training refreshes source-exclusive visual
prototypes, updates only the new expert and router column, and freezes them
afterward. Historical class weights and encoders retain their creation versions.
The frozen base encoder supplies common-space task evidence. Its task offsets
jointly minimize class-balanced task loss using source-exclusive support
features from every observed class. Expert logits only rank classes within
each task; no task identity is supplied at inference.

For staged development, reuse Session 0 and stop after Task 3 without changing
the normal full-run default:

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml --resume checkpoints/fscil_moe/session_00.pth --end_session 3
~~~

Resume from a checkpoint produced by this version (schema 9) with:

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml --resume checkpoints/fscil_moe/session_03.pth
~~~

Base-only schema 7/8 checkpoints can be reused. Earlier incremental
checkpoints cannot be resumed because their experts were optimized for a
different classifier objective. Compare the complete 11-session accuracy list
with the target using the same fixed support manifest; the support-only gate
diagnostics are not a substitute for test accuracy. `TaskDiagnostic` reports
the actual task selection rate and an oracle-task within-class upper bound;
the latter uses test labels only for analysis, never for predictions.

## Prototype-Confusion Guided Dynamic MoE-FSCIL

The new PC-MoE implementation is an independent training path and does not
reuse the legacy task-gated BiMC runner. It freezes the complete CLIP ViT
feature stream and adds layer-wise parallel residual experts over captured
frozen hidden states. Every layer uses one append-only Top-k Softmax router
over the complete old/new expert pool.

Prototype confusion (APC) compares each novel class's nearest old/within-new
prototype similarity with the base-session layer-wise confusion reference.
APC only ranks candidate layers. Candidate experts are trained jointly, then
a held-out, source-disjoint, class-balanced NLL over **all seen classes**
compares the candidate architecture with the unchanged-capacity baseline.
Expansion is committed only when global risk improves by the configured
minimum and the worst old-class risk increase remains below its limit.
Rejecting capacity never rejects the new classes: their support/text prototype
memory is still committed and classified by the existing expert pool.

Visual intra-modal calibration transports novel visual prototypes toward the
base visual prototype geometry; text intra-modal calibration combines CLIP
class text with available CuPL/LLM descriptions; inter-modal calibration
reliability-weights the two calibrated score streams. The dynamic MoE visual
embedding and calibrated text embedding meet in the same all-seen-class
logit space for training, capacity-risk evaluation, and inference.

Run the independent PC-MoE path with:

~~~BASH
python main_prototype_confusion.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/prototype_confusion_moe.yaml
~~~

To resume a PC-MoE checkpoint:

~~~BASH
python main_prototype_confusion.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/prototype_confusion_moe.yaml --resume checkpoints/prototype_confusion_moe/session_03.pth
~~~

Architecture invariant tests (not FSCIL performance experiments):

~~~BASH
python -m unittest tests.test_prototype_confusion_moe
~~~

Session checkpoints include expert topology/weights, all-class fit/calibration/
risk feature memories (no old images), text and description prototypes,
temperatures, APC reference distributions, support manifest, diagnostics and
random states. Each session also reports all-seen-class task accuracy (AA) and
base-class accuracy drop (PD).

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)

