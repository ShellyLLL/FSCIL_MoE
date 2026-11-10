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
frozen hidden states. Every layer uses one append-only global Top-k Softmax
router over the complete old/new expert pool; no task-specific router or
task-ID inference branch is used.

Prototype confusion (APC) compares each novel class's nearest old/within-new
prototype similarity with a frozen base-session layer-wise reference. For each
session, it constructs an empirical null distribution of maximum similarities
using base classes as pseudo-queries facing the same number K of competitors.
Competitors are sampled without replacement when possible and with replacement
when K exceeds the base-class pool. The finite-sample upper-tail empirical
p-value is computed for every layer/class pair. A Benjamini-Yekutieli correction
controls false discoveries under arbitrary dependence among these tests.
APC proposes capacity-demanding layer/class pairs; it does not authorize
capacity by itself. The calibration relies on approximate exchangeability of
base and novel class geometry and is not a distribution-free guarantee.
Each appended gate is initialized strictly below the historical Top-k boundary,
so historical Top-k membership and Softmax weights are exactly preserved at
birth. During incremental training, APC demand is retained at class level: only
novel classes whose calibrated confusion percentile crosses the capacity
threshold are positive route targets for the new expert. Historical classes
and current non-demand novel classes are protected negatives. A margin
objective gives the new gate a direct gradient before it is selected:
demanded novel support is trained to cross the old-pool Top-k boundary,
while protected features are trained to remain below it. Historical experts and historical router columns remain
frozen. Incremental training does not impose uniform expert usage.

Historical memory is strictly prototype-only. At the end of each session,
fit, calibration and risk records are compressed independently to one class
centroid per feature space, together with a count; per-example features,
source IDs and route records are discarded. During a new session, only that
session's support examples are available as sample-level features. Historical
classes contribute their stored centroids to route protection, calibration
and risk-anchor calculations. Consequently, the all-class risk gate is a
prototype-anchored risk proxy, not an empirical risk measured on replayed old
examples; it must not be described as old-sample replay or exact old test risk.

Candidate experts are assessed on the same global inference router. A
candidate is committed only when the class-balanced all-seen-class
prototype-anchored risk improves, old- and new-class risk regressions remain
bounded, historical prototype route drift/intrusion stay within limits, and
every new class reaches the minimum novel-expert activation coverage.
Expansion rejection never rejects the new classes: their current-session
class prototypes and text prototypes are retained and classified by the
existing expert pool.

Visual intra-modal calibration transports novel visual prototypes toward the
base visual prototype geometry; text intra-modal calibration combines CLIP
class text with available CuPL/LLM descriptions; inter-modal calibration
reliability-weights the two calibrated score streams. The dynamic MoE visual
embedding and calibrated text embedding meet in the same all-seen-class
logit space for training, capacity-risk evaluation, and inference.

PC-MoE is selected through the shared `main.py` entry point and the `PCMOE.ENABLE: True` setting in `bimc.yaml`. The PC-MoE architecture and runner remain unchanged; only configuration registration and dispatch are unified. Example CUB200 run:

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml
~~~

To stop after Base Session:

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml --end_session 0
~~~

To resume a PC-MoE checkpoint:

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml --resume checkpoints/prototype_confusion_moe/session_03.pth
~~~

Architecture invariant tests (not FSCIL performance experiments):

~~~BASH
python -m unittest tests.test_prototype_confusion_moe
~~~

Session checkpoints use schema 5 and include the base pairwise-similarity
geometry used to construct cardinality-matched empirical APC nulls, expert topology/weights,
prototype-only class fit/calibration/risk memories (one centroid per class
and feature space, with no old per-example features or source IDs), text and
description prototypes,
temperatures, APC reference distributions, support manifest, diagnostics and
random states. Each session reports all-seen-class accuracy and macro task-average
accuracy (AA). Final Acc and PD are reported only at the final real FSCIL session.
Expansion acceptance additionally requires risk reduction on APC-demanded
classes. Historical risk is a prototype-risk surrogate, not old-example empirical risk.

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)

