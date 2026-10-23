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

At each incremental session, the method first evaluates a **router-only** update
with experts frozen. Source-disjoint support folds compare candidates using the
same objective: global-class held-out cross-entropy plus a reference-drift penalty.
The reference is the immutable post-base MoE function, not the previous session's
mutable router state. A candidate must improve mean held-out objective by more
than one standard error and satisfy `MAX_REFERENCE_DRIFT`; a post-fit guard checks
the update trained on all current-session support examples. If router-only
adaptation is not supported, descriptor coverage gates temporary expert candidates.
At most one new expert is retained, and its candidate must pass the same validation
and drift constraints. Rejected or failed updates restore the exact MoE topology,
parameters, router state, and descriptor buffers from before the candidate.
Historical experts stay frozen and continue to participate in the shared Softmax
router.

Only the **current session's** support prototypes are refreshed after an update;
historical support images are not replayed. Each prototype state records the model
revision and fixed target feature-space reference. The fixed-reference constraint
reduces cross-session embedding drift, but cannot guarantee exact historical-class
invariance without historical samples. BiMC text/description fusion and base-path
visual calibration remain enabled. At inference, each test image is encoded once
by the current model and compared against one global bank of prototypes for all
seen classes. Report AA and PD over the full sequence; do not assume the added
constraints improve them without measurement.

The fixed-reference state uses checkpoint schema 11. Incremental checkpoints
from schema 10 or earlier are rejected because their historical prototypes were
not maintained under the fixed-reference contract; resume from a base-only
checkpoint or restart at Session 0. Reuse the same fixed support manifest for all
ablations.

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml --support_manifest checkpoints/fscil_moe/cub_seed1_support.json
~~~

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)

