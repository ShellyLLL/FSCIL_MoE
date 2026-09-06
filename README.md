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
support manifest, alongside session checkpoints, query-exclusive LOO training
and demand-driven dynamic experts. For reproducible comparisons across model
variants, explicitly reuse one manifest and run a no-expansion baseline:

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml --support_manifest checkpoints/fscil_moe/cub_seed1_support.json --expansion_mode none
~~~

The default `auto` mode computes each layer's discriminative coverage deficit:
only samples with a negative all-seen visual LOO margin and descriptor coverage
excess contribute expansion evidence. The largest positive eligible layer adds
one Linear Gate, Expert and Scale. Training uses one softmax over `NULL` and all
dynamic experts, refreshed query-exclusive visual prototypes each epoch, and only
visual classification plus historical-path invariance losses. Deployment uses the
same logits with hard Top-1 routing. A candidate is retained only when its hard
deployment LOO objective plus historical invariance is lower than the pre-candidate
objective; otherwise the exact pre-expansion state is restored. Force/session/block
overrides are intentionally unavailable in the production path.
The explicit full-auto CUB200 command is:

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml --expansion_mode auto
~~~

For staged development, reuse Session 0 and stop after Task 3 without changing
the normal full-run default:

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml --resume checkpoints/fscil_moe/session_00.pth --expansion_mode auto --end_session 3
~~~

Resume from a checkpoint produced by this safety-bounded version with:

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml --resume checkpoints/fscil_moe/session_03.pth
~~~

Legacy dynamic-expert checkpoint schemas are intentionally rejected when they
predate the current logit-NULL routing and transactional safety contract.

Each completed session also records per-expert/null routing and a deterministic
support-proxy leave-one-expert-out diagnostic (`accuracy_gain` and
`margin_gain`). It masks the selected residual without re-routing to another
expert, so the reported value is that expert's direct marginal contribution.
Historical experts additionally receive a support-only forward-interference
report. Accuracy targets are kept outside runtime; use
`python tools/report_regression.py --accuracies ...` after a run.

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)



