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

### Global-risk FSCIL MoE (new default)

`configs/trainers/bimc.yaml` now selects a fixed-feature parallel MoE with
all-seen-class training and inference. The S0 encoder is frozen; new routers
and optional low-rank experts learn against both new support and historical
cached features. Old class scoring functions are immutable. No task gate or
test task identity is used. This is **feature replay**, not a no-memory method.
See [the design, selection protocol, limitations, and ablations](docs/global_risk_moe.md).

Start a fresh CUB run:

```bash
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc.yaml --support_manifest checkpoints/cub_seed1_support.json
```

Reuse an existing base-only checkpoint (schema 7/8/9), preserving its original
output directory and re-evaluating S0:

```bash
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc.yaml --resume checkpoints/fscil_risk_stable/session_00.pth --end_session 3
```

Continue the new schema-10 run with the same training/calibration settings:

```bash
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc.yaml --resume checkpoints/fscil_global_risk/session_03.pth
```

The original configuration remains available as `configs/trainers/bimc_legacy.yaml`.
Use separate output directories and the same support manifest for comparisons.
To run the frozen-feature classifier without incremental experts, start from
S0 with `TRAINER.BiMC.GLOBAL_MOE.EXPAND False` and a separate checkpoint directory;
changing this setting midway through a schema-10 run is rejected.

Candidate decisions log identity/reuse/expand risk, historical-class regret,
parameter cost and accepted expert count. AA and PD use unrounded accuracies.
Defaults require base-class validation before performance claims; the included
CPU tests establish implementation invariants, not CUB accuracy improvements.

```bash
python -m pytest -q
```

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)

