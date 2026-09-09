# FSCIL MoE — BiMC with demand-driven adapters

Current implementation: **2.0.0-rc1** (checkpoint schema 7).

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

# MiniImagenet BIMC
python main.py --data_cfg ./configs/datasets/miniimagenet.yaml --train_cfg ./configs/trainers/bimc.yaml

# CUB200 BIMC
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml
~~~

### Dynamic FSCIL MoE runs

The `bimc.yaml` configuration now auto-saves a dataset/seed-specific fixed
support manifest, alongside session checkpoints, query-exclusive LOO training
and demand-driven dynamic experts. For reproducible comparisons across model
variants, explicitly reuse one manifest and run a no-expansion baseline:

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml --support_manifest checkpoints/fscil_moe/cub_seed1_support.json --expansion_mode none
~~~

The default `auto` mode first measures an all-seen, query-exclusive BiMC LOO
classification deficit. Only hard support sources then contribute descriptor
coverage evidence, using fixed fitting views `[0, 1, 2]`; disjoint views
`[100, 101]` calibrate descriptor statistics and are never used for fitting.
Among the last four ViT blocks, the largest positive deficit may propose exactly
one expert per session.

The four base experts keep their original four-way softmax permanently.
Incremental experts have independent gates and descriptors, so appending one
cannot renormalise the base router. Deployment requires base coverage failure,
descriptor coverage by a ready incremental expert, and a gate score above NULL;
it then activates at most one incremental expert, with the older expert winning
near ties. A new expert is zero-initialised and unavailable until its descriptor
has been fitted and calibrated.

Expansion is transactional. The candidate is trained with refreshed
query-exclusive BiMC banks plus historical gate-invariance anchors, then evaluated
under honest hard routing. It is committed only when new-class LOO accuracy (or
equal accuracy with a better margin) improves, accuracy on the fixed historical
training support subset (20 deterministic samples per class by default) does not fall, historical route preservation passes its
threshold, and the old/new harmonic objective improves. The test split is never
used for this decision. Otherwise the exact pre-candidate adapter is restored.
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

Schema-7 checkpoints record topology v2, historical class-centroid routing
anchors, support identity, per-session acceptance diagnostics and RNG state.
Older unified-router checkpoints are intentionally rejected because loading them
would violate the base-route invariance contract; retrain Session 0 once.

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)
