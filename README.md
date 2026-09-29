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

### Versioned demand-driven FSCIL (this development branch)

This branch preserves the patent's S1–S7 data flow: base MoE training, reuse
of existing experts, classification-aware expansion, visual prototype
calibration, LLM-description text calibration, and all-seen-class inference.
In contrast with reconstruction-threshold-only CIL expansion, the S4 controller
uses source-exclusive LOO classification margins to identify a discriminative
deficit, uses topology-matched descriptors to rank candidate vision layers,
then compares each candidate against **no expansion** using class-stratified,
source-exclusive cross-validation. The objective is validation CE plus a small
parameter-growth term; only the winning candidate is retrained on all five
support examples and committed. All other candidates are discarded.

Forgetting protection is architectural rather than a training heuristic:
each historical classifier bank records its owner topology. Existing expert
and routing columns are frozen; class-specific visual scores always use the
same historical prefix-softmax topology used to compute that class prototype.
A frozen Session-0 MoE supplies the shared reference space for base-to-novel
visual calibration. Final S7 scores combine a topology-consistent reference
fused prototype score and an owner-topology dynamic fused prototype score.
The text encoder and descriptions stay frozen. Neither old training images
nor a test-time task ID are required.

```bash
# Fix support images across variants by reusing the same manifest.
python main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/bimc.yaml \
  --support_manifest checkpoints/fscil_moe/cub200_support_seed1.json

# Stop after Session 3, then resume without resampling its support set.
python main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/bimc.yaml --end_session 3

python main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/bimc.yaml \
  --resume checkpoints/fscil_moe/session_03.pth

# CPU-only model invariants: no dataset, GPU or pretrained checkpoint required.
python -m unittest discover -s tests -p 'test_demand.py' -v
```

The Version 7 checkpoint contract is intentionally incompatible with old
single-space checkpoints. Run the new architecture from Session 0.
`TRAINER.BiMC.DEMAND.CV_FOLDS`, `COMPLEXITY_WEIGHT`, and
`DYNAMIC_WEIGHT` are configured in `configs/trainers/bimc.yaml`.
Complexity and fusion coefficients should be tuned on class-disjoint
pseudo-incremental splits of **base training classes**, never on incremental
test images. Full five-fold candidate selection can be computationally
expensive; it is performed once per increment, with at most one final
committed expert per session. No AA/PD improvement is claimed without a
complete, controlled multi-seed experiment.

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)



