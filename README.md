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

### Unified AE-driven dynamic FSCIL MoE (method model v1.1)

The method uses one Session-level novelty decision per expandable transformer
layer. It combines the historical AE/RD pool, dynamic expert migration,
Base-Softmax plus incremental-Sigmoid routing, selective training and freezing,
historical feature-space preservation, semantic-guided visual prototype
calibration, and one global visual prototype classifier.

Lifecycle:

1. The frozen CLIP ViT reference stream provides stable per-layer CLS inputs
   for RD detection, routing, and all expert generations. The adapted stream
   receives the routed residuals. The backbone remains frozen; historical
   experts and router columns remain frozen but stay active in every forward.
2. Base experts and the Base router are trained on Base classes. Base RDs are
   fitted using responsibility-weighted Base CLS features. Their reconstruction
   error means and standard deviations are estimated from a deterministic,
   class-balanced Base reference set and are then frozen. This reference set
   also supports later feature-space diagnostics; it is not used to construct
   pseudo-Sessions or support-size-specific expansion thresholds.
3. For an incremental Session and layer (l), every deployed RD (k) computes
   a standardized reconstruction error for each support source:
   (z_{i,l,k}=(e_{i,l,k}-mu_{l,k})/(sigma_{l,k}+epsilon)).
   The Session statistic for each RD is the mean over all Session sources:
   (ar z_{l,k}=N^{-1}sum_i z_{i,l,k}). The best historical explanation is
   (k^*=argmin_kar z_{l,k}). Reuse the existing pool when
   (ar z_{l,k^*}leq1.0); expand exactly one expert when
   (ar z_{l,k^*}>1.0). Labels validate the Session manifest but do not
   create class-wise expansion decisions.
4. On expansion, the selected historical expert and its RD are the same
   migration anchor. The new functional expert copies the selected expert's
   parameters and hidden width exactly; the new RD starts from that RD function
   and adds its low-rank residual branch. The new independent sigmoid Router
   column starts with a low activation bias. Train the new expert, new Router
   column, and new RD together. The incremental objective combines global
   visual-prototype classification, L2-SP to the immutable expert-initialization
   snapshot, historical final-feature replay, and RD reconstruction. Historical
   experts, historical Router columns, historical RD statistics, and the ViT
   remain frozen. The new RD's error statistics are calibrated on deterministic
   current-Session reference CLS after fitting and then frozen.
5. Reuse means no new modules and no historical parameter updates. All existing
   experts remain active in the deployed forward path. Expansion is layer-wise,
   and each layer can add at most one expert in a Session.
6. After adaptation, novel visual prototypes are calibrated using the existing
   Base-semantic retrieval rule. Semantic descriptions retrieve related Base
   description semantics and their corresponding Base visual prototypes;
   semantic confidence and support-prototype reliability determine the mixing
   strength. Historical visual prototypes remain unchanged.
7. All seen-class visual prototypes are merged into one global visual
   classifier. Training uses source-exclusive leave-one-out novel prototypes
   to prevent support-query leakage. Inference uses one current CLIP+MoE feature
   path and the global visual prototype bank. AE is not used at inference; there
   is no task ID, task gate, task-specific classifier, or text classifier.

Run architecture tests:

~~~bash
python -m unittest discover -s tests -p "test_risk_guided_moe.py" -v
python -m unittest discover -s tests -p "test_session1_diagnostics.py" -v
python -m unittest discover -s tests -p "test_config_schema.py" -v
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

