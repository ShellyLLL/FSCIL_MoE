# Unified FSCIL Dynamic-MoE Architecture

## Status

This branch now implements a single-current-encoder FSCIL path. It deliberately does not implement class-masked BiMC†, MGC, CMNN, per-class encoder versions, task gates, expert descriptors, or Top-K routing. Architecture acceptance and benchmark performance are separate; no AA/PD improvement is claimed until the complete test suite and controlled experiments pass.

## 1. One deployed prediction function

For every query image and every seen class, use the same current visual encoder and one fused prototype bank. The final cosine score normalizes the query and class prototype:

```
visual prototype -> visual intra-modal calibration
class-name text + description center -> text intra-modal calibration
calibrated visual + calibrated text -> inter-modal fusion
query image -> current enhanced visual encoder
score[c] = cosine(query_feature, fused_prototype[c])
prediction = argmax over all seen classes
```

For class c:
[
\mu_c^I=\operatorname{Norm}\left(\frac{1}{n_c}\sum_i\operatorname{Norm}(F_t(x_{c,i}))\right).
]
For novel classes, visual calibration uses only stored base visual prototypes:
[
s_{b,c}=\operatorname{Softmax}_{b\in Y_0}(\tau\cos(\mu_b^I,\mu_c^I)),
\quad
\tilde\mu_c^I=(1-\lambda_I)\mu_c^I+\lambda_I\sum_{b\in Y_0}s_{b,c}\mu_b^I.
]
Text calibration follows the patent's linear equation without normalizing the calibrated vector:
[
\tilde\mu_c^T=(1-\lambda_T)w_c+
\lambda_T\operatorname{Norm}\left(\frac{1}{n_c}\sum_j\operatorname{Norm}(g(t_{c,j}))\right).
]
Inter-modal fusion preserves the linear-combination magnitude; normalization occurs only inside the final cosine score:
[
\mu_c=\beta\tilde\mu_c^T+(1-\beta)\tilde\mu_c^I.
]
The only classifier is the scaled cosine scorer:
[
S_c(x)=s_{CLIP}\,\cos(F_t(x),\mu_c),\qquad
\hat y=\arg\max_{c\in Y_{\le t}}S_c(x).
]
The positive scale is shared by training, risk validation and inference; it leaves the cosine argmax unchanged.
There are no base/novel-specific score branches, category masks, covariance/nearest-neighbor score ensembles, session gates, or versioned class columns.

## 2. Two visual streams and MoE adapter

The frozen CLIP ViT is executed as two coupled streams. Every router receives the L2-normalized frozen reference CLS feature; stored historical raw-reference prototypes use the same normalization. Base experts use zero-initialized up-projections, and each incremental expert uses a zero up-projection plus a small independent Sigmoid gate at insertion, so insertion preserves the deployed function exactly.

The two streams are:

- Reference stream: the exact frozen ViT path without adapters. It supplies the raw CLS feature at each inserted block. It is the only input to all expert routers and APC raw prototypes.
- Enhanced stream: the same frozen Transformer blocks with residual adapters inserted. It carries the task-adapted representation used for the final CLIP image feature.

At a block, experts operate on the normalized enhanced hidden sequence. Base experts share one Softmax pool:
[
a^0_{l,k}=\operatorname{Softmax}_k(R_l^0(r^l)).
]
Every incremental expert has its own independent Sigmoid router:
[
g_{l,s}=\sigma(R_{l,s}^{new}(r^l)).
]
The pools are not jointly normalized and neither uses Top-K:
[
\Delta h^l=\sum_k a^0_{l,k}E^0_{l,k}(h^l)
+\sum_s g_{l,s}E^{new}_{l,s}(h^l),\quad
h^l_{out}=h^l+\Delta h^l.
]
New experts have zero-initialized up-projections, so appending an expert initially preserves the deployed function.

## 3. Expansion controller

Base-session raw reference prototypes are stored for each inserted block. A fixed per-layer threshold is the upper quantile of each base class's nearest other-base-class cosine similarity. During session t, each novel raw reference prototype is compared against every stored historical raw prototype. A layer becomes a candidate only when the maximum new-to-old similarity exceeds its base threshold.

APC only proposes candidate layers. It does not commit expansion based on prototype similarity alone.

For each source-grouped fold, APC proposals are computed from that fold's fitting sources only. The fold's own proposed layers are jointly trained and evaluated on its untouched held-out sources against the no-expansion baseline using the same all-seen-class cosine classifier. We do not intersect fold proposals, enumerate every layer subset, or select a winning subset on the same validation folds. A corrected resampled one-sided t bound is applied to the paired policy risk gains, including zero-gain no-expansion outcomes, to account conservatively for overlapping fitting folds. If accepted, the fixed APC rule is applied once to all current support sources and all resulting candidate layers are jointly trained. Per-fold candidates, baseline/expanded NLL, risk gains, uncertainty bound, and rejection reason are logged.

## 4. Training and activation/freeze policy

Base session:
- Freeze the CLIP visual and text encoders.
- Train only base experts and their shared Softmax routers.
- Build base visual prototypes and fixed raw reference prototypes after base training.
- Freeze all base experts and routers before incremental learning.

Incremental session:
- Freeze the CLIP backbone, text encoder, base experts/router, and every historical incremental expert/router.
- Add experts only at selected layers.
- Train only the newly added experts and their own Sigmoid routers.
- Classification uses current-session query embeddings against the full old-plus-new prototype bank. The query's own source is excluded from its current-class support prototype in both base and incremental optimization. The same scaled-cosine score function is used in base training, incremental training, candidate-risk evaluation and final inference. The positive CLIP scale does not change the cosine argmax, but keeps the cross-entropy risk definition consistent.
- Apply old-route suppression to the stored, L2-normalized historical raw-reference class prototypes for each selected layer. After training, audit the maximum new-expert gate over all historical class prototypes. If it exceeds `MAX_OLD_GATE`, roll back the entire expansion transaction.
- Freeze the new experts and routers after the session.

Historical class fused prototypes are retained as persistent classifier state. New class prototypes are generated by the current encoder. Historical image samples are not reloaded for incremental training or APC; only stored prototypes/anchors are used.

## 5. Historical protection: prototype-only constraint and limitation

The incremental stage never reloads old-session training images. It uses only immutable historical classifier prototypes and frozen-backbone raw-reference class prototypes captured when each class was learned. Therefore it cannot directly estimate the full historical image distribution or prove that every old-class query embedding is unchanged. The implemented protection is deliberately stated as a prototype constraint, not an exact invariance theorem:

- Historical experts and routers are frozen.
- New expert gates are penalized on stored historical raw-reference class prototypes and must satisfy the hard `MAX_OLD_GATE` bound after training.
- New-class queries are trained against every stored old fused prototype in the same all-class cosine loss.
- Historical class prototypes are not rewritten during later sessions.

This limits new-expert activation at the stored historical anchors, but does not mathematically guarantee zero representation drift for every unseen point in an old class. The post-session FSCIL test evaluation reports old-class accuracy and PD; it must be used to measure the actual residual forgetting. The candidate-selection folds directly measure held-out current-novel NLL against the full old-plus-new classifier, while historical gate suppression is a proxy constraint because old training images are unavailable. No claim of exact old-distribution preservation is made.

## 6. Checkpoint contract

Checkpoint schema 25 stores the current adapter topology and weights, immutable class prototype/text states, frozen-reference raw prototypes by layer, fixed APC thresholds, support manifest, metrics, and Python/NumPy/Torch RNG states. It does not store historical image embeddings or per-sample feature teachers. Resume restores adapter topology before loading adapter weights and restores the base visual prototype anchor from the base class state. Earlier schemas are intentionally rejected because their incremental objectives used historical-image replay and cannot be silently migrated.

## 7. Acceptance gates

- Base router weights are a Softmax distribution.
- Incremental gates are independent Sigmoid outputs and are not normalized across experts.
- No Top-K, descriptors, task gates, per-class encoder versions, category masks, MGC, or CMNN are in the execution path.
- New zero-up expert insertion preserves the current model output exactly.
- Only current-session experts and their own routers receive gradients in incremental training.
- Raw routing/APC features come from the frozen reference stream; expert transformations affect only the enhanced stream.
- Every query is scored against all seen class prototypes with one cosine formula.
- Source-grouped CV excludes each validation source from candidate training and its own class prototype.
- Candidate subset folds restore exact topology and parameter tensors before the next trial.
- Every cross-validation fold begins from the same captured Python, NumPy, Torch CPU and CUDA RNG states.
- All router inputs and stored raw-reference prototypes use the same L2-normalized feature definition.
- Base and incremental optimization use source-excluded class prototypes; no query contributes to its own class prototype.
- Checkpoint round-trip restores topology, weights, prototype state, APC thresholds and base calibration anchor.
- Full CUB200 experiments remain blocked until all architecture tests pass.

## Run

```bash
python -m pytest -q
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc.yaml
```
