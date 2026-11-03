# FSCIL 动态 MoE：可验证的稳定性与尚待验证的收益

审查起点：`3f9cb99f35c853f4f8e93db7bb35819485a73039`。本分支是研究候选方案，
没有在 CIFAR-100、CUB-200 或 miniImageNet 上完成性能实验，不能称为已证实的最优方案。

## 1. 原实现的问题及修复依据

| 问题 | 可复核原因 | 本分支处理 |
| --- | --- | --- |
| 冻结旧参数不能保证旧输出不变 | softmax 增加一列改变所有旧权重的分母；上游新残差也会改变下游旧专家的输入 | 类别绑定创建时的逐层专家前缀，旧类始终使用自己的历史路径 |
| 旧原型与新查询表示不一致 | 原型保留历史值，但所有测试图像都使用最新路径 | 查询与该类别原型始终使用同一条路径 |
| 视觉原型被错误重构 | 归一化融合不可通过简单减去文本项再归一化求逆 | 原始/校准视觉原型与融合原型独立保存 |
| 增量目标与部署目标不同 | 训练只有视觉原型；部署还有视觉模态内校准、文本校准和模态融合 | 训练 LOO 原型采用同一套校准与融合，历史竞争类用各自历史路径评分 |
| 支持原型在整个增量训练中固定 | 专家更新后，查询空间变化而支持缓存不变 | 每个 epoch 重新编码支持集，使用 stop-gradient 原型进行交替优化 |
| 下游描述器使用漂移输入 | 描述器虽被冻结，但上游扩展改变其输入分布 | 每个描述器在它训练时的历史路径上计算覆盖分数 |
| 描述器异常被等同于分类收益 | 重构误差度量分布偏移，不等于分类错误或扩展后泛化收益 | 只有同一支持源同时存在覆盖不足和全已见类别分类负间隔才提出扩展 |
| 文档与实现不一致 | README 声称存在 NULL、hard Top-1、回滚、CLI 和诊断工具，源代码未提供 | 文档改为实际实现；补充 auto/none、候选事务与 AA/PD |

SEMA [1] 为描述器和动态扩展提供先例；它没有给出“小样本下分布偏移必然需要扩展”
的结论。MoE-Adapters++ [2] 使用任务相关路由与分布选择器，也不能证明在 5-shot
下任务选择可靠。BiMC [3] 提供模态校准方法，但其冻结表示设定不能直接替代
可变视觉编码器的稳定性分析。稀疏 MoE [4] 的容量/计算优势也不是 FSCIL 泛化保证。

## 2. 采用的结构：类别绑定历史路径

令类别 c 建立时，在第 l 层已有 K(l,c) 个专家，保存向量 K(c)。
后续只追加专家和独立路由列，基类训练后冻结旧专家、旧列、Adapter LayerNorm、
残差尺度及全部 CLIP 参数。

对类别 c 的评分是：

`s_c(x) = cosine(f_{K(c)}(x), p_c)`

其中 p_c 是建立该类别时保存的 BiMC 融合原型，f 的每层 softmax 只在
前 K(l,c) 个专家上归一化。推理遍历所有不同路径，对所有已见类别统一 argmax。
类别的出生信息是分类器元数据；不读取测试样本标签，不先预测或使用测试任务 ID。
同一路径的多个类别共享一次前向。

**稳定性证明的范围：** 从输入开始逐层归纳，历史路径中每层的参数、归一化范围、
输入均不变，因此历史路径输出不变，旧原型不变，旧类别分数不变。
这要求确定性评估和冻结的旧模块；浮点后端的非确定性仍可能带来舍入差异。
本分支的 CPU 回归测试对多层扩展后的旧类分数做零容差比较。

**不成立的推论：** 旧分数不变不意味着旧类准确率不变。新增类别可能取得更高分，
导致 old→new 竞争错误。PD 还受新增类别难度、类别比例影响，并非纯粹的遗忘度量。
新旧路径虽然共用 CLIP 嵌入维度、单位归一化与文本锚点，但分数分布未必等价；
跨路径可比性必须用基类伪增量验证和正式消融检查，不能宣称已经校准完善。

**代价：** 保留的是专家前缀元数据，不是多份 ViT 权重或旧图像；但推理需要 P 个
不同路径的前向，P 最坏随接受扩展的会话数增长。本实现没有共享前缀激活缓存，
不能宣传为 constant-cost inference 或稀疏 Top-1 MoE。

## 3. 从训练到分类的闭环

1. 基类阶段：冻结视觉主干和文本编码器，训练基类专家/路由、Adapter LayerNorm
   与残差尺度。使用现有 CE、与原始 CLIP 的特征蒸馏及负载均衡。结束后冻结，
   拟合并校准描述器，重新提取最终原型，记录历史路径。
2. 增量阶段：只读当前会话支持集及历史原型/模型，不回放旧类原图。
   用当前路径建立按 `source_id` 排除查询所有视图的 LOO 原型。
   视觉校准、文本校准、模态融合与部署一致。
3. 全已见类别竞争：新类分数由当前路径计算；旧类分数由各自固定路径计算。
   所有分数使用同一温度。类 ID 显式映射到列，不能把全局标签直接当局部下标。
4. 需要扩展时：仅新增专家和对应路由列可训练。每个 epoch 刷新支持原型，
   其余参数冻结；训练和推理均为 soft routing，不切换 hard Top-1。
5. 完成后：用确定性支持提取流程重算 LOO 诊断。仅当 CE 严格下降且准确率不下降
   时保留候选，否则恢复参数、拓扑和候选阶段 RNG。异常同样回滚后抛出。
6. 保留候选后拟合新描述器，再构建最终 full-support 类别原型，冻结该类别路径。
   checkpoint v7 保存路径、拓扑、支持清单、原型、未舍入准确率和 RNG。

支持集每轮刷新是交替的 stop-gradient 优化，不是对所有支持编码反向传播的精确
元学习目标。同一 epoch 内原型仍是该轮开始的快照，不能称为逐步完全同步。

## 4. 扩展判据的性质与限制

对源样本 i，取全已见类别融合分类器的真实类别与最大竞争类别的差 m_i。
对层 l，每个描述器 e 在自身历史路径上得到 z(l,e,i)。定义：

`d_i = max(0, -m_i)`

`u(l,i) = max(0, min_e z(l,e,i) - z_threshold)`

`D(l,c) = mean_{i in class c} d_i * u(l,i)`

存在正 D 才提出候选。默认每会话最多提出一个候选，在所有层上选择最大类别
D 对应的层；这是显式容量预算。`z_threshold=1` 沿用分布描述器方法中的经验值，
并非具有显著性水平的统计检验。D 的乘积、最大类别排序和单候选预算也是需要消融
验证的设计假设，不是理论最优性结论。不能把这些超参数说成完全免调参。

LOO 只排除了查询进入自己的原型。网络参数仍见过这些支持源，候选也依据这些源
选择。因此候选接受条件只是**支持集拟合诊断**，不是独立验证集、交叉拟合风险估计
或泛化保证。5-shot 描述器校准样本极少，统计不确定性尤其明显。
真正的方案比较应在基类内部构造独立伪增量任务来选择阈值、训练轮数、容量预算、
学习率和模态融合权重，随后全部锁定；不得使用正式增量测试集挑参数。

本分支没有另加原型排斥、旧原型充当旧图像、测试集调阈值或精度目标控制。
旧的可训练 prototype refinement 与 conflict repulsion 开关在该协议下明确报错，
因为它们假设单一查询空间，不能静默叠加到历史路径分类器上。

## 5. 评测和可证伪实验

每个会话 i 在所有已见类别的测试集上计算 Top-1 A_i：

`AA = sum(A_0, ..., A_B) / (B + 1)`

`PD = A_0 - A_B`

准确率用百分数时，PD 单位是百分点，不除以 A_0，不取绝对值。保存完整精度，
仅显示时舍入。中途 `--end_session` 的统计标为 partial，不能替代完整 AA/PD。

最少比较：

| 实验 | 目的 |
| --- | --- |
| 冻结 CLIP + BiMC（关闭 VISUAL_MOE） | 检查新增训练是否真的优于无需适配的基线 |
| 相同 Session 0 checkpoint + `--expansion_mode none` | 隔离新增专家的净收益 |
| 相同 Session 0 checkpoint + `--expansion_mode auto` | 测量动态扩展的收益、参数与推理代价 |
| 对上游实现做独立复现 | 比较共享最新路径与历史路径，不能混用 checkpoint schema |

同一组方法复用支持 manifest；多个配对种子。报告逐会话 A_i、AA、PD、最终准确率、
基类/旧新类/当前新类准确率、old→new 错误流、接受/拒绝的扩展数、参数量、路径数和
推理耗时。不能通过压低 A_0 来“改善 PD”，也不能只报告最佳种子。
这些真实实验尚未运行，特别是跨路径校准与计算代价可能使该候选不优于固定基线。

## 6. 已运行验证与环境限制

`python -m pytest -q`：14 项测试通过，包括：

- 上游和下游同时扩展并强扰动新增参数，旧类别分数逐位不变。
- 路径上下文在非法路径和前向异常后正确恢复。
- 相同 source 的多个视图全部排除；单样本类不能伪造有效 LOO。
- 真实小型 ViT 增量反向传播，训练 logits 与双级校准公式一致。
- 候选接受、拒绝及训练异常的事务行为；拒绝后完整模型和 RNG 恢复。
- 视觉原型不被融合反演污染。
- checkpoint 路径/分数/指标恢复，旧 schema 拒绝。
- 三会话小型 ViT 连续运行和 Session 0 后恢复，最终全部模型张量及准确率一致。
- AA/PD 定义、负 PD、非法输入和 no-expansion 分支。

测试环境：Python 3.12、PyTorch 2.15.0+cpu、torchvision 0.29.2+cpu。
还进行了 compileall、CLI help 和 git diff --check。
合成数据测试只证明流程/不变量，不构成真实数据性能结果；未运行完整 ViT-B/16、
CUDA 或原 README 指定的 PyTorch 1.13.1 环境。

## 7. 主要依据

1. Wang et al., *Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning*, CVPR 2025.
   https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Self-Expansion_of_Pre-trained_Models_with_Mixture_of_Adapters_for_Continual_CVPR_2025_paper.pdf
2. Yu et al., *MoE-Adapters++: Toward More Efficient Continual Learning of Vision-Language Models Via Dynamic Mixture of Experts*.
   项目所附论文，重点参照 Dynamic Mixture-of-Experts Adapters / LEAS / DEeC。
3. Chen et al., *Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration*, CVPR 2025.
   https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Enhancing_Few-Shot_Class-Incremental_Learning_via_Training-Free_Bi-Level_Modality_Calibration_CVPR_2025_paper.pdf
4. Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, ICLR 2017.
   项目所附论文。

历史路径评分是本分支提出的结构选择，不归因于上述论文已有结论。
