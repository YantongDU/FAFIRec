# 文献驱动的 FAFIRec 改造与可证伪实验

更新日期：2026-11-18。工作树：`E:/论文/FAFIRec_Research_v3`；分支：`codex/literature-guided-fusion`。本文记录研究依据、实现约定与实验解释边界；目前没有本轮新实验结果。旧 `FAFIRec_Improve_v2` 的源码与结果保持冻结。

## 1. 要解决的问题

既有五种子实验中，验证集选出的强 CF 对照的测试 Recall@10/NDCG@10 为 **0.135447/0.135793**，旧改进版 FAFIRec 为 **0.132217/0.133300**。这些数值只说明当前协议下直接融合尚未超过强 CF；新设计与超参数不得依据旧测试分数选择。旧 diagnostics 的两个 CF 特征高度相关，完整模型也未证明优于 uniform，因此首先检验文本是否提供可利用的互补信息，再检验门控。

当前 ML-1M 数据为 5,950 用户、3,883 个元数据物品、574,619 个正反馈事件；按每用户时间顺序划分。静态文本只有 title/year/genre，使用已有冻结 Qwen3-Embedding-0.6B 的 1,024 维向量。本轮不把图像、多层语言模型隐藏状态或新增用户画像当作现有输入。

## 2. 核实过的 primary 来源与代码证据

以下均为论文原文、作者仓库或作者明确指向的复现；其他数据集上的论文结果不能直接作为本轮全目录评估的基准。

| 工作 | 论文与代码 | 已核实机制；本轮借鉴及限制 |
|---|---|---|
| **UniSRec，KDD 2022**，Towards Universal Sequence Representation Learning for Recommender Systems | [论文](https://arxiv.org/abs/2206.05941)；[官方 unisrec.py](https://github.com/RUCAIBox/UniSRec/blob/master/unisrec.py) | `PWLayer` 为可学习去中心投影，`MoEAdaptorLayer` 为多投影专家；有 sequence–item、sequence–sequence contrastive 预训练；`calculate_loss` 使用归一化、温度与全目录 CE。借鉴轻量语义适配和目标消融，未复制其跨域预训练/MoE。 |
| **VQ-Rec，WWW 2023**，Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders | [论文](https://arxiv.org/abs/2210.12316)；[官方 vqrec.py](https://github.com/RUCAIBox/VQ-Rec/blob/master/vqrec.py) | 文本→离散 PQ code→可学习 code embedding；`generate_fake_neg_item_emb` 提供半合成负例，`code_projection` 使用 Gumbel-Sinkhorn，downstream `calculate_loss` 使用 CE。原文讨论过度依赖文本的问题；本轮没有多域预训练，暂不引入完整 PQ/排列适配。 |
| **Recformer，KDD 2023**，Text Is All You Need: Learning Language Representations for Sequential Recommendation | [论文](https://arxiv.org/abs/2305.13731)；[作者仓库](https://github.com/JiachengLi1995/Recformer)；[作者指向的复现 models.py](https://github.com/AaronHeee/RecFormer/blob/main/recformer/models.py#L523) | 作者仓库明确说明由于 Amazon policy 暂不发布源码，并指向 AaronHeee 的 replication，后者不能称为作者官方实现。原文有 MLM+next-item contrastive 及缓存物品编码/两阶段训练；复现 `RecformerForSeqRec` 支持冻结候选向量与 full/sampled softmax。本轮只借鉴稳定候选空间与分类目标，不训练 Longformer。 |
| **LLM-ESR，NeurIPS 2024**，Large Language Models Enhancement for Long-tailed Sequential Recommendation | [官方 proceedings](https://papers.neurips.cc/paper_files/paper/2024/hash/2f0728449cb3150189d765fc87afc913-Abstract-Conference.html)；[全文](https://arxiv.org/html/2405.20646v2)；[官方 DualLLMSRS.py](https://github.com/Applied-Machine-Learning-Lab/LLM-ESR/blob/master/models/DualLLMSRS.py#L90) | 冻结语义表、可训练适配器、双视图与 logit-level fusion。Eq.5 的拼接点积等价于两个分数相加，**两个视图仍共享 sequence encoder**。本轮独立编码器/冻结 CF 是结合自身证据作出的改造。其原评估使用 100 个采样负例，指标不与本轮混比；没有复制用户检索自蒸馏。 |
| **PAD，SIGIR 2025**，Pre-train, Align, and Disentangle: Empowering Sequential Recommendation with Large Language Models | [论文](https://arxiv.org/abs/2412.04107)；[官方 Model_new3_3](https://github.com/Applied-Machine-Learning-Lab/PAD/blob/main/model/model_mmd_ada.py#L399) | 中间阶段为推荐锚定 MK-MMD；**最后阶段去掉 MK-MMD**，保留 ID-specific、text-specific、alignment 三专家，并按候选物品频率融合。代码有三个 user encoder，L614–633 按频次桶加权分数。保留独立 CF 能力、分数融合与频次控制由此得到设计启发；本轮不是 PAD 复现。 |
| **RLMRec，WWW 2024**，Representation Learning with Large Language Models for Recommendation | [论文](https://arxiv.org/abs/2310.15950)；[官方 LightGCN_plus](https://github.com/HKUDS/RLMRec/blob/main/encoder/models/general_cf/lightgcn_plus.py)；[loss_utils.py](https://github.com/HKUDS/RLMRec/blob/main/encoder/models/loss_utils.py) | contrastive 以相同用户/物品 ID 的跨视图表示为正对；也有 masked reconstruction 变体。`cal_loss` 按同一批 ID 索引两种表示，再计算 InfoNCE。**它不是匿名边际 MMD**。本轮只借鉴同物品身份对应，采用去重、单向 stop-gradient CF 锚定，不复制图模型、画像生成或完整训练目标。 |
| **IISAN，SIGIR 2024；IISAN-Versa，TKDE 2025** | [SIGIR 论文](https://arxiv.org/abs/2404.02059)，[DOI](https://doi.org/10.1145/3626772.3657725)；[TKDE 论文](https://arxiv.org/abs/2411.02992)，[DOI](https://doi.org/10.1109/TKDE.2025.3608071)；[官方仓库](https://github.com/GAIR-Lab/IISAN)，[IISANAdaptedMModel](https://github.com/GAIR-Lab/IISAN/blob/main/Code_Cached/model/model.py#L257) | 从冻结基础模型缓存隐藏状态，在旁路进行 intra/inter-modal adaptation；Versa 增加非对称编码器支持、层选择与维度适配。借鉴缓存与解耦训练原则。本轮只有单层最终 Qwen 物品向量，没有 image/text 双塔和多层 taps，不能将当前小投影称为完整 IISAN 或照搬其显存/加速比。 |

这些机制是已有研究贡献。本轮“冻结 CF anchor＋独立专家＋零初始化分数残差＋可选相对门控”是面向现有失败现象的组合适配；结果出来前不声称性能提升、SOTA 或方法新颖性。

## 3. 从表示混合改为可恢复 CF 的分数残差

旧方法对用户与候选分别进行 ID/text 混合，再计算点积，因而同时产生 CF–CF、text–text 和跨视图交叉项；共享编码器也同时接受推荐与对齐梯度。新分支保留旧协议、数据和验证选出的强 CF checkpoint，将其深复制后冻结，并始终置为 eval 模式。

设 CF 与独立专家的归一化用户/物品向量分别为 `c_u,c_i` 和 `z_u,z_i`，旧 anchor 的分数尺度为 `κ_cf`，专家尺度为 `κ_e`。首轮 BPR 配方取 8，CE 配方取 16；因此这两组比较同时改变目标和尺度，属于训练配方比较，不能孤立归因于 CE。新分数为：

```text
s(u,i) = κ_cf <c_u,c_i>
       + a · tanh(γ) · r(u) · f(i) · κ_e <z_u,z_i>.
```

`γ` 从 0 开始训练；`a≥0` 为验证选择的 calibration multiplier，其网格必须包含 0。实现通过拼接 `[c_u ; a·tanh(γ)·r(u)·(κ_e/κ_cf)·z_u]` 与 `[c_i ; f(i)·z_i]` 完成点积，**不对拼接后的向量再次 L2 归一化**。这样没有新增跨视图交叉项，CF 项的系数始终为 1。`a=0` 走 anchor 的直接分支，精确恢复原 CF；非零残差被验证选中后，测试集仍可能退化，不能把 fallback 性质写成泛化保证。

每个独立训练任务只挂接一个专家，对照类型如下：

- **ID residual**：独立可训练 ID 表与编码器，从相同 CF anchor 克隆，检验新增容量、继续优化与集成本身的收益。
- **Text residual**：冻结真实 Qwen 表＋可训练 centered affine adapter＋独立编码器。
- **Shuffled residual**：固定种子打乱真实物品与文本向量的对应关系，再按同一流程训练；保留文本分布和模型容量，破坏身份语义关联。

编码器均为 64 维、一层、从 CF 编码器克隆后独立训练，不与 anchor 或其他任务共享参数。Text/Shuffled 适配器只使用 train 中见过的物品拟合：

```text
t_i = (x_i − μ_train) W + b,
初始化最小化 Σ_i ||(x_i − μ_train)W + b − e_i^CF||² + ridge·||W||²。
```

截距不惩罚，中心/目标/计数仅来自 train，CPU float64 求解；随后投影可训练。此处借鉴去中心与适配思想，ridge 对 CF 空间初始化是本轮工程选择，不应归为 UniSRec 原算法。

## 4. 监督目标与候选池

专家阶段训练 `L = L_fused + 0.1 L_expert + λ_id L_identity`。默认 `L_fused/L_expert` 为相同负例上的 BPR；专家独立辅助项让 `tanh(γ)=0` 时语义分支仍有训练信号。归约继续按有效训练位置平均，未改成用户平衡。

`λ_id∈{0,0.01}`。将 batch 历史及正例中出现、且 train 见过的实际物品 ID 去重为集合 `I_B`，以同 ID 的冻结 CF 向量为正对，其他去重 ID 为负例，采用单向 InfoNCE（温度 0.1）：

```text
L_identity = mean_i[−log exp(<z_i,stopgrad(c_i)>/0.1)
                         / Σ_j∈I_B exp(<z_i,stopgrad(c_j)>/0.1)]。
```

这不是 marginal MMD；权重 0 是必需对照。Shuffled 也经过同一初始化与训练流程，因而该对照检验语义对应带来的增量，不是“完全不可学习”的随机噪声上限。

**CE 是匹配的目标消融。** 对 ID residual 与 Text residual 对称比较 BPR/CE；CE 的 fused 与 auxiliary 均使用同一规则：以全目录为基础，排除该用户 **full-train** 中其他正反馈物品，恢复当前监督正例，始终排除 PAD。每个有效位置分别归一化，通过 query-row 分块和 backward 重计算限制显存。这是本项目的 **masked catalog CE**，与标准 RecBole 全目录 CE 的负池不同，不能称为逐项复现 UniSRec/VQ-Rec。full-train 排除使用该用户训练窗口内的全部已知正例，可能包含相对该前缀的后续 train 事件，因此也不是严格在线的 prefix-only 负采样；不使用 validation/test 正例。目标消融必须保持两种专家的负池规则一致。

## 5. 三种路由及时间信息边界

| route | `r(u)` | `f(i)` | 目的 |
|---|---|---|---|
| global | 1 | 1 | 先验证文本分数本身是否互补。 |
| frequency | 1 | `1/(1+log1p(count_train(i)))` | 只由候选 train 计数抑制热门物品残差；没有使用待预测目标标签或测试频次。 |
| relative | `2·sigmoid(wᵀ x̃_u+b)` | 1 | 当前前缀条件下选择残差强度；与 frequency 是独立对照，不默认同时启用。 |

Relative gate 有四个严格由当前已观察前缀构造的特征：最近 4 个有效历史转移的 `CF probe loss − expert probe loss` 均值、标准差；当前两视图全目录预测熵差；`log1p(prefix length)`。两种 probe loss 和熵都用统一尺度 8，避免把不同温度误当作相对质量。每个历史转移使用 4 个固定、可复现的 probe 候选，排除该次已观察正例；这些诊断 probe 不等同于训练 BPR/CE 的 full-train 排除负池。

历史转移 `prefix_(t−1)→hist_t` 的结果在当前预测时已被观察，因此可进入统计；待预测的 `target_(t+1)` 及任何未来正例不得进入 gate。当前熵只比较两种模型的全目录分布，不索引目标。特征与 CF 目标均 detach，标准化统计只在 train 拟合，后续保持冻结。

先训练 global 专家，再冻结 anchor、专家编码器、适配器/ID表与 gain，仅训练 **4 权重＋1 截距** 的 router。线性层零初始化后 `2·sigmoid(0)=1`，恢复 global 强度作为起点。实现要求该阶段只接 global 专家，frequency 另作对照。若 global 选择 epoch0 且 gain=0，后续冻结 gain 的 router 也退化为零残差；这种情形须明确报告，不能当作门控效果检验。尽管前向特征满足 prefix 因果边界，专家本身已经由完整训练集拟合，历史 loss 仍是 **in-sample**；没有 cross-fitting、样本外误差标签或可靠性校准，不应称作“校准过的文本置信度”。

## 6. 可证伪假设与最小对照

| 假设 | 最小对照/观察 | 不支持时应如何解释 |
|---|---|---|
| 文本含有 CF 之外的可用信息 | 同 anchor、预算、目标下 Text vs ID residual vs Shuffled；同时报告 anchor；检查非零 `a` 是否被验证选中。 | Text 不超过容量匹配或打乱对照，就不能将收益归于语义；`a=0` 表示该配置验证不支持语义增量。 |
| 分数残差减少直接融合的破坏 | 初始化与 `a=0` 精确 fallback；与旧融合版对照，报告初始化分数/排序差异及训练成本。 | 只恢复 CF 而无额外收益，是稳定性改进，不是推荐精度创新。 |
| 频次或相对门控优于全局幅度 | 同一冻结专家比较 global/frequency/relative，分别保存每 seed 的 route、`a`、gain、验证分数与gate分布。 | 若无稳定增益，则保留更简单 global，不把 gate 激活变化当成性能证据。 |
| 对应身份监督或 CE 有独立收益 | `λ_id=0/.01`；ID/Text 的 BPR/CE 成对比较，保持候选池与训练位置权重一致。 | CE 在 ID 与 Text 上同样提升，是目标收益；identity 无增益则不支持对齐模块必要性。 |

先用验证集筛选小规模设计，再锁定统一配置完成 2025–2029 五种子；每种子保存所有 multiplier 的 calibration 轨迹及最终选择。旧强 CF 和 ID residual 是必要对照，Shuffled 防止将容量/优化效果误解为语义增益。除预先说明的每种子 calibration 外，模型级超参数仅按验证集选择；不得按测试结果挑种子、route 或对齐权重。

ID residual 的表参数约为 24.9 万，文本 affine 约为 6.56 万；且前者精确克隆 CF 表，后者仅由 ridge 近似。ID 是强继续训练/集成对照，不能称为严格参数量或初始分数匹配。Text 与 Shuffled 才共享相同参数化、预算和初始化流程；这组对照检验静态语义与物品身份的对应关系。用户 cold/warm 分组按训练历史长度定义，不等同于按物品频次定义的冷物品实验。

所有新旧 checkpoint、数据/特征/源码指纹、训练预算、运行时间、显存峰值及逐用户预测应可追溯。旧 test 已在多轮开发中使用，本轮只能称为**透明的迭代 benchmark**；即使后续超越旧分数，也不是 untouched holdout 的首次确认，不能据此单独宣布 SOTA 或 Strong Accept。

## 7. 论文需要同步修改的内容

1. **方法与符号**：以冻结 anchor、独立专家、分数残差及两阶段 router 替换共享表示混合公式；明确尺度、训练/冻结参数、零幅度 fallback 的适用范围。方法已经变化，不能沿用旧消融结论解释新模型。
2. **diagnostics 的定义**：从静态 CF 两特征改为 relative-prefix 四特征时，应重写计算时序、probe 和标准化；不再声称其是样本外可靠性或因果效果估计。
3. **对齐及来源归属**：MMD 改为可选同 ID InfoNCE，应重写损失及理论适用范围；删除由 characteristic kernel 推导而不再适用的论述，准确区分 PAD、RLMRec、LLM-ESR、IISAN 的原贡献与本轮组合适配。
4. **实验协议**：披露真实数据统计与按时间划分；此前以表 II 为准的稿件统计若无法由公开流程生成，需保留差异说明并用可复现统计更新，不伪造匹配子集。写清全目录多正例评估、cold/warm定义、full-train负池、masked CE和验证选参。
5. **主张强度**：目前无本轮结果；正式表格只放真实完成且可重放的运行。后续除均值/标准差，还应报告文本增量是否超过强 ID 和打乱对照、门控是否优于global，以及成本。统计显著性检验按已选协议如实报告，不能把五种子下某一种检验的分辨率限制泛化为所有检验。
