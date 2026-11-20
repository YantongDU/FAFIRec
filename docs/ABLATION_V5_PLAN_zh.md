# FAFIRec改造版的消融计划

本轮针对已冻结的五种子V4模型，不修改其骨干、协同算子、数据划分、候选集、诊断代理或历史结果。主方案是候选相关门控，推断系数alpha=0.025、beta=0、mu=0。消融不再搜索超参数，也不按新测试分数选择方案。

## 重新训练的机制消融

每个变体在2025–2029五个种子上使用相同的训练内部校准集，固定20轮、Adam学习率0.02、batch256、训练alpha=0.1/beta=0.2、L2系数0.001。保留原17个参数槽及注册顺序，正则始终是17个参数平方的均值；被禁用的路径保持零。这样不会因减少参数而偷偷改变正则强度。

| 变体 | 移除或限制的机制 |
|---|---|
| full | 完整模型复现；要求与旧V4权重、日志和排名一致 |
| gain_only | 只保留全局门控偏置，检验幅度校准是否已足够 |
| length_candidate | 只向门控提供显式历史长度；候选频次和冲突项保留 |
| user_only | 保留用户诊断，关闭候选频次与冲突项 |
| no_past | 屏蔽历史NLL差均值/标准差、probe可用比例和历史排名差 |
| no_current | 屏蔽用户级当前熵差与top20重叠摘要；不移除候选冲突项 |
| no_length | 仅屏蔽显式历史长度，其余诊断仍可能间接包含历史信息 |
| no_pop | 关闭整项用户条件化候选频次作用 |
| no_conflict | 关闭候选分数冲突作用 |

特征屏蔽在旧训练统计标准化后置零，并同时作用于user线性项和候选频次斜率路径。原模型训练系数与推断系数不同，这一配方在消融中保持一致。

## 仅推理敏感性

这些干预使用冻结的完整门控，不重新训练，因此单独列表，不冒充训练消融：

1. 七维诊断替换为训练均值，即标准化后为零。
2. 在训练历史长度分箱及probe可用数内联合置换质量诊断，保留长度和缺失指示；固定用户置换，不读待预测标签。
3. 用训练观察前缀上校准的一个平均gate替代全部门控变化。
4. 对当前前缀的未观察候选求gate均值，广播到该用户所有候选，仅去掉候选异质性。

记录有效混合系数alpha×gate的分布，以及top50完整列表与原模型一致的用户比例。它们说明已训练模型的依赖性，不能单独证明移除机制后重新训练的性能。

## 主骨干文本对应关系

额外在相同128维、2层、原始catalog CE和全转移窗口训练预算下，打乱ACE初始化物品表的非PAD行。使用独立局部随机数发生器，保持编码器及位置参数的初始化、训练随机数、表的整体矩和奇异值结构一致，只破坏文本向量与物品身份的对应。每个种子仍按既定验证规则选择epoch。

这一项准确称为“主骨干文本–物品对应关系打乱”。其它协同分支和原诊断代理保持冻结，因此它没有移除系统内所有文本信号，不能写成完整的w/o LLM实验。最终替换主骨干分数时，也不重新训练或调优其余门控。

## 复用对照与解释边界

- 去协同分支对应已有ACE全覆盖结果，可复用。
- 主方案mu本来为0，不能制造一个完全相同的w/o alignment行；对齐仅分析已完成的相同其它参数下mu=0/0.1配对。
- 原ACE到全转移ACE同时改变覆盖、窗口边界上下文及实际优化步数，准确称为训练布局配方比较。
- 删除分数标准化而固定混合系数会破坏量纲。已有固定系数、gain-only和平均gate对照用于检验幅度解释；不将这种单位失配包装成语义必要性证据。
- 外层测试此前已多次观察，本轮是冻结配方下的机制分析，不是独立新数据的确认实验。

## 核验与统计

训练完成后锁定全部五种子的消融权重及来源文件，再评分测试。完整模型复现先检验旧参数和训练日志逐位相同；验证/测试排名重载及CSV指标独立重算。测试使用两个独立重载过程。

主关注比较预先固定为full减gain_only的整体NDCG@50。8个重训消融和4个敏感性实验的整体NDCG@50共12项统一Holm校正。其它分组和指标提供五种子均值、样本标准差、配对差、描述性seed-bootstrap区间及胜负数，不逐表挑选显著结果。五种子双侧精确符号翻转检验最小非零p为0.0625；不添加显著性星号。若差值很小、区间跨0或去掉模块反而更好，应明确报告，不强行为每个设计寻找正面解释。

## 复跑入口与产物

先按V4运行说明保留五种子组件、缓存、验证锁、输入数据及实际文本嵌入。本轮会验证这些输入的指纹；公开仓库没有捆绑私有运行产物，不能只复制下列命令而缺少前置输入。使用原ML-1M环境，设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`。原组件目录默认沿用V4运行的本地路径，异机需要提供相应的原始产物和路径配置。

```bash
python experiments/ablation_v5/experiment.py --baseline-root /path/to/diagnostic_v4_final --output-root runs/ablation_v5 --parallel 1
python experiments/ablation_v5/shuffled_backbone.py train --output-dir runs/ablation_v5_shuffled_backbone
python experiments/ablation_v5/shuffle_evaluation.py --root runs/ablation_v5 --shuffle-root runs/ablation_v5_shuffled_backbone
python experiments/ablation_v5/report.py --run-root runs/ablation_v5 --shuffle-root runs/ablation_v5_shuffled_backbone --export-dir /path/to/new-report-dir --local-report /path/to/ablation-report.md
```

前两个训练入口可以在显存充足时各运行一个进程；后续对应关系评分须等待两套完整五种子锁。训练重启使用各入口的`--resume`，只接受原配置、原源文件及原输入。评分入口会核对已有结果并继续未完成部分。报告导出要求新目标文件，防止覆盖已审阅的结果；克隆已含结果的仓库后，请显式提供新的`--export-dir`和`--local-report`。

`runs/ablation_v5/combined_validation.json`、`combined_test.json`与两套verification记录保留全部重训/敏感性结果；`shuffled_identity_native_v2/`保留文本身份干预。单骨干行沿用其原生原始打分，融合行沿用旧融合标准化打分，以保留各自既定数值路径；第一版中发现的极近分数舍入差异及未完成输出保留供追溯。逐用户CSV、重载结果、模型与输入指纹留在本地运行目录。公开导出包括中文分析、五种子JSON和可插入Overleaf的LaTeX表格，分别在`docs/ABLATION_V5_RESULTS_zh.md`、`docs/ABLATION_V5_RESULTS.json`及`docs/ABLATION_V5_TABLE.tex`。
