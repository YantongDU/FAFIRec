# MovieLens-1M 本轮运行独立审计

核查日期：2026-11-17。对象为 `E:/论文/FAFIRec_ML1M` 的 `ml1m_paper_qwen3_v1` 实验。本次核查只读取源码、原始数据、准备数据、特征及运行溯源信息；未读取测试指标用于调参，未修改Python、配置或运行输出，也未使用GPU。

**结论：未发现需要中止当前训练或作废本轮结果的实质性正确性问题。** 本结论针对实现与数据完整性，不代表已完成五种子结果验证，也不支持旧表II复现或SOTA结论。

## 已核实的关键环节

| 核查项 | 结果与依据 |
|---|---|
| 原始数据与时间划分 | 以独立标准库流程从1000209条原始评分重建全部5950用户序列及3883部电影元数据，与准备文件逐条完全一致。评分门槛为≥4，无重复用户–电影评分，也无同一用户跨train/validation/test的物品交集。[筛选、排序与划分](E:/论文/FAFIRec_ML1M/prepare_ml1m_paper.py:105) |
| 目标与候选隔离 | 验证输入为train，测试输入为train+完整validation；保留完整目标集合。候选排除完整已观察历史，不根据真实目标恢复候选。[批次构造](E:/论文/FAFIRec_ML1M/fafirec/ratio_data.py:112)、[候选传递](E:/论文/FAFIRec_ML1M/fafirec/ratio_runner.py:143) |
| 多目标指标与分组 | Recall分母为不同目标物品数；NDCG的IDCG使用min(K,目标数)，按用户宏平均。冷/暖是训练量排名底部/顶部10%的相对分组。[指标](E:/论文/FAFIRec_ML1M/fafirec/multitarget_metrics.py:117)、[分组](E:/论文/FAFIRec_ML1M/fafirec/data.py:40) |
| 模型选择与优化 | 教师和学生仅以验证NDCG@50选模；学生最多150轮、耐心10轮。实际优化器为Adam，coupled L2为1e-5，λa=0.3。[优化与选择](E:/论文/FAFIRec_ML1M/fafirec/ratio_runner.py:174)、[配置](E:/论文/FAFIRec_ML1M/configs/ml1m_paper_qwen3.json:28) |
| 共享教师与固定诊断 | 初始化复制协同参数，不共享参数存储；诊断构造仅读取训练序列，使用验证选出的教师。缓存绑定教师参数、用户顺序、数据和特征身份，学生绑定缓存张量及文件哈希。[复制](E:/论文/FAFIRec_ML1M/fafirec/model.py:417)、[训练诊断](E:/论文/FAFIRec_ML1M/fafirec/diagnostics.py:93)、[运行绑定](E:/论文/FAFIRec_ML1M/fafirec/ratio_runner.py:283) |
| 文本特征与源码身份 | 实际特征3884×1024、float32、冻结、全有限；PAD行全零，3883部真实电影向量为单位范数。MovieID行映射、特征文件、生成器及当前运行源码哈希均匹配manifest；来源是固定revision的Qwen3-Embedding-0.6B。[映射与哈希校验](E:/论文/FAFIRec_ML1M/fafirec/data.py:67)、[生成溯源](E:/论文/FAFIRec_ML1M/generate_ml1m_text_features.py:250) |
| 目录投影缓存 | 无偏置线性投影与查表交换顺序，缓存在每次forward结束后清除；已检查现有等价性测试覆盖输出及重复物品的参数梯度。[缓存实现](E:/论文/FAFIRec_ML1M/fafirec/ml1m_model.py:54)、[等价测试](E:/论文/FAFIRec_ML1M/tests/test_ml1m_model.py:9) |

## 报告必须保留的边界

1. 实际统计为5950用户、3883部电影、574619条正反馈；train/validation/test分别为457322/54770/62527。逐用户取整使总体比例为**79.587%/9.532%/10.881%**，不能声称总事件数精确按80%/10%/10%划分。[划分约定](E:/论文/FAFIRec_ML1M/prepare_ml1m_paper.py:145)
2. 共**4542处划分边界时间戳相同**，相同时间戳按原始文件行序排列；“时间顺序”指时间戳非递减，不保证验证和测试事件严格晚于前一段。[排序规则](E:/论文/FAFIRec_ML1M/prepare_ml1m_paper.py:105)
3. 固定候选全集含全部元数据电影，其中**351部没有保留正反馈事件**；已见历史仍按既定规则排除。[目录范围](E:/论文/FAFIRec_ML1M/prepare_ml1m_paper.py:162)
4. `uniform`固定的是诊断得分0、可靠性0.5和用户对齐权重的均匀性；用户融合门为含可训练全局维度偏置的 `sigmoid(gate_bias)`，物品侧门亦可训练。因此它不是始终严格0.5/0.5的最终融合。[门控公式](E:/论文/FAFIRec_ML1M/fafirec/model.py:218)、[物品融合](E:/论文/FAFIRec_ML1M/fafirec/model.py:284)

数据人口、文本checkpoint、硬件和新版方法与旧实验章节的差异继续以[本轮实验协议](E:/论文/FAFIRec_ML1M/docs/ML1M_PAPER_PROTOCOL_zh.md:22)为准；不应将本轮受控比较直接替换为旧表II结果。
