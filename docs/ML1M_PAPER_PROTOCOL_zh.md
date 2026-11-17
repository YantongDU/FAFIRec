# MovieLens-1M 本轮实验协议

本轮执行最新版线性诊断方法，并沿用原实验章节仍适用的设置。它用于实际测试新方法，不是对旧表 II 数值的严格复现。此前 Pantry 项目及结果保留在独立目录，本轮工作目录为 `E:/论文/FAFIRec_ML1M`。

## 已明确采用的设置

| 项目 | 本轮设置与依据 |
|---|---|
| 正反馈 | 评分 >3，即评分4或5，与论文一致 |
| 划分 | 用户已确认按每个用户的时间顺序划分；相同时间戳按源文件行序。实际有4542处划分边界时间戳相同，因此时间顺序为非递减，不保证严格晚于 |
| 比例与舍入 | train=floor(0.8n)，validation=floor(0.1n)，test=剩余；保留完整验证/测试段。逐用户取整后，总体比例为79.587%/9.532%/10.881% |
| 用户资格 | 训练至少2条、验证和测试各至少1条，且有训练未见物品可供负采样；没有额外迭代k-core |
| 冷/暖分组 | 按完整训练计数排序，底部/顶部各10%；并列按数字用户ID稳定排序 |
| 主要指标 | 每用户多目标 Recall@10、NDCG@10，按用户宏平均；不能把Hit@10当多目标Recall |
| 模型选择 | 验证 NDCG@50；连续10轮不提高早停；学生最多150轮 |
| 训练 | d=64、batch=2048、Adam、lr=0.001，与论文主要设置一致 |
| 对齐权重 | lambda_a=0.3，与原实验段默认值一致 |
| 权重衰减 | Adam的coupled L2为1e-5，位于论文的1e-6/1e-5/1e-4范围；本轮未声称完成该范围的超参搜索 |
| 重复运行 | 2025--2029五个独立训练种子；报告均值和样本标准差 |
| 对照 | CF、均匀融合、完整诊断控制；共享教师初始化、文本特征、负样本/批次流和验证选择规则 |

## 必须明确记录的补齐与差异

- 原稿没有固定文本编码器checkpoint；本轮选择官方公开的 [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)，固定revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。这是真实decoder-LLM语义特征，不是词汇特征，也不是PAD原始LLM2Vec张量。
- 仅编码静态电影标题（含年份）和类型，无query指令；左padding、最后有效token池化、L2归一化。全部3883电影已生成1024维特征，无截断。PAD行0为零，逐行MovieID和所有哈希已保存。
- 序列长度50、1层、2个注意力头、dropout0.1、教师最多30轮，是原文未锁定的补齐设置。教师亦仅按验证指标选择。
- 候选集为全部电影元数据目录，排除当前已观察历史；不按真实目标修改候选集。验证观察train，测试观察train+完整validation；测试目标不进入输入或诊断缓存。
- 全部3883部元数据电影中，351部在保留用户的正反馈事件中从未出现；这些电影仍属于固定候选全集，不能把本轮候选范围描述为仅含有保留正反馈的电影。[数据准备与目录范围](E:/论文/FAFIRec_ML1M/prepare_ml1m_paper.py:162)
- `uniform`固定诊断得分为0、可靠性为0.5，并使用均匀的用户对齐权重；用户融合门仍含可训练的全局维度偏置 `gate_bias`，物品侧融合门也可训练，因此不应描述为最终表示始终按0.5/0.5平均。[可靠性与融合门](E:/论文/FAFIRec_ML1M/fafirec/model.py:218)、[对齐权重](E:/论文/FAFIRec_ML1M/fafirec/model.py:300)
- 固定原始ML-1M发布数据按已给规则得到5950用户、3883物品、574619正反馈事件，与原表的3132/3354/156507不同。原稿没有给出进一步筛选与子集索引，因此不应强行解释为同一预处理数据。
- users.dat的人口属性可用且已记录，但最新版模型使用物品文本，不额外消费用户人口特征。
- 本轮使用RTX4060Ti、PyTorch2.6.0+CUDA12.6；原实验段的RTX8000/PyTorch2.0不在本机。训练保持FP32，LLM编码单独使用FP16。
- 新方法使用零初始化线性控制、固定核均匀混合与固定诊断缓存；原实验段中的非线性可靠性、EMA刷新、可学习核权重及额外门控正则尚未与新版方法统一，本轮不重新引入旧机制。
- 这三个匹配对照检验当前方法，并未实现原表全部11个竞争模型，不能据此宣称达到原表的SOTA结论。

## 数据与特征溯源

GroupLens官方数据主机在本机的证书链无法验证，本轮没有关闭TLS验证；使用[公开归档](https://zenodo.org/records/14467693)中与公开MD5完全一致的原始ZIP。原始文件哈希、下载来源和数据资格计数见 `data/raw/download_provenance.json` 与 `data/ml1m_paper_ratio/manifest.json`。数据及模型权重均位于被Git忽略的data目录。

特征文件为 `data/ml1m_qwen3_06b.pt`，形状3884x1024。映射、模型revision、模型权重和生成器SHA、编码版本及GPU使用量见同名manifest与 `data/qwen_encode_gpu_metrics.json`。

只读独立核查记录见 [ML1M_RUN_AUDIT_zh.md](E:/论文/FAFIRec_ML1M/docs/ML1M_RUN_AUDIT_zh.md)。本轮训练/验证/测试事件数分别为457322/54770/62527，总计574619；上述总体比例和同时间戳边界均由原始数据独立重建核实。[排序、取整及划分实现](E:/论文/FAFIRec_ML1M/prepare_ml1m_paper.py:105)

## 运行

```powershell
.\.venv\Scripts\python.exe train_ml1m.py --config configs/ml1m_paper_qwen3.json
.\.venv\Scripts\python.exe train_ml1m.py --config configs/ml1m_paper_qwen3.json --resume
.\.venv\Scripts\python.exe analyze_ml1m.py runs/ml1m_paper_qwen3_v1
```

独立重载使用 `evaluate_ml1m.py`。结果表、逐用户目标集合及推荐列表、训练日志、检查点和固定诊断缓存保存在对应runs目录。未完成的五种子网格不能作为完整均值表汇总。
