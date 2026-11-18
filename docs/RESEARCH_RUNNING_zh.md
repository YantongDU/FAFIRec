# 文献驱动的独立残差实验

本轮实现依据见 [文献与方法说明](LITERATURE_GUIDED_DESIGN_zh.md)。所有实验使用已有 ML-1M 时间划分、静态 Qwen 文本向量及验证选出的强 CF；不修改旧实验。它是新的方法组合，不是所引用论文的完整复现。

## 输入与版本

本机新工作树为 `E:/论文/FAFIRec_Research_v3`。运行使用 `E:/论文/FAFIRec_ML1M/.venv/Scripts/python.exe`，Python3.12.14、PyTorch2.6.0+cu126、FP32。源码首次冻结为 `65a2656`；最终配置和独立种子调度工具为 `0ffb546`。

配置默认读取相邻冻结目录 `../FAFIRec_Improve_v2/runs/ml1m_stable_transfer_v2` 与 `../FAFIRec_Improve_v2/runs/cf_lr_validation_v2`。它们包含前轮完整 manifest、teacher/cache 来源记录、CF 验证选参锁及五个被选模型。需要保留其数据、特征 sidecar、映射与原始指纹。输入缺失、旧源文件语义变更或 hash 不符会明确拒绝；不会改用随机文本。上述私有数据和模型不随 Git 分发。在另一台机器运行需先完成已有数据准备/CF 基线流程，并明确配置实际路径。

本机 PowerShell 每次运行前设置：

```powershell
$env:TEMP='E:/论文/FAFIRec_Research_v3/data/local_temp'
$env:TMP=$env:TEMP
$env:PYTHONPATH=''
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
```

## 先验证，再固定五种子

以下命令假定项目目录为当前目录，`python` 为上述环境。

```powershell
python research_ml1m.py --config configs/ml1m_research_pilot.json
python tools/compare_research.py --project-dir . --run-dir runs/literature_pilot_2025 --split validation --output runs/literature_pilot_2025/analysis/compare_validation.json
python tools/replay_research_grid.py --run-dir runs/literature_pilot_2025 --split validation --output-dir runs/literature_pilot_2025/replay_validation --selection-lock runs/literature_pilot_2025/validation_selection_lock.json
```

首轮只有 seed2025，比较 BPR尺度8、CE尺度16和可选身份对齐，不能将目标与尺度共同变化归结为 CE 的单因素作用。它选中了 `text_ce`，不加身份对齐。文本 CE 的最好验证模型位于150轮上限，因此后续三个控制统一提高到最多300轮，保留每5轮验证、最少30轮、连续3次未提高后早停、epoch0可选及同一幅度网格。这是训练预算调整的验证依据，不是利用测试集选参。

```powershell
python tools/run_research_replicas.py --config configs/ml1m_research_selected.json --parallel 2
```

最终配置预先列出2025–2029五种子、ID/真实文本/打乱文本三个专家类型及global/relative两阶段，共30个被选模型。为利用硬件，每个seed拥有独立 `seed_N/config.json`、`seed_N/run` 与日志。除seed和输出路径外所有配置相同。master plan绑定完整配置、源码和文件hash。最多两个子进程并行，本机每个进程4 CPU线程；不改变batch2048、精度或损失。断点恢复必须保持原plan：

```powershell
python tools/run_research_replicas.py --config configs/ml1m_research_selected.json --parallel 2 --resume
```

每轮保存原子 `last.pt`、优化器及随机状态；`best.pt`包含验证所选权重和幅度。专家阶段训练独立分支和gain，冻结 CF。router阶段冻结所有专家，只训练4权重和1截距。每个阶段分别保留结果，不覆盖global模型。

## 逐模型核验与共同选择

对每个seed的run，先执行 `tools/compare_research.py --split validation`，保存到该seed目录的 `compare_validation.json`；再以 `tools/replay_research_grid.py --split validation` 保存独立排名和该run的验证选参锁。前者检查完整网格、训练/校准预算及原强CF对照，独立重算逐用户指标；后者要求排名CSV与训练导出逐字节相同。

五个seed共同选择一个variant/stage，以五seed平均验证NDCG@50为依据；不能给每seed挑不同stage后拼成一行成绩。`tools/combine_research_replicas.py`负责跨run检查和共同锁定。测试应在全部验证核验和共同锁定后显式执行；`replay_research_grid.py --split test --repeat`重新加载两次，并要求排名与指标相同。`compare_research.py --split test --test-dir ...`重新检查目标、历史排除、物品范围、完整用户覆盖及所有指标，随后进行共同五seed统计。

ID分支是强继续训练/集成对照，并非参数量与初始分数严格匹配。真实/打乱文本采用同参数化与训练过程，检验语义和物品身份的对应关系。router比global多一阶段训练，还增加逐用户幅度自由度；其差值代表整套额外阶段的效果，不能单独证明某一个特征的因果贡献。原test已在历史开发中被观察，本轮需如实称为迭代benchmark。
