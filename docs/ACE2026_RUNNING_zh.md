# ACE 的本地运行与核验

方法来源和统一评估的适配见 [基线说明](ACE2026_BASELINE_zh.md)，原论文其他模型的数值见 [文献参考表](ACE_PAPER_BASELINE_REFERENCE_zh.md)。本流程以已有 ML-1M 数据及静态文本向量为输入。

## 运行环境和输入

本机项目为 `E:/论文/FAFIRec_ACE2026`，使用 `E:/论文/FAFIRec_ML1M/.venv/Scripts/python.exe`，Python 3.12.14、PyTorch 2.6.0+cu126、FP32。下面的 `python` 指该环境。

`master_config` 指向前轮 `FAFIRec_Research_v3/configs/ml1m_research_selected.json`，由其读取冻结数据、Qwen3-Embedding-0.6B向量、映射及来源记录。共享输入校验会读取旧CF模型等来源文件；ACE模型从静态文本谱初始化和随机编码器开始训练，原来的CF权重不参与ACE模型初始化或预测。私有数据、文本向量和旧模型不随Git分发，换机器前须按 [既有输入准备流程](RESEARCH_RUNNING_zh.md) 准备实际文件，再设置相应路径。

本机每次运行前设置：

```powershell
$env:TEMP='E:/论文/FAFIRec_ACE2026/data/local_temp'
$env:TMP=$env:TEMP
$env:PYTHONPATH=''
$env:PYTHONIOENCODING='utf-8'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
```

配置中的 `svd_cache` 可设为 `null`，由每次调用根据实际文本计算SVD；为了共享同一组分解，本轮预先保存 `data/ace2026_qwen_svd.pt`。需要新建缓存时，在项目根目录运行：

```python
from pathlib import Path
from tools import ace2026_experiment as ace
from baselines.ace2026_model import build_ace_svd
from fafirec.runner import torch_dump

cfg = ace.load_config('configs/ace2026_pilot.json')
master = ace.master_config(cfg['master_config'])
_, features, *_ = ace.reference.load_anchors({**master, 'seeds': [2025]})
cache = Path(cfg['svd_cache'])
if cache.exists():
    raise FileExistsError(cache)
torch_dump(cache, build_ace_svd(features))
```

缓存的文本和分解指纹会在构造模型时校验。不得拿其他文本向量的缓存替换。正式运行后保持模型、训练入口及执行工具源码不变，使用新目录开展后续实验。

## 只用验证集选择配置

预先固定九组 `lambda={0,50,500}` × `scale={0.1,0.5,1}`，学习率0.0005、seed2025。选出几何参数后，仅比较0.001和0.0001，与原0.0005共同选择。所有配置最多400轮、每5轮验证、至少30轮、连续3次验证无提高后早停。

以下示例用于还没有这些输出的新实验；已存在的输出保持不可覆盖。仓库保存了本轮选出的配置，因此新实验的选择工具使用 `data/*_reproduced.json` 输出路径。

```powershell
python tools/ace2026_experiment.py train --config configs/ace2026_pilot.json
python tools/select_ace2026_config.py refine --pilot runs/ace2026_pilot --output-config data/ace2026_lr_reproduced.json --output-record runs/ace2026_tuning/pilot_selection.json
python tools/ace2026_experiment.py train --config data/ace2026_lr_reproduced.json
python tools/select_ace2026_config.py final --pilot runs/ace2026_pilot --refinement runs/ace2026_lr_validation --output-config data/ace2026_selected_reproduced.json --output-record runs/ace2026_tuning/final_selection.json
```

选择工具核验完整候选网格、checkpoint及日志，逐用户CSV重算验证指标，并记录所有候选的文件SHA。不会评分测试集。此前FAFIRec测试集已在历史开发中被观察，本轮仍应称作迭代benchmark。

## 五个独立种子与排名重放

选择一个共同配方后，分别训练2025–2029，最多同时运行两个进程：

```powershell
python tools/run_ace2026_replicas.py --config data/ace2026_selected_reproduced.json --parallel 2
python tools/finalize_ace2026_aggregate.py --root runs/ace2026_final_replicas
```

本轮实际运行使用保存的 `configs/ace2026_selected.json`。每个seed拥有独立配置、日志、manifest和模型。调度计划固定输入配置和源码；每轮保存优化器、随机状态和原子checkpoint。恢复同一次训练时添加 `--resume`，保持配置和并行数不变。

核验工具先完成五个模型的验证选择锁，再在独立Python进程中重放验证集，要求CSV与训练时逐字节相同；测试集在两个全新进程中分别加载和排名，两次CSV及指标文件必须相同。随后从CSV重算指标并汇总五seed均值和样本SD。完整核验后生成 `combined_validation.json`、`combined_test.json`、两个 `*_verification.json` 和 `aggregate_finalization_audit.json`。恢复核验时添加 `--resume`；已完成步骤只有在所有文件和指标再次校验通过后才会复用。若希望只核验验证集，可先用冻结的 `verify_ace2026_replicas.py --through validation`，随后由上述完整入口加 `--resume` 完成。

首轮汇总核验暴露了分组排列顺序的差异：逐用户重算结果与JSON中的group键顺序不同，按列表下标比较产生误报。稳定入口以 `(variant, stage, group, metric)` 对齐24行，拒绝缺行或重复，完整核对所有字段；原核验源码、计划、模型和排名文件保留。它同时记录自身及原入口的SHA，修复没有改变任何实验分数。

## 与冻结 FAFIRec 结果比较

```powershell
python tools/compare_ace2026_results.py --ace-root runs/ace2026_final_replicas --fafirec-root E:/论文/FAFIRec_Research_v3/runs/literature_final_replicas --output-json data/ace2026_comparison_reproduced.json --output-md data/ace2026_comparison_reproduced.md --repo-md docs/ACE2026_REPRODUCED_RESULTS_zh.md --report-date 2026-11-20
```

报告工具检查两者数据/特征和评估协议一致、五seed配对、模型选择和调参记录完整，才生成结果。FAFIRec固定为前轮共同验证选出的文本残差改造版本，不按ACE测试结果重新选择。报告保留架构、损失、样本组织及调参策略的差异；优化步数按实际批量解释，不能当成严格匹配计算量的比较。

原论文Table 1中Beauty、Toys、Yelp、ML-20M的其他baseline只列作文献参考；它们不参与本机ML-1M实测的统计。
