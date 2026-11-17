# ML-1M 迁移初始化与验证实验运行说明

新入口用于检查[原因审计与固定验证方案](ML1M_IMPROVEMENT_DIAGNOSIS_zh.md)中的初始化和优化问题。旧五种子实测结果继续保留在[历史结果记录](ML1M_ACTUAL_RESULTS_zh.md)，不能将新候选的验证结果写成已完成的性能提升或SOTA结论。

最终融合、对齐与推理公式未改变；新增的是训练内投影初始化、可训练门的起点、参数分组学习率和epoch0选模。初始化与训练预算是必须报告的实验设置。

## 准备参考运行

公开配置为 [`configs/ml1m_transfer_validation.json`](../configs/ml1m_transfer_validation.json)。其中 `reference_run_dir` 必须指向一个已完成的version2 ML-1M多目标运行目录，包含其清单声明的完整种子/对照网格、共享教师、诊断缓存和选中检查点。该目录的 `manifest.json` 所引用的数据与文本特征也必须仍可访问。

运行前复制配置到本地路径，再修改 `reference_run_dir` 和新的 `output_dir`，保留其余候选设置。相对路径按项目根目录解析；输出目录不能与参考目录重叠。程序检查数据、映射、特征、检查点、缓存与运行身份，不应通过手改历史清单绕过校验。

本机已有配置 `data/ml1m_transfer_validation_local.json`，其参考目录为：

```text
E:/论文/FAFIRec_ML1M/runs/ml1m_paper_qwen3_v1
```

新代码目录为 `E:/论文/FAFIRec_Improve`。本机可复用 `E:/论文/FAFIRec_ML1M/.venv/Scripts/python.exe` 这一独立环境；以下命令中的 `python` 均指已安装项目依赖的同一环境。不要覆盖旧源码或旧运行输出。

## 仅用验证集比较四个候选

在项目根目录执行公开配置：

```bash
python validate_ml1m.py --config configs/ml1m_transfer_validation.json
python validate_ml1m.py --config configs/ml1m_transfer_validation.json --resume
```

本机使用已指向旧目录的配置：

```powershell
& 'E:/论文/FAFIRec_ML1M/.venv/Scripts/python.exe' validate_ml1m.py --config data/ml1m_transfer_validation_local.json
& 'E:/论文/FAFIRec_ML1M/.venv/Scripts/python.exe' validate_ml1m.py --config data/ml1m_transfer_validation_local.json --resume
```

这里两个命令分别表示首次启动与中断后继续，按实际情况选择一个；不要并发启动同一输出目录。精确继续要求源码、配置、运行环境、数据和参考运行均保持一致。已完成任务将校验并复用；改变设置须使用新的输出目录。

当前pilot固定seed2025，四个候选均使用原30轮共享教师及其固定诊断，以隔离初始化和优化因素：

| variant名称 | 投影初始化 | CF门先验 | 对齐系数 | 门/控制器学习率倍率 |
|---|---|---:|---:|---:|
| `prior_only` | random | 0.9 | 0.3 | 1 |
| `ridge_prior` | ridge | 0.9 | 0.3 | 1 |
| `ridge_light_alignment` | ridge | 0.9 | 0.03 | 1 |
| `ridge_fast_gates` | ridge | 0.9 | 0.03 | 10 |

ridge正则系数固定为1，与表中的对齐系数是不同参数。学生最多150轮，验证NDCG@50选模，耐心10轮，包含epoch0。训练入口不调用测试评估；训练/验证记录中移除test目标。旧完成记录的测试字段仅随参考文件被完整性绑定，不参与候选选择。

默认输出位于 `runs/ml1m_transfer_pilot_v1`：

- `selection.json`：候选的验证汇总、按验证NDCG@50选出的名称以及原对照的验证表现。
- `summary.json`、各 `seed_*/<variant>_summary.json`：初始验证值、选中轮数、初始化记录及验证指标。
- 各 `seed_*/<variant>_validation.csv`：完整验证目标及逐用户推荐列表。
- 各候选的 `best.pt`、`last.pt`、训练日志和运行清单。

单种子pilot用于初筛，不能替代跨种子复核；完成并锁定候选后，才能进行明确的测试评估。

## 独立重载候选

验证pilot的检查点使用 `evaluate_transfer.py`，其身份参数为 `--variant`，不能与普通ratio运行的 `--control` 混用。输出目录必须为空或尚不存在。

例如重载一个已完成候选的验证输出：

```bash
python evaluate_transfer.py --run-dir runs/ml1m_transfer_pilot_v1 --seed 2025 --variant prior_only --split validation --output-dir runs/transfer_replay_validation_2025
```

测试重载必须显式指定 `--split test`。以下PowerShell示例从已锁定的验证选择中读取名称，避免人工挑选测试结果：

```powershell
$selectedVariant = (Get-Content runs/ml1m_transfer_pilot_v1/selection.json -Raw | ConvertFrom-Json).selected_variant
python evaluate_transfer.py --run-dir runs/ml1m_transfer_pilot_v1 --seed 2025 --variant $selectedVariant --split test --output-dir runs/transfer_replay_test_2025
```

测试命令只在验证选择和预定复核完成后执行。不能对候选逐个查看测试结果再改选。程序仍校验源代码、参考输入、教师、缓存与检查点身份；测试使用train+完整validation作为观察历史，保留全部test目标，候选与指标约定不变。

## 通用训练入口的新增设置

`train_ml1m.py` 的JSON配置支持以下字段。缺省数值保持原训练行为，但源码身份已改变；旧运行仍应由保留的旧目录程序重载，新实验应写入新目录。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `projection_init` | `"random"` | `random`使用随机投影；`ridge`使用训练物品与固定教师拟合初始投影 |
| `cf_prior` | `0.5` | 取值严格在(0,1)内；新候选为0.9，两侧融合门仍可训练 |
| `ridge` | `1.0` | 必须为正，定义无截距ridge初始化的正则强度 |
| `gate_lr_multiplier` | `1.0` | 正数；本轮比较1和10，仅调整门与控制器参数组的学习率 |
| `include_initial_checkpoint` | `false` | 布尔值；为true时把epoch0加入验证选模；验证pilot固定开启 |

ridge仅读取训练正反馈序列中出现过的唯一物品、这些物品的静态文本及选中教师ID向量。求解后投影仍可训练，没有增加独立推理编码器或把测试交互送入初始化。

这里的“训练已见”专指在正反馈训练序列中出现；未出现的物品仍可能作为负样本被更新。排除它们是为了不把缺少正反馈锚定的物品作为回归目标，不能据此断言它们从未获得任何CF梯度或信号。

普通新配置仍通过以下入口运行：

```bash
python train_ml1m.py --config path/to/new_config.json
python train_ml1m.py --config path/to/new_config.json --resume
```

通用入口支持将 `teacher_epochs` 设为150，并按既定验证指标和早停规则选出教师，再构建固定训练诊断。修改教师后必须重新生成对应诊断，不能复用不匹配的旧缓存。

## 成熟教师阶段与报告边界

本轮四候选pilot仍复用30轮教师，**尚未修复教师预算不足**。计划中的 `configs/ml1m_stable_transfer.json` 将在pilot验证选择锁定后创建：教师上限150轮，CF、uniform与diagnostic使用同一个选中教师、对应固定诊断及可比较的学生训练/选择规则。该文件与实际运行状态以后续锁定配置为准。

初始化、门控学习率、对齐系数、epoch0选模及教师预算均应逐项记录。成熟教师阶段同时改变多个设置，其整体差异不能归因于某一个组件；需要固定教师的候选比较和匹配对照共同解释。未完成前不声明改进、显著优势或新的SOTA。

原始数据、LLM权重、文本特征、诊断缓存、检查点及逐用户运行输出均保留在本地被忽略的 `data/`、`runs/` 中，不提交Git。可提交的配置和汇总文档不应混入这些输入或缓存。
