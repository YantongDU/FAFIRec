# ML-1M 迁移初始化与验证实验运行说明

本说明对应新代码树 `E:/论文/FAFIRec_Improve_v2`，用于检查[原因审计与验证记录](ML1M_IMPROVEMENT_DIAGNOSIS_zh.md)中的初始化和优化问题。原五种子实测结果继续保留在[历史结果记录](ML1M_ACTUAL_RESULTS_zh.md)；旧改进树 `E:/论文/FAFIRec_Improve` 的源码及输出冻结，只读留存，不能用v2对它继续训练或覆盖结果。

最终融合、对齐与推理公式未改变；新增的是训练内投影初始化、可训练门的起点、参数分组学习率、epoch0选模，以及v2的最短训练窗口和教师/学生独立学习率。初始化、选模与训练预算均须报告。目前仅有单种子验证证据，不能写成诊断独立增益、已完成的性能提升或新SOTA。

## 准备参考运行

公开配置为 [`configs/ml1m_transfer_validation.json`](../configs/ml1m_transfer_validation.json)。其中 `reference_run_dir` 必须指向一个已完成的version2 ML-1M多目标运行目录，包含其清单声明的完整种子/对照网格、共享教师、诊断缓存和选中检查点。该目录的 `manifest.json` 所引用的数据与文本特征也必须仍可访问。

默认 `allow_partial_reference=false`，仍严格要求完整网格。v2仅在显式设置 `allow_partial_reference=true` 时允许使用局部参考；此时每个请求的seed都必须已经完整完成CF、uniform、diagnostic三组，只有教师完成并不够。程序仍绑定整个 `summary.json` 的哈希、请求种子范围、教师、诊断与全部相关检查点身份，不会绕过数据或完整性检查。即使其他未请求种子继续写入总summary，也会改变参考身份，因此开始前应冻结参考运行。

运行前复制配置到本地路径，再修改 `reference_run_dir` 和新的 `output_dir`，保留其余候选设置。相对路径按项目根目录解析；输出目录不能与参考目录重叠。程序检查数据、映射、特征、检查点、缓存与运行身份，不应通过手改历史清单绕过校验。

原30轮教师四候选的本机配置和完整结果保留在旧v1目录。v2当前准备使用 `data/strong_teacher_validation_local.json`，其参考目录为：

```text
E:/论文/FAFIRec_Improve/runs/ml1m_stable_transfer_v1
```

该参考只有seed2025具备三对照完整结果，因此v2配置显式开启局部参考、只请求seed2025。不能将它视为完整五种子实验。本机可复用 `E:/论文/FAFIRec_ML1M/.venv/Scripts/python.exe` 这一独立环境；以下命令中的 `python` 均指已安装项目依赖的同一环境。

## 验证入口与已完成的四候选pilot

在项目根目录执行公开配置：

```bash
python validate_ml1m.py --config configs/ml1m_transfer_validation.json
python validate_ml1m.py --config configs/ml1m_transfer_validation.json --resume
```

以上公开配置描述原四候选方案；它们已在v1完成，不必为读取历史结果重新运行。v2本机的成熟教师适应窗口实验使用：

```powershell
& 'E:/论文/FAFIRec_ML1M/.venv/Scripts/python.exe' validate_ml1m.py --config data/strong_teacher_validation_local.json
& 'E:/论文/FAFIRec_ML1M/.venv/Scripts/python.exe' validate_ml1m.py --config data/strong_teacher_validation_local.json --resume
```

这里两个命令分别表示首次启动与中断后继续，按实际情况选择一个；不要并发启动同一输出目录。精确继续要求源码、配置、运行环境、数据和参考运行均保持一致。已完成任务将校验并复用；改变设置须使用新的输出目录。

已完成的四候选pilot固定seed2025，均使用原30轮共享教师及其固定诊断，以隔离初始化和优化因素。下列数值来自旧v1的 `runs/ml1m_transfer_pilot_v1/selection.json`：

| variant名称 | 投影初始化 | CF门先验 | 对齐系数 | 门学习率倍率 | 全体验证NDCG@50 |
|---|---|---:|---:|---:|---:|
| `prior_only` | random | 0.9 | 0.3 | 1 | 0.22824934952625314 |
| `ridge_prior` | ridge | 0.9 | 0.3 | 1 | 0.22580992226145127 |
| `ridge_light_alignment` | ridge | 0.9 | 0.03 | 1 | **0.22922126688101443** |
| `ridge_fast_gates` | ridge | 0.9 | 0.03 | 10 | 0.22178844246910676 |

按预定验证规则选中 `ridge_light_alignment`，即ridge、CF先验0.9、对齐系数0.03、门学习率倍率1；ridge正则系数固定为1。学生最多150轮、耐心10轮、包含epoch0，原pilot没有最短训练窗口。该单种子选择不证明ridge单独有效，也不证明诊断优于uniform。验证入口不调用测试评估；训练/验证记录中移除test目标。旧完成记录的测试字段仅随参考文件被完整性绑定，不参与候选选择。

原四候选输出保留在旧v1的 `runs/ml1m_transfer_pilot_v1`。v2本机配置的新输出为 `runs/strong_teacher_adaptation_pilot_v1`，记录结构相同：

- `selection.json`：候选的验证汇总、按验证NDCG@50选出的名称以及原对照的验证表现。
- `summary.json`、各 `seed_*/<variant>_summary.json`：初始验证值、选中轮数、初始化记录及验证指标。
- 各 `seed_*/<variant>_validation.csv`：完整验证目标及逐用户推荐列表。
- 各候选的 `best.pt`、`last.pt`、训练日志和运行清单。

单种子pilot用于初筛，不能替代跨种子复核；完成并锁定候选后，才能进行明确的测试评估。

## 为何增加最短适应窗口

四候选选定后，v1曾启动150轮上限教师的新运行。只有seed2025完成三对照，其验证选模元数据为：

| 对照 | 选中学生epoch | 实际学生训练轮数 | 选中验证NDCG@50 |
|---|---:|---:|---:|
| CF | 55 | 65 | 0.23217186130194542 |
| uniform | 0 | 10 | 0.22345551823110546 |
| diagnostic | 0 | 10 | 0.22345551823110546 |

uniform和diagnostic均在连续10轮未超越初始化后早停，最终选中了相同的epoch0。这个现象支持检查“成熟教师转入多模态后，需要更长适应窗口或更小学生学习率”的假设，尚不证明该假设成立。该run不继续充当最终五种子结果；旧输出与源码留存，不在这里查看或解释它的新测试指标。

v2准备基于这个已经完成的seed2025成熟教师，固定ridge、CF先验0.9、对齐系数0.03和门倍率1，比较三个候选：

| variant名称 | 学生学习率 | min_epochs |
|---|---:|---:|
| `warm30_lr1e3` | 0.001 | 30 |
| `warm30_lr3e4` | 0.0003 | 30 |
| `warm30_lr1e4` | 0.0001 | 30 |

三个候选仅按验证NDCG@50选模；当前尚未锁定最终学生学习率。教师及诊断在该pilot中固定复用，不随候选重新训练。

## 独立重载候选

验证pilot的检查点使用 `evaluate_transfer.py`，其身份参数为 `--variant`，不能与普通ratio运行的 `--control` 混用。必须使用该运行记录的冻结源码版本；v1的四候选检查点仍由旧v1程序重载，不能直接用v2替代。输出目录必须为空或尚不存在。

例如重载一个已完成候选的验证输出：

```bash
python evaluate_transfer.py --run-dir runs/strong_teacher_adaptation_pilot_v1 --seed 2025 --variant warm30_lr1e3 --split validation --output-dir runs/transfer_replay_validation_2025
```

测试重载必须显式指定 `--split test`。以下PowerShell示例从已锁定的验证选择中读取名称，避免人工挑选测试结果：

```powershell
$selectedVariant = (Get-Content runs/strong_teacher_adaptation_pilot_v1/selection.json -Raw | ConvertFrom-Json).selected_variant
python evaluate_transfer.py --run-dir runs/strong_teacher_adaptation_pilot_v1 --seed 2025 --variant $selectedVariant --split test --output-dir runs/transfer_replay_test_2025
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
| `min_epochs` | `0` | 非负整数且不超过学生epoch预算；规定至少训练多少轮才允许早停 |
| `teacher_lr` | `null` | 默认继承`lr`；可独立固定教师为0.001，再比较较小学生`lr` |

`min_epochs=30`期间仍持续保存全局最优检查点，epoch0依然有资格获选；stale计数照常累积，并不在第30轮重置patience。如果前30轮一直未提高且patience为10，第30轮就可以停止，不必再额外等待10轮。它保证适应时间，不保证模型必须采用某个更新后的检查点。

ridge仅读取训练正反馈序列中出现过的唯一物品、这些物品的静态文本及选中教师ID向量。求解后投影仍可训练，没有增加独立推理编码器或把测试交互送入初始化。

这里的“训练已见”专指在正反馈训练序列中出现；未出现的物品仍可能作为负样本被更新。排除它们是为了不把缺少正反馈锚定的物品作为回归目标，不能据此断言它们从未获得任何CF梯度或信号。

普通新配置仍通过以下入口运行：

```bash
python train_ml1m.py --config path/to/new_config.json
python train_ml1m.py --config path/to/new_config.json --resume
```

通用入口支持将 `teacher_epochs` 设为150，并按既定验证指标和早停规则选出教师，再构建固定训练诊断。修改教师后必须重新生成对应诊断，不能复用不匹配的旧缓存。

## 最终配置与报告边界

原四候选pilot复用30轮教师；v1的成熟教师运行未构成完整五种子对照。v2适应窗口pilot已按验证NDCG@50选出学生`lr=0.0003`、`min_epochs=30`，对应分数0.23146774133484；其初始值为0.22345551823110546。最终配置已锁定于 `configs/ml1m_stable_transfer.json`：教师上限150轮、`teacher_lr=0.001`，ridge初始化、CF门起点0.9、对齐系数0.03、门控学习率倍率1。此为单种子验证选择，完整多种子测试结果另行记录。

最终CF、uniform与diagnostic需使用同一个选中教师、对应固定诊断及可比较的学生训练/选择规则。新教师必须重新构建匹配诊断，不能复用另一教师的缓存。配置文件是否已锁定、运行是否完整，均以实际记录为准。

初始化、门控学习率、对齐系数、epoch0选模及教师预算均应逐项记录。成熟教师阶段同时改变多个设置，其整体差异不能归因于某一个组件；需要固定教师的候选比较和匹配对照共同解释。未完成前不声明改进、显著优势或新的SOTA。

原测试结果已知，后续过程是透明的迭代benchmark开发，不能描述成从未观察过测试结果的pristine确认试验。即使后续完整运行有更高均值，也需报告配置选择过程、全部匹配对照与多种子不确定性，不能直接宣称诊断组件的独立增益。

原始数据、LLM权重、文本特征、诊断缓存、检查点及逐用户运行输出均保留在本地被忽略的 `data/`、`runs/` 中，不提交Git。可提交的配置和汇总文档不应混入这些输入或缓存。

## 完整重载与保守CF核对

主五种子网格完成后，先汇总并独立重载全部15个选中检查点：

```bash
python analyze_ml1m.py runs/ml1m_stable_transfer_v2
python analyze_ml1m.py runs/ml1m_stable_transfer_v2 --baseline cf --output-dir runs/ml1m_stable_transfer_v2/analysis_vs_cf
python verify_ml1m_replays.py runs/ml1m_stable_transfer_v2
```

首个种子显示CF在学习率0.001下的验证NDCG@50为0.23217186130194542，高于共同微调设置0.0003下的0.22662161309268164。因此预先安排独立CF学习率核对，完整复用主实验五个教师和诊断，按五种子平均验证NDCG@50比较0.001和0.0003；不能只据共同设置较弱的CF宣称整体优势。

```bash
python validate_ml1m.py --config configs/ml1m_cf_validation.json
python analyze_transfer.py runs/cf_lr_validation_v2 --comparison cf_lr1e3 --baseline reference:cf
python analyze_cf_baseline.py --project-dir .
```

最后一个入口要求主15组重载、两类完整分析、实际主配置及训练前的配置选择锁均存在。本机默认文件为 `data/ml1m_stable_transfer_local.json` 与 `data/final_transfer_selection_lock.json`；其他目录用 `--main-config`、`--config-lock`、`--main-run`、`--search-run` 等参数传入相应记录。锁须真实对应该次实验的验证选择及配置SHA，不能用测试结果事后改变参数。

工具先锁定验证选择，再单独评估和重载所选CF，重新核对真实测试目标、已观察历史排除、用户分组与逐用户指标。输出 `analysis/strong_cf_comparison.json`，同时保留原匹配CF和验证选择的CF；精确平手保留原CF。已有重载目录或测试输出不会被覆盖。
