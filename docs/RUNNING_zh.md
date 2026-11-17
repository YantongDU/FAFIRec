# 按最新版方法运行实验

本版已经把旧的非线性可靠性估计替换为零初始化、可学习正负系数的线性诊断控制；训练入口为根目录 `training.py`。旧代码位于 `legacy/`，不再参与当前实验。

## 先验证完整流程

在项目目录安装 `requirements.txt` 中的依赖后执行：

```bash
python run_tests.py
python prepare_data.py --source prime-pantry --input dataset --output data/prime_pantry --allow-unverified-order
python training.py --config configs/pantry_lexical_smoke.json
python analyze_results.py runs/pantry_lexical_smoke
```

这会使用仓库已有数据、128 个用户、2 个种子和 9 种控制完成快速检查。特征为明确标注的词汇特征，不是 LLM 向量；输出不可用作正式实验结论。Pantry 原文件无时间戳，因此需显式接受原有顺序。

## 再进行正式实验

1. 根据 `DATA_FORMAT.md` 准备 ML-1M、Netflix、Book-Crossing 或其他真实数据。评分阈值、冷用户分组、候选集必须写入最终论文的实验协议。Book-Crossing 没有时间戳，本代码不会把文件顺序描述成可靠时间顺序。
2. 使用真实文本编码器生成物品向量。`generate_text_features.py` 可对准备后的物品目录调用 LLM2Vec，自动写出物品行映射与文件校验清单；需要实际模型/适配器和足够硬件。也可提供其他来源的真实向量及同格式清单。
3. 检查相应配置文件的输入路径、预算和超参数；仅使用验证集选择超参数及检查点，然后运行 `python training.py --config configs/ml1m.json` 等入口。
4. 使用 `python analyze_results.py runs/ml1m` 导出跨种子均值、样本标准差、配对比较及论文表格。必须使用完整匹配的用户、种子与实验配置。

默认采用全部非 padding 物品作为候选集。`exclude_seen_restore_target` 是额外提供的目标条件化离线约定，不能与默认全目录协议混报。

## 中断恢复与独立评估

```bash
python training.py --config configs/ml1m.json --resume
python evaluate.py --run-dir runs/ml1m --seed 2025 --control diagnostic --output-dir runs/ml1m_reloaded --audit-users 128
```

恢复只接受相同源码、数据、配置和记录的运行环境，在完整 epoch 边界恢复模型、优化器、随机状态与早停进度。已经完成的配置核对结果文件和检查点校验值后跳过。源码或输入变化时应创建新输出目录。

独立评估核对模型、特征映射、用户顺序、教师与诊断缓存。排序审计比较同一检查点的协同分支和融合分支，明确区分 top-K 集合保持和内部顺序保持，不把“保持原排序”当成“推荐一定更好”。

## 与旧实验的关系

这是对最新版方法的可运行实现。旧论文表格不自动适用于本版；正式结果应在实际三数据集和真实文本特征上重新运行获得。配置里的训练预算和超参数是起始设置，不是已经验证最优的设置。
