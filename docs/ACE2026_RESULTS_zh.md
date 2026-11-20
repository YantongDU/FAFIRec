# ACE（SIGIR 2026）与当前 FAFIRec 的比较

报告日期：2026-11-20（Asia/Shanghai）。

本表为同一ML-1M输入和评估协议下的实际五种子实验；FAFIRec指前轮完成的文本残差改造版本。原稿Table II的历史数值不参与此处统计，ACE原论文其他数据集的数值另列为文献参考。

FAFIRec固定为原共同验证选择 `text_ce/router`；ACE固定为验证选出的同一谱参数、尺度和学习率，各seed只按验证选择epoch。没有根据测试改选模型。

## 全部用户

| 模型 | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |
|---|---:|---:|---:|---:|
| 当前FAFIRec | 0.162811 ± 0.002932 | 0.161769 ± 0.002416 | 0.396403 ± 0.004731 | 0.241545 ± 0.002847 |
| ACE-SASRec（统一协议适配） | 0.169937 ± 0.000925 | 0.169444 ± 0.000475 | 0.412928 ± 0.001877 | 0.252323 ± 0.000630 |

## 冷用户

| 模型 | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |
|---|---:|---:|---:|---:|
| 当前FAFIRec | 0.294958 ± 0.003576 | 0.201479 ± 0.002745 | 0.572045 ± 0.008578 | 0.280625 ± 0.004030 |
| ACE-SASRec（统一协议适配） | 0.299216 ± 0.005933 | 0.207720 ± 0.002110 | 0.601569 ± 0.004237 | 0.294268 ± 0.000430 |

## 暖用户

| 模型 | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |
|---|---:|---:|---:|---:|
| 当前FAFIRec | 0.047696 ± 0.000710 | 0.168293 ± 0.001597 | 0.163959 ± 0.001119 | 0.163238 ± 0.000779 |
| ACE-SASRec（统一协议适配） | 0.049501 ± 0.000981 | 0.171457 ± 0.002848 | 0.166250 ± 0.001260 | 0.165004 ± 0.001379 |

以预先指定的整体NDCG@50看，ACE较高；FAFIRec减ACE的均值差为 -0.010778。其他指标和分组分别见表，不能将一个指标扩展为全面胜出。

## 配对比较

| 用户组 | 指标 | FAFIRec − ACE | seed-bootstrap 95% CI | 双侧exact p | Holm p |
|---|---|---:|---|---:|---:|
| all | recall10 | -0.007125 | [-0.008822, -0.005123] | 0.0625 | 1.0000 |
| all | ndcg10 | -0.007675 | [-0.009280, -0.005977] | 0.0625 | 1.0000 |
| all | recall50 | -0.016525 | [-0.019285, -0.014153] | 0.0625 | 1.0000 |
| all | ndcg50 | -0.010778 | [-0.012638, -0.008917] | 0.0625 | 1.0000 |
| cold | recall10 | -0.004258 | [-0.009356, +0.001232] | 0.1875 | 1.0000 |
| cold | ndcg10 | -0.006241 | [-0.009196, -0.003328] | 0.0625 | 1.0000 |
| cold | recall50 | -0.029524 | [-0.034174, -0.025154] | 0.0625 | 1.0000 |
| cold | ndcg50 | -0.013643 | [-0.016476, -0.010811] | 0.0625 | 1.0000 |
| middle | recall10 | -0.008149 | [-0.010039, -0.006209] | 0.0625 | 1.0000 |
| middle | ndcg10 | -0.008418 | [-0.010415, -0.006421] | 0.0625 | 1.0000 |
| middle | recall50 | -0.016679 | [-0.019418, -0.014189] | 0.0625 | 1.0000 |
| middle | ndcg50 | -0.011546 | [-0.013637, -0.009456] | 0.0625 | 1.0000 |
| warm | recall10 | -0.001804 | [-0.002439, -0.001292] | 0.0625 | 1.0000 |
| warm | ndcg10 | -0.003164 | [-0.005294, -0.000481] | 0.1250 | 1.0000 |
| warm | recall50 | -0.002291 | [-0.003908, -0.001021] | 0.0625 | 1.0000 |
| warm | ndcg50 | -0.001766 | [-0.003006, -0.000297] | 0.1250 | 1.0000 |

区间仅重采样五个训练seed，条件于同一固定评估用户；种子少，区间为描述性结果。双侧精确符号翻转只有32种符号，最小非零p为0.0625；Holm覆盖上述全部16项。没有把重复用户当成额外独立样本。

## 方法与比较边界

共同使用5950用户、3883物品、同一Qwen3-Embedding-0.6B静态元数据向量、逐用户时间80/10/10划分、全目录排除观察历史、多正例用户宏平均指标。ACE的谱分解只使用静态内容，不使用验证或测试反馈。

ACE保留未中心化SVD谱初始化、whole-matrix标准差缩放、可训练物品表、raw-dot SASRec和原生全类CE。它使用128维/2层/批量256，从头训练；FAFIRec使用64维/1层/批量2048及CF预训练、不同损失与残差结构。因此是同输入/同评估比较，不能称严格容量、损失或计算量匹配的单因素实验。训练布局为同一最后50个shifted位置，和作者原生one-prefix加载方式不同；不能称原论文分数复现。

ACE先在seed2025上比较九组谱参数/尺度，再比较两组学习率，锁定配方后做五seed复验；FAFIRec沿用原先五seed共同验证的variant/stage选择。二者调参策略和搜索预算也不完全匹配。

实际训练记录（ACE每个模型768,768个可训练参数）：

| seed | ACE选中轮次 | ACE实际训练轮次 | ACE优化步数 | FAFIRec expert+router步数 |
|---|---:|---:|---:|---:|
| 2025 | 135 | 150 | 3600 | 540 |
| 2026 | 95 | 110 | 2640 | 435 |
| 2027 | 100 | 115 | 2760 | 420 |
| 2028 | 130 | 145 | 3480 | 480 |
| 2029 | 115 | 130 | 3120 | 750 |

步数对应不同批量和损失，不可直接视为计算量比。FAFIRec此处未计入CF预训练、文本编码或调参；ACE调用耗时保留在JSON，若中途恢复则只代表该次调用，不作为累计耗时。没有据此宣称效率优势。

ACE实际共同配置：

```json
{
  "master_config": "E:\\论文\\FAFIRec_Research_v3\\configs\\ml1m_research_selected.json",
  "variants": [
    {
      "reg_weight": 500.0,
      "scale": 0.1,
      "lr": 0.0005,
      "name": "ace_selected"
    }
  ],
  "epochs": 400,
  "min_epochs": 30,
  "eval_interval": 5,
  "patience": 3,
  "batch_size": 256,
  "eval_batch": 128,
  "max_len": 50,
  "weight_decay": 0.0,
  "grad_clip": null,
  "device": "cuda",
  "threads": 4,
  "d_model": 128,
  "n_layers": 2,
  "n_heads": 2,
  "inner_size": 256,
  "dropout": 0.5,
  "attn_dropout": 0.3,
  "layer_norm_eps": 1e-12,
  "initializer_range": 0.02,
  "ce_chunk_size": 1024,
  "svd_cache": "E:\\论文\\FAFIRec_ACE2026\\data\\ace2026_qwen_svd.pt"
}
```

## ACE原论文其他baseline：仅作文献参考

下表为作者原论文Table1的ML-20M结果，平均4次运行，leave-one-out；不是上方ML-1M实测，不与本机五个seed合并统计。原文未给出的标准差或@50数值没有补造。

| 模型（原论文ML-20M） | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| SASRec | 0.1893 | 0.1018 | 0.2643 | 0.1207 |
| LLM2X | 0.1916 | 0.1053 | 0.2660 | 0.1240 |
| WhitenRec+ | 0.1973 | 0.1095 | 0.2765 | 0.1294 |
| LLMEmb | 0.1973 | 0.1087 | 0.2767 | 0.1287 |
| AlphaRec | 0.1992 | 0.1096 | 0.2810 | 0.1302 |
| AlphaFuse | 0.2003 | 0.1092 | 0.2798 | 0.1292 |
| ACE | 0.2032 | 0.1110 | 0.2825 | 0.1310 |

来源：[ACE论文Table1](https://arxiv.org/html/2605.29322v1#S3.T1)，[官方代码](https://github.com/DCheol/ACE)，固定commit `3e099052d9759b841a5192a2ae20a3b3ac02c6c6`。四数据集完整文献参考值见 [参考表](https://github.com/YantongDU/FAFIRec/blob/codex/ace2026-baseline/docs/ACE_PAPER_BASELINE_REFERENCE_zh.md)。

原测试分数在历史开发中已被观察，本轮是透明迭代benchmark，不是未触碰的首次确认集；不据此宣称SOTA或Strong Accept。

## 审计文件

- [combined_validation.json](<E:\论文\FAFIRec_ACE2026\runs\ace2026_final_replicas\combined_validation.json>): `6aa89b4017c72c872e4983ff86892491983f00a324999a770d8b3badb4da9418`
- [combined_test.json](<E:\论文\FAFIRec_ACE2026\runs\ace2026_final_replicas\combined_test.json>): `ae24a2ef36ad706c299539064784e67b3e6ce22469fe252d3febc5e8c186c26a`
- [five_seed_selection_lock.json](<E:\论文\FAFIRec_ACE2026\runs\ace2026_final_replicas\five_seed_selection_lock.json>): `e94039c3922e842cc131ba5b63fe2c9311f99bb561305fdff2cbb781981495a1`
- [combined_validation.json](<E:\论文\FAFIRec_Research_v3\runs\literature_final_replicas\combined_validation.json>): `0f3f457431918c62055a2f2b18f45d36bfc09e06f54247ad477edb6451069f21`
- [combined_test.json](<E:\论文\FAFIRec_Research_v3\runs\literature_final_replicas\combined_test.json>): `36671549f779a106e1a94432c6906486946bc69d1171aaf1c78021ba104291b0`
- [validation_verification.json](<E:\论文\FAFIRec_ACE2026\runs\ace2026_final_replicas\validation_verification.json>): `790e0662812008b909db09ccc158b593472e8cf98ce3c548a20e12ef68656d0e`
- [test_verification.json](<E:\论文\FAFIRec_ACE2026\runs\ace2026_final_replicas\test_verification.json>): `8ee8022e586763df8f47b40dac0c505495ff40bc580c9add8a0d8c9dcacc2c67`
- [final_selection.json](<E:\论文\FAFIRec_ACE2026\runs\ace2026_tuning\final_selection.json>): `1782ddb794e230bc13a8d0ef780210ed1b2209a5a15c6e782e95400768b94ad4`
- [ACE_PAPER_BASELINE_REFERENCE.csv](<E:\论文\FAFIRec_ACE2026\docs\ACE_PAPER_BASELINE_REFERENCE.csv>): `73d26fdccb352c84bf7e4982900a3aabc0b101376345fbab8e8e797d5d89070a`
- [compare_ace2026_results.py](<E:\论文\FAFIRec_ACE2026\tools\compare_ace2026_results.py>): `5e745aaa7062349782b34782b7002041675faab6fd75776ab90cf344e2f0ead9`

## 可复查结果与论文表格

五seed逐项分数、均值/样本SD、11组验证候选和训练记录见 [结果JSON](https://github.com/YantongDU/FAFIRec/blob/codex/ace2026-baseline/docs/ACE2026_RESULTS.json)；[Overleaf表格](https://github.com/YantongDU/FAFIRec/blob/codex/ace2026-baseline/docs/ACE2026_TABLE.tex)配有[ACE文献条目](https://github.com/YantongDU/FAFIRec/blob/codex/ace2026-baseline/docs/ACE2026_REFERENCE.bib)。seed2025从头重训后的所有模型张量与选参时完全相同。

汇总核验使用分组/指标键完整对齐24行，排除列表排列顺序引起的误报；冻结源码、模型、逐用户排名均保留。额外核验文件SHA256：`c5e2521afd8b453b66cee864d97898918efd215370b900ccee717b2c824a144d`。
