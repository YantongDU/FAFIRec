"""Report locked five-seed results without selecting anything on test."""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_results import paired_seed_comparison, _holm_adjust
from analyze_transfer import _same
from fafirec.runner import file_digest
from tools import ace2026_experiment as ace
from experiments.diagnostic_v4 import final_runs as fr

GROUPS = ('all', 'cold', 'middle', 'warm')
METRICS = ('recall10', 'ndcg10', 'recall50', 'ndcg50')
LABELS = {'ace_original': 'ACE（原last50训练）', 'ace_coverage': 'ACE（全部转移训练）',
    'popularity_only': '全部转移ACE＋热门校准', 'global_fusion': '全局互补融合',
    'diagnostic_fusion': '验证最优诊断门控', 'uniform_alignment': '全局融合＋均匀关系对齐',
    'diagnostic_matched': '诊断门控（匹配混合系数）', 'diagnostic_alignment': '诊断门控＋关系对齐',
    'global_no_pop_control': '全局融合（有效幅度校准对照）'}


def index(report):
    result = {}
    for r in report['aggregates']:
        values = r['per_seed_scores']
        ace.require(r['seed_ids'] == report['seed_ids'] and len(values) == 5, 'Five-seed metric coverage differs')
        _same(statistics.mean(values), r['mean'], 'Recorded aggregate mean')
        _same(statistics.stdev(values), r['sample_std'], 'Recorded sample SD')
        key = r['variant'], r['group'], r['metric']
        ace.require(key not in result, 'Duplicate aggregate metric')
        result[key] = r
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/diagnostic_v4_final'))
    p.add_argument('--old-ace', type=Path, default=Path('E:/论文/FAFIRec_ACE2026/runs/ace2026_final_replicas'))
    p.add_argument('--report-date', default='2026-11-20')
    p.add_argument('--local-report', type=Path, default=Path('E:/论文/FAFIRec_诊断改进与ACE对比.md'))
    args = p.parse_args()
    root = args.root.resolve()
    from experiments.diagnostic_v4 import effective_scale_control as scale_control
    scale_control.run(root)
    auxiliary_file = root / 'effective_scale_control/combined_test.json'
    auxiliary = ace.read(auxiliary_file)
    for path, checksum in auxiliary['artifact_sha256'].items():
        ace.require(file_digest(path) == checksum, 'Fixed scale control artifact changed: ' + path)
    val, test = [ace.read(root / f'combined_{split}.json') for split in ('validation', 'test')]
    lock_path = root / 'five_seed_selection_lock.json'
    lock = fr.verify_lock(lock_path)
    ace.require(val['selected_recipe'] == test['selected_recipe'] == lock['selected_recipe']
        and val['decision'] == test['decision'] == lock['candidates'], 'Test changed the selected recipe')
    ace.require(val['seed_ids'] == test['seed_ids'] == fr.SEEDS, 'Final seed order differs')
    vr = index(val)
    choices = []
    for row in lock['candidates']:
        metric = vr[row['recipe'], 'all', 'ndcg50']
        _same(metric['per_seed_scores'], row['per_seed_validation_ndcg50'], 'Locked validation per-seed choice')
        _same(metric['mean'], row['mean_validation_ndcg50'], 'Locked validation choice mean')
        choices.append((metric['mean'], row['recipe']))
    ace.require(sorted(choices, key=lambda r: (-r[0], r[1]))[0][1] == lock['selected_recipe'],
        'Primary recipe is not the common validation winner')
    for report in (val, test):
        ace.require(report['all_artifact_hashes_verified'] and report['selection_lock_sha256'] == file_digest(lock_path),
            'Incomplete final artifact audit')
        for path, checksum in report['artifact_sha256'].items():
            ace.require(file_digest(path) == checksum, 'Final artifact changed: ' + path)
    for split in ('validation', 'test'):
        proof = ace.read(root / f'{split}_verification.json')
        recipes = [r['name'] for r in test['common']['recipes']]
        ace.require(proof['complete'] and proof['seed_ids'] == fr.SEEDS
            and len(proof['results']) == len(recipes) * 5
            and {(r['seed'], r['variant']) for r in proof['results']} == {(s, n) for s in fr.SEEDS for n in recipes},
            'Independent replay coverage is incomplete')
        for row in proof['results']:
            for a, b in (('per_user_file', 'per_user_sha256'), ('metrics_file', 'metrics_sha256')):
                ace.require(file_digest(row[a]) == row[b], 'Independent replay changed')
            if split == 'test':
                ace.require(row['independent_replay_identical']
                    and file_digest(row['reloaded_file']) == row['reloaded_sha256'] == row['per_user_sha256']
                    and file_digest(row['reloaded_metrics_file']) == row['reloaded_metrics_sha256'] == row['metrics_sha256'],
                    'Two independent test replays differ')
    old = ace.read(args.old_ace / 'combined_test.json')
    ace.verify_lock(old['selection_lock_file'])
    ace.require(old['seed_ids'] == test['seed_ids'] and old['common']['identity'] == test['common']['identity']
        and old['common']['evaluation_config'] == test['common']['evaluation_config'], 'ACE input/evaluation differs')
    for path, checksum in old['artifact_sha256'].items():
        ace.require(file_digest(path) == checksum, 'Original ACE artifact changed: ' + path)
    rows, oldrows = index(test), index(old)
    auxrows = index(auxiliary)
    for group in GROUPS:
        for metric in METRICS:
            rows['global_no_pop_control', group, metric] = auxrows['global_no_pop_control', group, metric]
    oldname = old['aggregates'][0]['variant']
    for group in GROUPS:
        for metric in METRICS:
            rows['ace_original', group, metric] = oldrows[oldname, group, metric]
    selected = lock['selected_recipe']
    final_plan = ace.read(root / 'master_plan.json')
    components_root = Path(final_plan['components_root'])
    completion_file = components_root / 'completion.json'
    ace.require(file_digest(completion_file) == final_plan['components_completion_sha256'], 'Component completion changed')
    component_completion = ace.read(completion_file)
    for receipt in component_completion['receipts'].values():
        for path, checksum in receipt['artifact_sha256'].items():
            ace.require(file_digest(path) == checksum, 'Component training provenance changed: ' + path)
    costs = []
    for seed in fr.SEEDS:
        base = components_root / f'seed_{seed}'
        cv = ace.read(base / 'coverage/all_windows/summary.json')
        teacher = ace.read(base / 'teacher/summary.json')
        costs.append({'seed': seed, 'coverage_selected_epoch': cv['selected_epoch'],
            'coverage_epochs_completed': cv['epochs_completed'], 'coverage_optimizer_steps': cv['optimizer_steps'],
            'teacher_fixed_epochs': teacher['epochs'], 'teacher_optimizer_steps': teacher['epochs'] * 24,
            'coverage_seconds_this_invocation': cv['elapsed_seconds_this_invocation']})
    comparisons = []
    for control in ('ace_original', 'ace_coverage', 'global_fusion', 'global_no_pop_control'):
        for group in GROUPS:
            for metric in METRICS:
                a, b = rows[selected, group, metric], rows[control, group, metric]
                differences = [x - y for x, y in zip(a['per_seed_scores'], b['per_seed_scores'])]
                comparisons.append({'selected_recipe': selected, 'control': control, 'group': group, 'metric': metric,
                    'relative_percent': 100 * (a['mean'] / b['mean'] - 1.),
                    **paired_seed_comparison(differences, bootstrap_samples=10000, seed=2025)})
    _holm_adjust(comparisons)
    files = [root / 'combined_validation.json', root / 'combined_test.json', lock_path,
        root / 'validation_verification.json', root / 'test_verification.json',
        args.old_ace / 'combined_test.json', ROOT / 'configs/diagnostic_v4_selected.json',
        ROOT / 'configs/diagnostic_v4_scale_control.json', auxiliary_file, completion_file, Path(__file__)]
    artifact_hashes = {str(path.resolve()): file_digest(path) for path in files}
    names = ['ace_original'] + [r['name'] for r in test['common']['recipes']] + ['global_no_pop_control']
    snapshot = {'schema': 'fafirec_v4_public_results_v1', 'report_date': args.report_date,
        'seed_ids': test['seed_ids'], 'selected_recipe': selected, 'validation_decision': lock['candidates'],
        'recipes': test['common']['recipes'], 'identity': test['common']['identity'],
        'evaluation_config': test['common']['evaluation_config'], 'validation_aggregates': val['aggregates'],
        'test_aggregates': test['aggregates'], 'original_ace_test_aggregates': old['aggregates'],
        'fixed_scale_control_test_aggregates': auxiliary['aggregates'],
        'paired_comparisons': comparisons, 'artifact_sha256': artifact_hashes,
        'component_training_costs': costs,
        'historical_test_known': True, 'new_test_used_for_selection': False,
        'interpretation': 'Performance recipe study. Global calibration, training coverage and collaborative complementarity are separate from diagnostic novelty.'}
    lines = ['# FAFIRec改造：五种子与ACE对比', '', f'日期：{args.report_date}（Asia/Shanghai）。', '',
        f'共同验证选择：`{selected}`。以下均为固定ML-1M时间划分、全目录排除已观察历史、多正例用户宏平均下的真实五种子测试结果。', '',
        '本轮改造采用ACE型强序列骨干、独立协同分支及显式校准，并检验诊断控制和可选关系对齐；它是新的实现组合，原稿Table II的历史分数不参与此处统计。', '']
    for group, title in (('all', '全部用户'), ('cold', '冷用户'), ('warm', '暖用户')):
        lines += ['## ' + title, '', '| 模型 | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |', '|---|---:|---:|---:|---:|']
        for name in names:
            label = LABELS.get(name, name) + ('（共同验证选中）' if name == selected else '')
            cells = [f"{rows[name, group, m]['mean']:.6f} ± {rows[name, group, m]['sample_std']:.6f}" for m in METRICS]
            lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## 主要结论与归因', '']
    for control in ('ace_original', 'ace_coverage', 'global_fusion', 'global_no_pop_control'):
        r = next(r for r in comparisons if r['control'] == control and r['group'] == 'all' and r['metric'] == 'ndcg50')
        lines.append(f"- 相对{LABELS[control]}，选中配方整体NDCG@50差 {r['mean_difference']:+.6f}，相对变化 {r['relative_percent']:+.2f}%。")
    lines += ['', '训练覆盖变化由ACE自身的全部转移训练控制检验；热门校准与全局互补融合分别单列。EASE和ACE是已有方法，不能将骨干或简单加分本身作为新颖性证明。', '',
        '诊断门控应与同一混合系数的global比较，还应注意gate可以学到近似常数幅度；选中的配方若为global，诊断特征不参与其预测或对齐权重，不能把总提升归因于诊断。均匀对齐的小差值也不能提前称为稳定机制收益。', '',
        '新协同分支和all-windows序列骨干均可利用全部训练反馈，但架构、容量、计算方式不同。关系对齐对应显式物品身份下的score-profile目标，不是原稿MK-MMD；原分布对齐理论不应沿用。该目标同时改变重构与有效正则，现有差值不能单独证明语义先验的因果贡献。', '',
        '内层代理只在训练前段拟合，固定135轮，不用校准尾部或外层验证/测试早停。gate用内部尾部的多正例块训练，诊断只用已观察历史；外层测试允许train+validation作为已观察前缀，训练频次和关系算子保持冻结。内层代理是last50模型，不能称为全转移主骨干自身的已校准可靠性。', '',
        '## 参数与训练成本', '',
        f"序列骨干为128维、2层、768,768个可训练参数；协同算子完整存储 {test['common']['identity']['items'] ** 2:,} 个数值，采用零对角约束和闭式拟合，不能将其与序列骨干的参数数目混为严格容量匹配。诊断代理作为额外模型训练，其成本也应披露。", '',
        '| seed | 骨干选中轮次 | 骨干实际轮次 | 骨干优化步数 | 诊断代理轮次/步数 |', '|---|---:|---:|---:|---:|']
    for row in costs:
        lines.append(f"| {row['seed']} | {row['coverage_selected_epoch']} | {row['coverage_epochs_completed']} | {row['coverage_optimizer_steps']} | {row['teacher_fixed_epochs']} / {row['teacher_optimizer_steps']} |")
    lines += ['', 'global控制本身不需要诊断代理参与预测或生成均匀对齐权重；当前统一核验实现仍计算代理以便与诊断控制共享流程，不据此宣称推断效率优势。协同闭式拟合、gate和验证搜索为额外成本；并行训练耗时不直接当作累计GPU时间。', '',
        '## 验证选择记录', '', '| 配方 | 五seed验证NDCG@50均值 |', '|---|---:|']
    for row in lock['candidates']:
        lines.append(f"| {row['recipe']} | {row['mean_validation_ndcg50']:.9f} |")
    lines += ['', 'seed2025初筛132个候选；进入五seed复验的有限配方在测试前固定，再按共同验证均值选一个配方。没有按测试逐seed挑选模型。已观察过的原测试仍属于迭代benchmark，需额外数据集或独立确认实验支撑泛化主张。', '',
        '## 配对不确定性', '', '| 对照 | 组 | 指标 | 选中配方 − 对照 | seed-bootstrap 95% CI | 双侧exact p | Holm p |', '|---|---|---|---:|---|---:|---:|']
    for r in comparisons:
        ci = r['bootstrap_95_ci']
        lines.append(f"| {r['control']} | {r['group']} | {r['metric']} | {r['mean_difference']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {r['two_sided_sign_flip_p']:.4f} | {r['holm_adjusted_p']:.4f} |")
    lines += ['', '固定幅度校准对照在新测试评分前声明为alpha=0.05、beta=0、mu=0，取自已有验证候选；它不参与七配方主模型选择，用于检查近似常数门控是否只是重调混合幅度。', '',
        '以上64项同属一个Holm比较族。区间只重采样五个训练seed，条件于同一用户与数据划分，属于描述性结果。双侧精确符号翻转最小非零p为0.0625；不添加虚构显著性星号，不把重复用户当额外独立训练重复。', '',
        '## 文件与来源', '',
        '[逐种子JSON](DIAGNOSTIC_V4_RESULTS.json)、[Overleaf表格](DIAGNOSTIC_V4_TABLE.tex)、[实现与选择说明](DIAGNOSTIC_V4_DESIGN_zh.md)。', '',
        '方法来源：[ACE，SIGIR 2026](https://arxiv.org/abs/2605.29322)，[EASE，WWW 2019](https://arxiv.org/abs/1905.03375)，[语义/协同互补研究，SIGIR 2026](https://arxiv.org/html/2604.22195v1)。ACE论文其他数据集的参考指标仍单独保存，未混入此ML-1M表。', '']
    for path, checksum in artifact_hashes.items():
        lines.append(f'- `{Path(path).name}` SHA256: `{checksum}`')
    rendered = '\n'.join(lines) + '\n'
    tex = [r'\begin{table*}[t]', r'\centering\small', r'\setlength{\tabcolsep}{4pt}',
        r'\caption{ML-1M test results under the shared chronological, multi-positive full-catalog protocol. Mean $\pm$ sample standard deviation over five seeds. The model recipe is selected by common validation NDCG@50; architectures and training budgets differ.}',
        r'\label{tab:fafirec-v4-ace}', r'\begin{tabular}{lcccc}', r'\hline',
        r'Model & Recall@10 & NDCG@10 & Recall@50 & NDCG@50 \\', r'\hline']
    for name in names:
        label = name.replace('_', r'\_') + (' (selected)' if name == selected else '')
        cells = [f"${rows[name, 'all', m]['mean']:.6f} \\pm {rows[name, 'all', m]['sample_std']:.6f}$" for m in METRICS]
        tex.append(label + ' & ' + ' & '.join(cells) + r' \\')
    tex += [r'\hline', r'\end{tabular}', r'\end{table*}', '']
    outputs = [ROOT / 'docs/DIAGNOSTIC_V4_RESULTS_zh.md', args.local_report,
        ROOT / 'docs/DIAGNOSTIC_V4_RESULTS.json', ROOT / 'docs/DIAGNOSTIC_V4_TABLE.tex']
    ace.require(not any(p.exists() for p in outputs), 'Choose fresh immutable report output paths')
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path in outputs[:2]:
        path.write_text(rendered, encoding='utf-8', newline='\n')
    outputs[2].write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    outputs[3].write_text('\n'.join(tex), encoding='utf-8', newline='\n')
    print('Complete locked five-seed report:', selected)


if __name__ == '__main__':
    main()
