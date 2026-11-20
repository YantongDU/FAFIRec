"""Analyze fixed ablations separately from frozen-model sensitivity tests."""
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
from experiments.ablation_v5 import experiment as ex
from experiments.diagnostic_v4 import final_runs as fr

METRICS = ('recall10', 'ndcg10', 'recall50', 'ndcg50')
LABELS = {'full': '完整模型（重训复现）', 'gain_only': '只学习全局幅度',
    'length_candidate': '仅长度诊断，保留候选项', 'no_length': '去显式历史长度',
    'user_only': '仅用户门控，去全部候选项', 'no_past': '去历史预测诊断',
    'no_current': '去当前用户级预测摘要', 'no_pop': '去候选频次项', 'no_conflict': '去候选冲突项',
    'mean_diagnostics': '诊断替换为训练均值', 'permuted_quality': '分层置换质量诊断',
    'scalar_mean_gate': '替换为训练期平均gate', 'user_mean_gate': '仅保留每用户平均gate'}


def checked_index(report):
    ace.require(report['seed_ids'] == ex.SEEDS, 'Five-seed order differs')
    values = {}
    for row in report['aggregates']:
        ace.require(row['seed_ids'] == ex.SEEDS and len(row['per_seed_scores']) == 5, 'Incomplete metric seeds')
        _same(statistics.mean(row['per_seed_scores']), row['mean'], 'Aggregate mean')
        _same(statistics.stdev(row['per_seed_scores']), row['sample_std'], 'Aggregate sample SD')
        key = row['variant'], row['group'], row['metric']
        ace.require(key not in values, 'Duplicate ablation metric')
        values[key] = row
    return values


def contrast(a, b, **metadata):
    delta = [x - y for x, y in zip(a['per_seed_scores'], b['per_seed_scores'])]
    return {**metadata, 'reference_scores': a['per_seed_scores'], 'comparison_scores': b['per_seed_scores'],
        'wins': sum(x > 0 for x in delta), 'ties': sum(x == 0 for x in delta), 'losses': sum(x < 0 for x in delta),
        **paired_seed_comparison(delta, bootstrap_samples=10000, seed=2025)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=Path('runs/ablation_v5'))
    parser.add_argument('--local-report', type=Path, default=Path('E:/论文/FAFIRec_消融实验与分析.md'))
    args = parser.parse_args(); root = args.run_root.resolve()
    plan = ex.read_plan(root); lock = ex.verify_lock(root / 'ablation_lock.json')
    validation, test = [ace.read(root / f'combined_{s}.json') for s in ('validation', 'test')]
    for report in (validation, test):
        ace.require(report['source_hashes'] == ex.source_identity() and report['all_artifact_hashes_verified']
            and report['ablation_lock_sha256'] == file_digest(root / 'ablation_lock.json'), 'Aggregate provenance differs')
        ex.verify_hashes(report['artifact_sha256'])
    rows = checked_index(test)
    proofs = {split: ace.read(root / f'{split}_verification.json') for split in ('validation', 'test')}
    for split, proof in proofs.items():
        ace.require(proof['complete'] and proof['seed_ids'] == ex.SEEDS
            and len(proof['results']) == 65 and {(r['seed'], r['variant']) for r in proof['results']}
            == {(s, v) for s in ex.SEEDS for v in ex.variants(plan)}, 'Replay coverage differs')
        for row in proof['results']:
            ace.require(file_digest(row['per_user_file']) == row['per_user_sha256']
                and file_digest(row['metrics_file']) == row['metrics_sha256'], 'Replay output changed')
            if row['variant'] == 'full':
                ace.require(row['baseline_full_identical'], 'Full model is not identical to frozen v4')
            if split == 'test':
                ace.require(row['independent_replay_identical']
                    and file_digest(row['reloaded_file']) == row['reloaded_sha256'] == row['per_user_sha256']
                    and file_digest(row['reloaded_metrics_file']) == row['reloaded_metrics_sha256'] == row['metrics_sha256'],
                    'Independent test gate reload differs')
    oldroot = Path(plan['baseline_root'])
    old = ace.read(oldroot / 'combined_test.json'); oldrows = checked_index(old)
    fr.verify_lock(oldroot / 'five_seed_selection_lock.json')
    for group in ex.GROUPS:
        for metric in METRICS:
            _same(rows['full', group, metric]['per_seed_scores'],
                oldrows['diagnostic_fusion', group, metric]['per_seed_scores'], 'Frozen full result parity')
    primary_family, descriptive = [], []
    for name in ex.variants(plan):
        if name == 'full': continue
        for group in ex.GROUPS:
            for metric in METRICS:
                row = contrast(rows['full', group, metric], rows[name, group, metric],
                    variant=name, category='sensitivity' if name in ex.SENSITIVITIES else 'retrained',
                    group=group, metric=metric, direction='full-minus-ablation')
                descriptive.append(row)
                if group == 'all' and metric == 'ndcg50': primary_family.append(row)
    ace.require(len(primary_family) == 12, 'Predeclared 12-comparison family differs')
    _holm_adjust(primary_family)
    prior_pairs = []
    for a, b, label in (
        ('uniform_alignment', 'global_fusion', '全局门控下增加关系对齐'),
        ('diagnostic_alignment', 'diagnostic_matched', '诊断门控下增加关系对齐'),
        ('diagnostic_matched', 'diagnostic_fusion', '加入beta=0.2热门先验')):
        for group in ex.GROUPS:
            for metric in METRICS:
                prior_pairs.append(contrast(oldrows[a, group, metric], oldrows[b, group, metric],
                    label=label, left=a, right=b, group=group, metric=metric))
    fixed = ace.read(oldroot / 'effective_scale_control/combined_test.json'); fixedrows = checked_index(fixed)
    fixed_comparison = contrast(rows['full', 'all', 'ndcg50'],
        fixedrows['global_no_pop_control', 'all', 'ndcg50'], label='full-minus-fixed-alpha0.05')
    shuffle_file = root / 'shuffled_identity/combined_test.json'
    ace.require(shuffle_file.exists(), 'Complete partial text identity intervention before final reporting')
    shuffle = ace.read(shuffle_file); ex.verify_hashes(shuffle['artifact_sha256']); sr = checked_index(shuffle)
    shuffle_comparisons = []
    for name, reference in (('shuffled_backbone_only', 'ace_coverage'),
        ('shuffled_backbone_frozen_fusion', 'diagnostic_fusion')):
        for group in ex.GROUPS:
            for metric in METRICS:
                shuffle_comparisons.append(contrast(oldrows[reference, group, metric], sr[name, group, metric],
                    variant=name, reference=reference, group=group, metric=metric, direction='real-identity-minus-shuffled'))
    gate_stats = []
    for row in proofs['test']['results']:
        gate_stats.append({k: row[k] for k in ('seed', 'variant', 'effective_multiplier_distribution', 'ranking_vs_full')})
    files = [root / 'plan.json', root / 'ablation_lock.json', root / 'combined_validation.json',
        root / 'combined_test.json', root / 'validation_verification.json', root / 'test_verification.json',
        shuffle_file, oldroot / 'combined_test.json', oldroot / 'effective_scale_control/combined_test.json', Path(__file__)]
    snapshot = {'schema': 'fafirec_v5_public_ablation_report_v1', 'report_date': '2026-11-20',
        'seed_ids': ex.SEEDS, 'primary_contrast': 'full-minus-gain_only; overall NDCG@50',
        'training_recipe': plan['training_recipe'], 'inference_parameters': plan['inference_parameters'],
        'specifications': plan['ablation_specifications'], 'test_aggregates': test['aggregates'],
        'validation_aggregates': validation['aggregates'], 'primary_family': primary_family,
        'descriptive_paired_comparisons': descriptive, 'existing_matched_pairs': prior_pairs,
        'fixed_scale_comparison': fixed_comparison, 'shuffled_identity_aggregates': shuffle['aggregates'],
        'shuffled_identity_paired': shuffle_comparisons, 'gate_and_ranking_statistics': gate_stats,
        'artifact_sha256': {str(p.resolve()): file_digest(p) for p in files},
        'no_hyperparameter_selection': True, 'historical_test_known': True}
    lines = ['# FAFIRec改造版：消融实验与分析', '', '日期：2026-11-20（Asia/Shanghai）。', '',
        '本轮补做45个固定预算门控训练（9种设置×5seed）、4类冻结门控敏感性检验，并补做5个主骨干文本身份打乱训练。旧模型、数据与评估协议保持冻结；没有按本轮测试分数选择或重新调参。', '',
        '## 1. 重训消融', '', '| 设置 | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |', '|---|---:|---:|---:|---:|']
    def add_table(names):
        for name in names:
            cells = [f"{rows[name, 'all', m]['mean']:.6f} ± {rows[name, 'all', m]['sample_std']:.6f}" for m in METRICS]
            lines.append('| ' + LABELS[name] + ' | ' + ' | '.join(cells) + ' |')
    add_table(plan['retrained_variants'])
    lines += ['', '所有版本保持原17个参数槽和L2分母。被禁用路径的参数维持零；特征先用原内层训练统计标准化再屏蔽，user项与频次斜率项同时屏蔽。full在五seed上重新训练，原参数、日志及验证/测试逐用户CSV均完全复现旧V4模型。', '',
        '## 2. 预先固定的主要比较', '', '| 比较：完整模型减消融 | NDCG@50差 | 95% seed-bootstrap CI | 胜/平/负 | exact p | Holm p |', '|---|---:|---|---|---:|---:|']
    for r in primary_family:
        ci = r['bootstrap_95_ci']
        lines.append(f"| {LABELS[r['variant']]} | {r['mean_difference']:+.8f} | [{ci[0]:+.8f}, {ci[1]:+.8f}] | {r['wins']}/{r['ties']}/{r['losses']} | {r['two_sided_sign_flip_p']:.4f} | {r['holm_adjusted_p']:.4f} |")
    gain = next(r for r in primary_family if r['variant'] == 'gain_only')
    ci = gain['bootstrap_95_ci']
    interpretation = ('区间跨零，当前没有稳定的诊断净收益证据。' if ci[0] <= 0 <= ci[1]
        else '完整模型均值更高，但五seed不确定性和其它校准对照仍需一同考虑。' if gain['mean_difference'] > 0
        else '去掉诊断的全局幅度版本更高，不支持保留完整门控以提高精度。')
    lines += ['', f"核心比较full−gain-only为 {gain['mean_difference']:+.8f}，胜出 {gain['wins']}/5 次。{interpretation}", '',
        f"此前固定alpha=0.05的融合对照与完整模型只差 {fixed_comparison['mean_difference']:+.8f}。本轮gain-only使用相同训练预算学习一个全局偏置，进一步区分幅度校准与个性化诊断。", '',
        '12项整体NDCG@50比较统一作Holm校正；其中full−gain-only为预先指定主关注项。双侧精确符号翻转在五seed下最小非零p为0.0625；不添加显著性星号。区间只重采样训练seed，条件于固定数据/用户，是描述性不确定性。负差表示删去该项反而更好，不会被解释成模块必要。', '',
        '## 3. 冷暖用户', '', '| 设置 | 冷用户NDCG@50 | 暖用户NDCG@50 |', '|---|---:|---:|']
    for name in plan['retrained_variants']:
        lines.append('| ' + LABELS[name] + ' | ' + ' | '.join(f"{rows[name,g,'ndcg50']['mean']:.6f} ± {rows[name,g,'ndcg50']['sample_std']:.6f}" for g in ('cold', 'warm')) + ' |')
    lines += ['', '## 4. 仅推理敏感性', '', '| 干预 | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |', '|---|---:|---:|---:|---:|']
    add_table(plan['sensitivity_variants'])
    lines += ['', '这些干预保持已训练权重不变，不能冒充重新训练消融。训练均值替换使标准化诊断为零，但候选频次偏置及冲突项仍可能起作用；分层置换联合打乱质量诊断并保留长度/缺失指示。平均gate来自训练观察前缀，不用验证或测试指标校准；每用户平均gate只去候选异质性。', '',
        '| 干预 | top10列表完全相同用户比例 | top50列表完全相同用户比例 | 有效alpha×gate均值 |', '|---|---:|---:|---:|']
    for name in plan['sensitivity_variants']:
        gs = [r for r in gate_stats if r['variant'] == name]
        lines.append(f"| {LABELS[name]} | {statistics.mean(r['ranking_vs_full']['top10_identical_fraction'] for r in gs):.4%} | {statistics.mean(r['ranking_vs_full']['top50_identical_fraction'] for r in gs):.4%} | {statistics.mean(r['effective_multiplier_distribution']['pooled_mean'] for r in gs):.6f} |")
    lines += ['', '完整的有效混合系数分布、标准差和确定性采样分位数保留在JSON；排名变化与指标增益是不同问题，不能因gate数值变化就声称推荐更好。', '',
        '## 5. 主骨干文本–物品对应关系', '', '| 配对 | 原对应NDCG@50 | 打乱对应NDCG@50 | 差值 |', '|---|---:|---:|---:|']
    for variant, reference, title in (('shuffled_backbone_only', 'ace_coverage', '序列骨干单独'),
        ('shuffled_backbone_frozen_fusion', 'diagnostic_fusion', '其余模块冻结的融合')):
        a, b = oldrows[reference,'all','ndcg50']['mean'], sr[variant,'all','ndcg50']['mean']
        lines.append(f'| {title} | {a:.6f} | {b:.6f} | {a-b:+.6f} |')
    lines += ['', '打乱仅作用于主骨干初始非PAD物品表，保留表的整体矩、奇异值结构、原编码器/位置初始化与训练随机数。五个打乱骨干均按原全转移训练配方、原验证早停规则训练，随后冻结其它分支与门控进行替换。诊断代理仍保留原文本信息，因此这是一项局部身份对应消融，不能写成完整w/o LLM；冻结其它模块也不代表整个无文本系统重训后的最优上限。', '',
        '## 6. 已有严格匹配对照', '', '| 只改变的配方因素 | 全部用户NDCG@50差 | 冷用户差 | 暖用户差 |', '|---|---:|---:|---:|']
    for label in dict.fromkeys(r['label'] for r in prior_pairs):
        values = [next(r for r in prior_pairs if r['label']==label and r['group']==g and r['metric']=='ndcg50')['mean_difference'] for g in ('all','cold','warm')]
        lines.append('| ' + label + ' | ' + ' | '.join(f'{v:+.8f}' for v in values) + ' |')
    lines += ['', '主方案mu本来就是0，不存在额外的w/o alignment主模型。上表对齐差来自其它相同beta/门控配方的mu0/0.1配对，同时会改变有效正则，不能将差值全归于语义对齐。加入beta时也会影响候选冲突输入，是该配方参数的系统效应，冷热用户可能有不同取舍。', '',
        '## 7. 对论文与实现的建议', '',
        '优先保留有明确实测收益的强训练布局、协同分支与分数校准。只有在相同幅度、同预算的重训对照下，诊断稳定带来额外收益，才适合把它作为核心必要机制。若gain-only、均值诊断或置换诊断与完整模型接近，应选择更简单的解释，而不是以门控激活图代替性能证据。', '',
        '原ACE到全转移ACE同时改变监督覆盖、窗口边界和实际优化步数，仅作为训练布局配方比较。固定系数下删除zscore会破坏量纲，本轮不把这种失配作为“语义必要性”证据。当前仍是已经观察过测试集的ML-1M机制分析，额外数据集和独立确认实验仍有价值。', '',
        '代码、源文件、锁与排名文件的指纹均保留；训练内层标签隔离、五种子完整锁、全模型原结果复现及双重门控重载都经过核对。新测试分数未用于选择或调整任何消融模型。', '']
    for path, checksum in snapshot['artifact_sha256'].items():
        lines.append(f'- `{Path(path).name}`: `{checksum}`')
    rendered = '\n'.join(lines) + '\n'
    tex = [r'\begin{table*}[t]', r'\centering\footnotesize', r'\setlength{\tabcolsep}{4pt}',
        r'\caption{Fixed-budget retraining ablations on ML-1M. Mean $\pm$ sample standard deviation across five seeds. All variants retain the same 17-parameter regularization space; no hyperparameters are reselected.}',
        r'\label{tab:fafirec-ablation-v5}', r'\begin{tabular}{lcccc}', r'\hline',
        r'Variant & Recall@10 & NDCG@10 & Recall@50 & NDCG@50 \\', r'\hline']
    for name in plan['retrained_variants']:
        cells = [f"${rows[name,'all',m]['mean']:.6f} \\pm {rows[name,'all',m]['sample_std']:.6f}$" for m in METRICS]
        tex.append(name.replace('_',r'\_') + ' & ' + ' & '.join(cells) + r' \\')
    tex += [r'\hline',r'\end{tabular}',r'\end{table*}','']
    outputs = [ROOT / 'docs/ABLATION_V5_RESULTS_zh.md', args.local_report,
        ROOT / 'docs/ABLATION_V5_RESULTS.json', ROOT / 'docs/ABLATION_V5_TABLE.tex']
    ace.require(not any(p.exists() for p in outputs), 'Choose fresh report output paths')
    for p in outputs: p.parent.mkdir(parents=True, exist_ok=True)
    for p in outputs[:2]: p.write_text(rendered,encoding='utf-8',newline='\n')
    outputs[2].write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    outputs[3].write_text('\n'.join(tex),encoding='utf-8',newline='\n')
    print('Complete fixed ablation report; full-minus-gain-only NDCG@50:',gain['mean_difference'])


if __name__ == '__main__': main()
