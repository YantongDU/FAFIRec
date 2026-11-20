"""Compare locked ACE/FAFIRec runs; keep paper-reported datasets separate."""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze_results import paired_seed_comparison, _holm_adjust
from analyze_transfer import _same
from fafirec.runner import file_digest
from tools import ace2026_experiment as ace
from tools.write_research_report import load_audited


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked_rows(report):
    rows = {}
    for row in report['aggregates']:
        values = row['per_seed_scores']
        ace.require(row['seed_ids'] == report['seed_ids'] and len(values) == 5, 'Seed score coverage differs')
        _same(statistics.mean(values), row['mean'], 'Aggregate mean')
        _same(statistics.stdev(values), row['sample_std'], 'Aggregate sample SD')
        key = row['variant'], row['stage'], row['group'], row['metric']
        ace.require(key not in rows, 'Duplicate aggregate metric')
        rows[key] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ace-root', type=Path, required=True)
    parser.add_argument('--fafirec-root', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, required=True)
    parser.add_argument('--repo-md', type=Path, required=True)
    parser.add_argument('--report-date', required=True)
    args = parser.parse_args()
    ace_root, ref_root = args.ace_root.resolve(), args.fafirec_root.resolve()
    val, test = read(ace_root / 'combined_validation.json'), read(ace_root / 'combined_test.json')
    ace.require(val['protocol'] == test['protocol'] == ace.PROTOCOL and val['split'] == 'validation'
                and test['split'] == 'test' and val['seed_ids'] == test['seed_ids'], 'ACE report identity differs')
    ace.require(val['all_artifact_hashes_verified'] and test['all_artifact_hashes_verified'], 'ACE audit incomplete')
    lock_path = Path(test['selection_lock_file'])
    lock = ace.verify_lock(lock_path)
    ace.require(val['selection_lock_sha256'] == test['selection_lock_sha256'] == file_digest(lock_path)
                and val['epoch_selections'] == test['epoch_selections'], 'ACE selection changed between splits')
    for report in (val, test):
        for path, checksum in report['artifact_sha256'].items():
            ace.require(file_digest(path) == checksum, 'ACE artifact changed: ' + path)
    for split in ('validation', 'test'):
        verification = read(ace_root / f'{split}_verification.json')
        ace.require(verification['complete'] and verification['seed_ids'] == test['seed_ids']
                    and verification['selection_lock_sha256'] == file_digest(lock_path), 'Replay verification incomplete')
        ace.require(len(verification['results']) == 5
                    and sorted(row['seed'] for row in verification['results']) == test['seed_ids'],
                    'Replay verification must cover each final seed exactly once')
        for row in verification['results']:
            job = next(j for j in lock['jobs'] if j['seed'] == row['seed'])
            ace.require(row['variant'] == job['variant'] and row['split'] == split, 'Replay job differs')
            ace.require(file_digest(row['per_user_file']) == row['per_user_sha256'], 'Ranking replay changed')
            saved = read(row['metrics_file'])
            ace.require(file_digest(row['metrics_file']) == row['metrics_sha256']
                        and saved['seed'] == row['seed'] and saved['variant'] == row['variant']
                        and saved['split'] == split and saved['per_user_sha256'] == row['per_user_sha256']
                        and saved['fingerprint'] == job['fingerprint'], 'Replay metrics identity differs')
            if split == 'test':
                ace.require(row['independent_replay_identical'] and file_digest(row['reloaded_file'])
                            == row['reloaded_sha256'] == row['per_user_sha256'], 'Independent test reload differs')
                ace.require(test['artifact_sha256'].get(row['per_user_file']) == row['per_user_sha256']
                            and test['artifact_sha256'].get(row['metrics_file']) == row['metrics_sha256'],
                            'Verified test replay differs from aggregate inputs')
                ace.require(file_digest(row['reloaded_metrics_file']) == row['reloaded_metrics_sha256'],
                            'Independent test metrics changed')
                _same(read(row['reloaded_metrics_file']), saved, 'Independent test metrics identity')
            else:
                original = str(Path(job['run_dir']) / f"seed_{job['seed']}/{job['variant']}/validation.csv")
                ace.require(row['identical_to_original'] and lock['artifact_sha256'][original]
                            == row['per_user_sha256'], 'Verified validation differs from selected rankings')
    tuning_path = ace_root.parent / 'ace2026_tuning/final_selection.json'
    tuning = read(tuning_path)
    ace.require(tuning['selection_split'] == 'validation' and tuning['no_new_test_used']
                and tuning['selected_variant'] == lock['common']['config']['variants'][0],
                'ACE hyperparameters differ from validation selection')
    ace.require(file_digest(tuning['final_config_file']) == tuning['final_config_sha256'],
                'Selected master config changed')
    final_config = ace.load_config(tuning['final_config_file'])
    ace.require({k: v for k, v in final_config.items() if k not in {'output_dir', 'seeds'}}
                == lock['common']['config'], 'Selected training budget changed')
    for path, checksum in tuning['artifact_sha256'].items():
        ace.require(file_digest(path) == checksum, 'ACE validation tuning artifact changed: ' + path)
    # Reuse the existing complete audit reader; no model is evaluated here.
    load_audited(ref_root)
    ref_val, ref_test = read(ref_root / 'combined_validation.json'), read(ref_root / 'combined_test.json')
    ace.require(ref_test['seed_ids'] == test['seed_ids'], 'Comparison seeds differ')
    common = test['common']
    ace.require(common['identity'] == ref_test['common_provenance']['identity'], 'Compared inputs differ')
    ace.require(common['evaluation_config'] == ref_test['common_provenance']['evaluation_config'], 'Compared evaluation differs')
    ace.require(ref_val['decision'] == ref_test['decision'], 'FAFIRec method reselected on test')
    method = (ref_val['decision']['selected_variant'], ref_val['decision']['selected_stage'])
    ace_method = (lock['common']['config']['variants'][0]['name'], 'single')
    ar, fr = checked_rows(test), checked_rows(ref_test)
    comparisons = []
    for group in ('all', 'cold', 'middle', 'warm'):
        for metric in ('recall10', 'ndcg10', 'recall50', 'ndcg50'):
            a, f = ar[*ace_method, group, metric], fr[*method, group, metric]
            differences = [x - y for x, y in zip(f['per_seed_scores'], a['per_seed_scores'])]
            comparisons.append({'group': group, 'metric': metric, 'direction': 'FAFIRec-minus-ACE',
                'fafirec_scores': f['per_seed_scores'], 'ace_scores': a['per_seed_scores'],
                'relative_mean_change_percent': 100. * (f['mean'] / a['mean'] - 1.) if a['mean'] else None,
                **paired_seed_comparison(differences, bootstrap_samples=10000, seed=2025)})
    _holm_adjust(comparisons)
    artifacts = {str(p): file_digest(p) for p in (ace_root / 'combined_validation.json', ace_root / 'combined_test.json',
        lock_path, ref_root / 'combined_validation.json', ref_root / 'combined_test.json',
        ace_root / 'validation_verification.json', ace_root / 'test_verification.json', tuning_path,
        ROOT / 'docs/ACE_PAPER_BASELINE_REFERENCE.csv', Path(__file__))}
    costs = []
    for job in lock['jobs']:
        summary = read(Path(job['run_dir']) / f"seed_{job['seed']}/{job['variant']}/summary.json")
        selected_stages = [r for r in ref_test['selections']
                           if r['seed'] == job['seed'] and r['variant'] == method[0]
                           and r['stage'] in ('expert', 'router')]
        ace.require(len(selected_stages) == 2, 'FAFIRec selected two-stage cost coverage differs')
        costs.append({'seed': job['seed'], 'ace_selected_epoch': summary['selected_epoch'],
            'ace_epochs_completed': summary['epochs_completed'], 'ace_optimizer_steps': summary['optimizer_steps'],
            'ace_trainable_parameters': summary['trainable_parameter_count'],
            'ace_elapsed_seconds_this_invocation': summary['elapsed_seconds_this_invocation'],
            'fafirec_expert_plus_router_steps': sum(r['optimizer_steps'] for r in selected_stages),
            'fafirec_stages': [{k: r[k] for k in ('stage', 'selected_epoch', 'epochs_completed',
                'optimizer_steps', 'active_trainable_parameter_count')} for r in selected_stages]})
    result = {'schema': 'ace2026_fafirec_same_input_comparison_v1', 'seed_ids': test['seed_ids'],
        'ace_model': ace_method, 'fafirec_model': method, 'paired_comparisons': comparisons,
        'ace_recipe': common['config'], 'ace_epoch_selections': test['epoch_selections'], 'training_costs': costs,
        'artifact_sha256': artifacts, 'test_model_selection': False,
        'scope': 'Shared inputs/evaluation; different native architectures, objectives and training schedules. Historical test exposed.'}
    lines = ['# ACE（SIGIR 2026）与当前 FAFIRec 的比较', '', f'报告日期：{args.report_date}（Asia/Shanghai）。', '',
        '本表为同一ML-1M输入和评估协议下的实际五种子实验；FAFIRec指前轮完成的文本残差改造版本。原稿Table II的历史数值不参与此处统计，ACE原论文其他数据集的数值另列为文献参考。', '',
        f'FAFIRec固定为原共同验证选择 `{method[0]}/{method[1]}`；ACE固定为验证选出的同一谱参数、尺度和学习率，各seed只按验证选择epoch。没有根据测试改选模型。', '']
    for group, label in (('all', '全部用户'), ('cold', '冷用户'), ('warm', '暖用户')):
        lines += ['## ' + label, '', '| 模型 | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |', '|---|---:|---:|---:|---:|']
        for name, rows, key in (('当前FAFIRec', fr, method), ('ACE-SASRec（统一协议适配）', ar, ace_method)):
            values = [f"{rows[*key, group, m]['mean']:.6f} ± {rows[*key, group, m]['sample_std']:.6f}"
                      for m in ('recall10', 'ndcg10', 'recall50', 'ndcg50')]
            lines.append('| ' + name + ' | ' + ' | '.join(values) + ' |')
        lines.append('')
    primary = next(r for r in comparisons if r['group'] == 'all' and r['metric'] == 'ndcg50')
    direction = 'FAFIRec较高' if primary['mean_difference'] > 0 else 'ACE较高' if primary['mean_difference'] < 0 else '相同'
    lines += [f"以预先指定的整体NDCG@50看，{direction}；FAFIRec减ACE的均值差为 {primary['mean_difference']:+.6f}。其他指标和分组分别见表，不能将一个指标扩展为全面胜出。", '',
        '## 配对比较', '', '| 用户组 | 指标 | FAFIRec − ACE | seed-bootstrap 95% CI | 双侧exact p | Holm p |', '|---|---|---:|---|---:|---:|']
    for row in comparisons:
        ci = row['bootstrap_95_ci']
        lines.append(f"| {row['group']} | {row['metric']} | {row['mean_difference']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {row['two_sided_sign_flip_p']:.4f} | {row['holm_adjusted_p']:.4f} |")
    lines += ['', '区间仅重采样五个训练seed，条件于同一固定评估用户；种子少，区间为描述性结果。双侧精确符号翻转只有32种符号，最小非零p为0.0625；Holm覆盖上述全部16项。没有把重复用户当成额外独立样本。', '',
        '## 方法与比较边界', '',
        '共同使用5950用户、3883物品、同一Qwen3-Embedding-0.6B静态元数据向量、逐用户时间80/10/10划分、全目录排除观察历史、多正例用户宏平均指标。ACE的谱分解只使用静态内容，不使用验证或测试反馈。', '',
        'ACE保留未中心化SVD谱初始化、whole-matrix标准差缩放、可训练物品表、raw-dot SASRec和原生全类CE。它使用128维/2层/批量256，从头训练；FAFIRec使用64维/1层/批量2048及CF预训练、不同损失与残差结构。因此是同输入/同评估比较，不能称严格容量、损失或计算量匹配的单因素实验。训练布局为同一最后50个shifted位置，和作者原生one-prefix加载方式不同；不能称原论文分数复现。', '',
        'ACE先在seed2025上比较九组谱参数/尺度，再比较两组学习率，锁定配方后做五seed复验；FAFIRec沿用原先五seed共同验证的variant/stage选择。二者调参策略和搜索预算也不完全匹配。', '',
        '实际训练记录（ACE每个模型768,768个可训练参数）：', '',
        '| seed | ACE选中轮次 | ACE实际训练轮次 | ACE优化步数 | FAFIRec expert+router步数 |',
        '|---|---:|---:|---:|---:|']
    for row in costs:
        lines.append(f"| {row['seed']} | {row['ace_selected_epoch']} | {row['ace_epochs_completed']} | {row['ace_optimizer_steps']} | {row['fafirec_expert_plus_router_steps']} |")
    lines += ['', '步数对应不同批量和损失，不可直接视为计算量比。FAFIRec此处未计入CF预训练、文本编码或调参；ACE调用耗时保留在JSON，若中途恢复则只代表该次调用，不作为累计耗时。没有据此宣称效率优势。', '',
        'ACE实际共同配置：', '', '```json', json.dumps(common['config'], ensure_ascii=False, indent=2), '```', '',
        '## ACE原论文其他baseline：仅作文献参考', '',
        '下表为作者原论文Table1的ML-20M结果，平均4次运行，leave-one-out；不是上方ML-1M实测，不与本机五个seed合并统计。原文未给出的标准差或@50数值没有补造。', '',
        '| 模型（原论文ML-20M） | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |', '|---|---:|---:|---:|---:|']
    with (ROOT / 'docs/ACE_PAPER_BASELINE_REFERENCE.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['dataset'] == 'ML-20M':
                lines.append('| ' + row['model'] + ' | ' + ' | '.join(row[m] for m in ('Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20')) + ' |')
    lines += ['', '来源：[ACE论文Table1](https://arxiv.org/html/2605.29322v1#S3.T1)，[官方代码](https://github.com/DCheol/ACE)，固定commit `3e099052d9759b841a5192a2ae20a3b3ac02c6c6`。四数据集完整文献参考值见 [参考表](https://github.com/YantongDU/FAFIRec/blob/codex/ace2026-baseline/docs/ACE_PAPER_BASELINE_REFERENCE_zh.md)。', '',
        '原测试分数在历史开发中已被观察，本轮是透明迭代benchmark，不是未触碰的首次确认集；不据此宣称SOTA或Strong Accept。', '',
        '## 审计文件', '']
    for path, checksum in artifacts.items():
        lines.append(f'- [{Path(path).name}](<{path}>): `{checksum}`')
    rendered = '\n'.join(lines) + '\n'
    for path in (args.output_json, args.output_md, args.repo_md):
        ace.require(not path.exists(), 'Choose new report output paths')
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    for path in (args.output_md, args.repo_md):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding='utf-8', newline='\n')
    print('Complete locked comparison and source-labeled paper reference generated.')


if __name__ == '__main__':
    main()
