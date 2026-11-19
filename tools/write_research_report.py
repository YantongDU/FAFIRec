"""Render audited five-seed Research v3 JSON reports; never run a model.

Requires combined_validation.json, combined_test.json and their common immutable
validation lock. Missing/incomplete inputs fail before any report is written.
Only Python's standard library is used; this utility never imports torch.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCHEMA = 'research_v3_combined_five_seed_report_v1'
GROUPS = ('all', 'cold', 'warm')
METRICS = ('recall10', 'ndcg10', 'recall50', 'ndcg50')
METRIC_LABEL = dict(zip(METRICS, ('R@10', 'N@10', 'R@50', 'N@50')))
GROUP_LABEL = {'all': '全部用户', 'cold': 'cold 用户', 'warm': 'warm 用户'}
KIND_LABEL = {'id': 'ID', 'text': '真实文本', 'shuffled': '打乱文本'}
STAGE_LABEL = {'frozen': '冻结', 'expert': 'global', 'router': 'relative'}
ANCHOR = ('strong_cf_anchor', 'frozen')
SINGLE = ('single_cf_ce', 'single')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def number(value):
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value), '需要有限实数')
    return value


def same(actual, expected, label):
    require(math.isclose(number(actual), number(expected), rel_tol=1e-10, abs_tol=1e-12),
            label + ' 与独立复算不一致')


def metric_key(row):
    return row['variant'], row['stage'], row['group'], row['metric']


def comparison_key(row):
    return (row['comparison'], tuple(row['candidate']), tuple(row['baseline']),
            row['group'], row['metric'])


def audit_aggregates(report, models, seeds):
    indexed = {metric_key(row): row for row in report['aggregates']}
    expected = {(v, s, g, m + str(k)) for v, s in models
                for g in ('all', 'cold', 'middle', 'warm')
                for m in ('recall', 'ndcg', 'hit') for k in (10, 50)}
    require(len(indexed) == len(report['aggregates']) and set(indexed) == expected,
            '聚合指标缺失、重复或模型网格不完整')
    for row in indexed.values():
        scores = row['per_seed_scores']
        require(row['seed_ids'] == seeds and row.get('seed_count') == 5 and len(scores) == 5,
                '聚合种子顺序或数量不一致')
        require(all(0 <= number(v) <= 1 for v in scores), '排序指标超出 [0,1]')
        same(row['mean'], statistics.mean(scores), '指标均值')
        same(row['sample_std'], statistics.stdev(scores), '指标样本标准差')
    return indexed


def audit_comparisons(report, metrics, seeds):
    rows = report['paired_comparisons']
    indexed = {comparison_key(row): row for row in rows}
    require(len(indexed) == len(rows) and rows, '配对比较缺失或重复')
    for row in rows:
        candidate, baseline = tuple(row['candidate']), tuple(row['baseline'])
        left = metrics[*candidate, row['group'], row['metric']]['per_seed_scores']
        right = metrics[*baseline, row['group'], row['metric']]['per_seed_scores']
        differences = [a - b for a, b in zip(left, right)]
        require(row['seed_ids'] == seeds and row['paired_seeds'] == 5,
                '配对种子或推断范围不一致')
        for key, expected in [('candidate_scores', left), ('baseline_scores', right),
                              ('per_seed_differences', differences)]:
            require(len(row[key]) == 5, '配对数组长度不一致')
            for actual, value in zip(row[key], expected):
                same(actual, value, key)
        mean = statistics.mean(differences)
        same(row['mean_difference'], mean, '配对差均值')
        same(row['sample_std_difference'], statistics.stdev(differences), '配对差样本标准差')
        ci = row['bootstrap_95_ci']
        require(isinstance(ci, list) and len(ci) == 2 and number(ci[0]) <= number(ci[1]),
                '缺少有效 seed-bootstrap 区间')
        tolerance = 1e-12 * max(1., abs(mean))
        p = sum(abs(sum(s * d for s, d in zip(signs, differences)) / 5)
                >= abs(mean) - tolerance
                for signs in itertools.product((-1., 1.), repeat=5)) / 32
        same(row['two_sided_sign_flip_p'], p, 'exact sign-flip p')
        require(0 <= number(row['holm_adjusted_p']) <= 1, '无效 Holm p')
        expected_relative = 100 * mean / statistics.mean(right) if statistics.mean(right) else None
        if expected_relative is None:
            require(row['relative_mean_gain_percent'] is None, '零基线相对变化应为空')
        else:
            same(row['relative_mean_gain_percent'], expected_relative, '相对变化')
    running = 0.
    ordered = sorted(rows, key=lambda row: row['two_sided_sign_flip_p'])
    for rank, row in enumerate(ordered):
        running = max(running, min(1., (len(rows) - rank) * row['two_sided_sign_flip_p']))
        same(row['holm_adjusted_p'], running, '完整比较族 Holm p')
    return indexed


def load_audited(replica_root):
    root = replica_root.resolve()
    paths = {split: root / f'combined_{split}.json' for split in ('validation', 'test')}
    require(all(p.is_file() for p in paths.values()),
            '尚无完整 combined_validation.json 和 combined_test.json；不生成结果报告')
    reports = {split: read(path) for split, path in paths.items()}
    validation, test = reports['validation'], reports['test']
    for split, report in reports.items():
        require(report.get('schema') == SCHEMA and report.get('split') == split
                and report.get('replica_count') == 5 and report.get('jobs_per_replica') == 6
                and report.get('complete_configured_grid_verified') is True
                and report.get('master_plan_verified') is True
                and report.get('model_evaluation_called') is False, '输入不是完整审计后的五种子汇总')
        require(report.get('missing_matched_controls') == [], '缺少预先声明的匹配控制')
        same(report['signflip_minimum_two_sided_p'], .0625, '五种子 exact p 分辨率')
        for filename, checksum in report['artifact_sha256'].items():
            require(sha256(Path(filename)) == checksum, '报告绑定产物已改变：' + filename)
    seeds, cfg = validation['seed_ids'], validation['master_config']
    require(len(seeds) == len(set(seeds)) == 5 and cfg['seeds'] == seeds, '必须是五个不同的共同种子')
    for key in ('seed_ids', 'master_config', 'common_provenance', 'decision', 'selections',
                'selection_lock_file', 'selection_lock_sha256', 'selection_lock_fingerprint',
                'bootstrap_samples', 'bootstrap_seed', 'combiner_sha256'):
        require(validation[key] == test[key], '验证与测试汇总不共用同一锁定设置：' + key)
    variants = {v['name']: v for v in cfg['variants']}
    require(len(variants) == 3 and {v['kind'] for v in variants.values()} == {'id', 'text', 'shuffled'}
            and cfg['train_router'] is True, '需要 ID/真实文本/打乱文本 × 两阶段，加冻结 CF，共七行')
    require(all({k: value for k, value in v.items() if k not in {'name', 'kind'}}
                == {k: value for k, value in cfg['variants'][0].items() if k not in {'name', 'kind'}}
                for v in variants.values()), '三个控制的训练配方不匹配')
    models = [ANCHOR] + [(name, stage) for kind in ('id', 'text', 'shuffled')
                         for name, variant in variants.items() if variant['kind'] == kind
                         for stage in ('expert', 'router')]
    metrics, comparisons = {}, {}
    for split, report in reports.items():
        metrics[split] = audit_aggregates(report, models, seeds)
        comparisons[split] = audit_comparisons(report, metrics[split], seeds)
        selected_rows = {metric_key(r): r for r in report['locked_selected_model_scores']}
        chosen = (report['decision']['selected_variant'], report['decision']['selected_stage'])
        require(selected_rows == {k: r for k, r in metrics[split].items() if k[:2] == chosen},
                '锁定模型分数与完整聚合不一致')
    require(set(comparisons['validation']) == set(comparisons['test']), '测试改变了比较族')
    ranked = sorted((metrics['validation'][v, s, 'all', 'ndcg50'] for v, s in models if (v, s) != ANCHOR),
                    key=lambda r: (-r['mean'], r['variant'], r['stage']))
    decision = validation['decision']
    chosen = decision['selected_variant'], decision['selected_stage']
    require(chosen == (ranked[0]['variant'], ranked[0]['stage'])
            and decision['candidates'] == ranked, '最终选择不是共同五种子验证 NDCG@50 决定')
    lock_path = Path(validation['selection_lock_file']).resolve()
    require(lock_path.is_file() and sha256(lock_path) == validation['selection_lock_sha256'], '共同验证锁已缺失或改变')
    lock = read(lock_path)
    require(lock['lock_fingerprint'] == digest({k: v for k, v in lock.items() if k != 'lock_fingerprint'})
            == validation['selection_lock_fingerprint'] and lock['decision'] == decision
            and lock['selection_split'] == 'validation' and lock['master_config'] == cfg
            and lock['seed_ids'] == seeds and lock['new_test_used_for_selection'] is False
            and lock['historical_test_known'] is True and Path(lock['replica_root']).resolve() == root,
            '共同验证锁内容或指纹不一致')
    selections = {(r['seed'], r['variant'], r['stage']): r for r in validation['selections']}
    require(len(selections) == len(validation['selections']) == 30
            and set(selections) == {(seed, v, s) for seed in seeds for v, s in models if (v, s) != ANCHOR},
            '参数/训练/校准记录不是完整三十个模型')
    return dict(root=root, paths=paths, reports=reports, seeds=seeds, cfg=cfg, variants=variants,
                models=models, metrics=metrics, comparisons=comparisons, chosen=chosen,
                selections=selections, lock_path=lock_path,
                report_date=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d'))


def load_single_audited(context, test_path, validation_path=None):
    """Attach an independently locked baseline; never change the core choice."""
    test_path = test_path.resolve()
    validation_path = (validation_path.resolve() if validation_path is not None
                       else test_path.parent / 'combined_validation.json')
    paths = {'validation': validation_path, 'test': test_path}
    require(all(p.is_file() for p in paths.values()), '单分支 CF-CE 需要完整验证和测试汇总')
    reports = {split: read(path) for split, path in paths.items()}
    validation, test = reports['validation'], reports['test']
    shared = context['reports']['validation']['common_provenance']
    for split, report in reports.items():
        require(report.get('protocol') == 'single_cf_catalog_ce_v1' and report.get('split') == split
                and report.get('seed_ids') == context['seeds']
                and report.get('all_artifact_hashes_verified') is True,
                '单分支 CF-CE 不是完整独立审计后的五种子报告')
        for key in ('identity', 'runtime', 'evaluation_config', 'reference'):
            require(report[key] == shared[key], '单分支 CF-CE 与主实验不可直接比较：' + key)
        require(all(report['source_hashes'].get(name) == checksum
                    for name, checksum in shared['source_hashes'].items()),
                '单分支 CF-CE 与核心模型共享源码不同')
        project = Path(__file__).resolve().parents[1]
        for filename, checksum in report['source_hashes'].items():
            require(sha256(project / filename) == checksum, '单分支 CF-CE 源码已改变：' + filename)
        for filename, checksum in report['artifact_sha256'].items():
            require(sha256(Path(filename)) == checksum, '单分支 CF-CE 产物已改变：' + filename)
    for key in ('seed_ids', 'selection_lock_file', 'selection_lock_sha256', 'selection_lock_fingerprint',
                'source_hashes', 'runtime', 'identity', 'evaluation_config', 'reference', 'recipe', 'epoch_selections'):
        require(validation[key] == test[key], '单分支 CF-CE 验证/测试设置或 epoch 选择不一致：' + key)
    lock_path = Path(validation['selection_lock_file']).resolve()
    require(sha256(lock_path) == validation['selection_lock_sha256'], '单分支 CF-CE 验证锁已改变')
    lock = read(lock_path)
    require(lock.get('protocol') == 'single_cf_catalog_ce_v1'
            and lock.get('selection_split') == 'validation'
            and lock.get('new_test_used_for_selection') is False and lock.get('historical_test_known') is True
            and lock.get('seed_ids') == context['seeds'] and lock.get('master_config') == context['cfg']
            and lock.get('source_hashes') == validation['source_hashes']
            and lock.get('runtime') == validation['runtime']
            and lock.get('lock_fingerprint') == digest({k: v for k, v in lock.items() if k != 'lock_fingerprint'})
            == validation['selection_lock_fingerprint'], '单分支 CF-CE 验证锁内容或指纹不一致')
    require(sha256(Path(lock['master_config_file'])) == lock['master_config_sha256'],
            '单分支 CF-CE 锁定 master 配置文件已改变')
    for filename, checksum in lock['validation_artifact_sha256'].items():
        require(sha256(Path(filename)) == checksum, '单分支 CF-CE 锁定模型或验证产物已改变')
    selections = {row['seed']: row for row in validation['epoch_selections']}
    require(len(selections) == len(validation['epoch_selections']) == 5
            and set(selections) == set(context['seeds']), '单分支 CF-CE epoch 记录缺失或重复')
    metrics = {split: audit_aggregates(report, [SINGLE], context['seeds'])
               for split, report in reports.items()}
    for seed, row in selections.items():
        index = context['seeds'].index(seed)
        same(row['validation_ndcg50'], metrics['validation'][*SINGLE, 'all', 'ndcg50']['per_seed_scores'][index],
             '单分支 CF-CE epoch 验证分数')
    locked_jobs = {row['seed']: row for row in lock['jobs']}
    require(len(locked_jobs) == len(lock['jobs']) == 5 and set(locked_jobs) == set(context['seeds']),
            '单分支 CF-CE 验证锁的任务覆盖不完整')
    for seed, row in locked_jobs.items():
        require(row['selected_epoch'] == selections[seed]['selected_epoch'], '单分支 CF-CE 锁定 epoch 已改变')
        index = context['seeds'].index(seed)
        for key, metric in metrics['validation'].items():
            same(row['validation'][key[2]][key[3]], metric['per_seed_scores'][index],
                 '单分支 CF-CE 锁定验证指标')
    same(lock['mean_validation_ndcg50'], metrics['validation'][*SINGLE, 'all', 'ndcg50']['mean'],
         '单分支 CF-CE 锁定验证均值')
    recipe = validation['recipe']
    for key in ('epochs', 'min_epochs', 'patience', 'eval_interval', 'batch_size', 'eval_batch',
                'ce_chunk_size', 'threads', 'device', 'grad_clip'):
        require(recipe[key] == context['cfg'][key], '单分支 CF-CE 预算不匹配：' + key)
    require(recipe['score_scale'] == 16. and recipe['lr'] == .001 and recipe['weight_decay'] == 1e-5
            and recipe['optimizer'] == 'Adam' and recipe['dtype'] == 'float32'
            and recipe['initial_checkpoint_eligible'] is True, '单分支 CF-CE 训练配方不符')
    for split in ('validation', 'test'):
        context['metrics'][split].update(metrics[split])
    context['single'] = dict(paths=paths, reports=reports, metrics=metrics,
                             selections=selections, lock_path=lock_path)
    context['single_comparisons'] = single_comparisons(context)
    return context


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    low = math.floor(index)
    return values[low] + (index - low) * (values[min(low + 1, len(values) - 1)] - values[low])


def single_comparisons(context):
    """A separately disclosed family, with all 5**5 bootstrap samples exact."""
    rows = []
    for group in ('all', 'cold', 'middle', 'warm'):
        for metric in METRICS:
            left = context['metrics']['test'][*context['chosen'], group, metric]['per_seed_scores']
            right = context['metrics']['test'][*SINGLE, group, metric]['per_seed_scores']
            differences = [a - b for a, b in zip(left, right)]
            mean = statistics.mean(differences)
            bootstrap = [statistics.mean(differences[i] for i in indices)
                         for indices in itertools.product(range(5), repeat=5)]
            tolerance = 1e-12 * max(1., abs(mean))
            p = sum(abs(sum(s * d for s, d in zip(signs, differences)) / 5) >= abs(mean) - tolerance
                    for signs in itertools.product((-1., 1.), repeat=5)) / 32
            rows.append({'group': group, 'metric': metric, 'candidate_scores': left,
                'baseline_scores': right, 'mean_difference': mean, 'per_seed_differences': differences,
                'bootstrap_95_ci': [percentile(bootstrap, .025), percentile(bootstrap, .975)],
                'two_sided_sign_flip_p': p})
    core = [{'two_sided_sign_flip_p': row['two_sided_sign_flip_p']}
            for row in context['reports']['test']['paired_comparisons']]
    family = core + rows
    running = 0.
    for rank, row in enumerate(sorted(family, key=lambda row: row['two_sided_sign_flip_p'])):
        running = max(running, min(1., (len(family) - rank) * row['two_sided_sign_flip_p']))
        row['holm_adjusted_p'] = running
        row['holm_family_size'] = len(family)
    return rows


def display_models(context):
    return context['models'] + ([SINGLE] if context.get('single') else [])


def label(context, model):
    if model == ANCHOR:
        return '强 CF anchor'
    if model == SINGLE:
        return '单分支 CF-CE（本地实现）'
    name, stage = model
    return KIND_LABEL[context['variants'][name]['kind']] + ' / ' + STAGE_LABEL[stage]


def fmt(value, signed=False):
    return format(0. if abs(value) < .5e-6 else value, '+.6f' if signed else '.6f')


def mean_sd(row, latex=False):
    separator = r' \pm ' if latex else ' ± '
    return fmt(row['mean']) + separator + fmt(row['sample_std'])


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def link(path, title=None):
    return f'[{title or path.name}](<{path.as_posix()}>)'


def required_pairs(context):
    by_kind = {v['kind']: name for name, v in context['variants'].items()}
    pairs = [('candidate-minus-frozen-CF', context['chosen'], ANCHOR, '验证所选 − 强 CF')]
    for stage in ('expert', 'router'):
        for kind, key, title in [('id', 'semantic-minus-matched-ID', '真实文本 − ID'),
                                 ('shuffled', 'text-minus-shuffled', '真实文本 − 打乱文本')]:
            pairs.append((key, (by_kind['text'], stage), (by_kind[kind], stage), title + ' / ' + STAGE_LABEL[stage]))
    for kind in ('id', 'text', 'shuffled'):
        name = by_kind[kind]
        pairs.append(('relative-router-minus-global', (name, 'router'), (name, 'expert'),
                      KIND_LABEL[kind] + '：relative − global'))
    return pairs


def comparison_table(context, groups, metrics):
    rows, indexed = [], context['comparisons']['test']
    for key, candidate, baseline, title in required_pairs(context):
        for group in groups:
            for metric in metrics:
                row = indexed.get((key, candidate, baseline, group, metric))
                require(row is not None, '缺少必要配对比较：' + title)
                ci = row['bootstrap_95_ci']
                rows.append([title, GROUP_LABEL[group], METRIC_LABEL[metric], fmt(row['mean_difference'], True),
                             ', '.join(fmt(d, True) for d in row['per_seed_differences']),
                             f'[{fmt(ci[0])}, {fmt(ci[1])}]',
                             f"{row['two_sided_sign_flip_p']:.4f}", f"{row['holm_adjusted_p']:.4f}"])
    return table(['比较', '用户组', '指标', '均值差', '逐 seed 差（按声明顺序）',
                  'seed-bootstrap 95% CI', 'exact p', 'Holm p'], rows)


def single_comparison_table(context):
    rows = []
    for row in context['single_comparisons']:
        if row['group'] not in GROUPS:
            continue
        ci = row['bootstrap_95_ci']
        rows.append([GROUP_LABEL[row['group']], METRIC_LABEL[row['metric']],
            fmt(row['mean_difference'], True), ', '.join(fmt(d, True) for d in row['per_seed_differences']),
            f'[{fmt(ci[0])}, {fmt(ci[1])}]', f"{row['two_sided_sign_flip_p']:.4f}",
            f"{row['holm_adjusted_p']:.4f}"])
    return table(['用户组', '指标', '原七行验证所选 − 单分支 CF-CE', '逐 seed 差',
                  '全枚举 seed-bootstrap 95% CI', 'exact p', '合并族 Holm p'], rows)


def render_markdown(context):
    cfg, reports = context['cfg'], context['reports']
    selected, metrics = context['chosen'], context['metrics']
    seeds, selections = context['seeds'], context['selections']
    selected_name = label(context, selected)
    selected_kind = context['variants'][selected[0]]['kind']
    get = lambda model, metric, group='all', split='test': metrics[split][*model, group, metric]
    changes = [get(selected, m)['mean'] - get(ANCHOR, m)['mean'] for m in METRICS[:2]]
    direction = '两项点估计均提高' if all(d > 0 for d in changes) else ('两项点估计均下降' if all(d < 0 for d in changes) else '两项点估计有升有降或持平')
    lines = ['# FAFIRec：文献改造与五种子实验结果', '',
             f"报告日期：{context['report_date']}（Asia/Shanghai）。所有数值来自完整独立审计后的 JSON，未运行模型或重新挑选测试最优行。", '',
             f'原七行模型的共同验证锁选择 **{selected_name}**（`{selected[0]}/{selected[1]}`）。其测试 R@10 为 **{mean_sd(get(selected, "recall10"))}**，N@10 为 **{mean_sd(get(selected, "ndcg10"))}**；相对冻结强 CF，{direction}，绝对均值差分别为 {fmt(changes[0], True)}、{fmt(changes[1], True)}。', '']
    if context.get('single'):
        n10 = get(selected, 'ndcg10')['mean'] - get(SINGLE, 'ndcg10')['mean']
        n50 = get(selected, 'ndcg50')['mean'] - get(SINGLE, 'ndcg50')['mean']
        lines += [f'相对附加单分支 CF-CE，本方法的测试 N@10/N@50 点估计差为 **{fmt(n10, True)}/{fmt(n50, True)}**。原七行共同锁只决定原方法候选中的选择，不能当作包含这一新增基线后的总体最优证明。', '']
    if selected_kind != 'text':
        lines += [f'获选分支使用{KIND_LABEL[selected_kind]}，因此不能把本轮整体结果写成真实文本模型获胜。文本是否有增量，应看下方真实文本与 ID、打乱文本的对照。', '']
    else:
        lines += ['真实文本被验证选中，不等于已隔离出语义收益；仍须同时比较强 ID 继续训练/集成对照和同结构打乱文本。', '']
    lines += ['原测试集在历史开发中已被观察。本轮新 checkpoint、校准与共同 variant/stage 选择只使用验证集，属于**透明的迭代 benchmark**，不能声称 untouched holdout、SOTA 或达到 Strong Accept。', '',
              '## 完整测试结果', '', f"以下全部为五个训练 seed 的均值 ± 样本标准差，单位是原始 [0,1] 指标，不是百分数。保留强 CF 与三类专家的两个阶段，共七行{'，另列独立单分支 CF-CE 第八行' if context.get('single') else ''}；“†”只标记原七行共同验证锁的获选行，不标记测试最优。cold/warm 按用户训练历史长度分组，不是候选物品冷暖。", '']
    if not context.get('single'):
        lines[lines.index('## 完整测试结果'):lines.index('## 完整测试结果')] = [
            '**尚缺单分支 CF-CE 对照，不能孤立归因融合结构。** 当前相对旧 CF 的变化包含额外训练、CE 配方、校准与融合的共同作用；ID residual 也有残差/集成结构，不能代替单分支 CE 基线。', '']
    for group in GROUPS:
        rows = []
        for model in display_models(context):
            title = label(context, model)
            if model == selected:
                title = '**' + title + ' †**'
            rows.append([title] + [mean_sd(get(model, m, group)) for m in METRICS])
        lines += ['### ' + GROUP_LABEL[group], '', table(['模型', *METRIC_LABEL.values()], rows), '']
    validation_rows = [[('**' + label(context, model) + ' †**') if model == selected else label(context, model),
                        mean_sd(get(model, 'ndcg50', split='validation'))] for model in display_models(context)]
    lines += ['## 共同验证选择', '', '以五 seed 平均验证 NDCG@50 选择一个共同 variant/stage；每个 seed 不再单独选择不同阶段。每个 checkpoint 的幅度仍按同一个预先声明网格在该 seed 的验证集校准。精确相同时按 variant、stage 名排序；α 校准精确同分优先 0。', '',
              table(['模型', '验证 N@50'], validation_rows), '',
              '种子顺序：`' + ', '.join(map(str, seeds)) + '`。获选模型的逐 seed 测试指标与校准如下：', '']
    rows = []
    for index, seed in enumerate(seeds):
        selection = selections[seed, *selected]
        rows.append([seed] + [fmt(get(selected, m)['per_seed_scores'][index]) for m in METRICS]
                    + [selection['calibration']['alpha'], selection['calibration']['route'],
                       fmt(selection['effective_alpha_times_tanh_gain'], True)])
    lines += [table(['seed', *METRIC_LABEL.values(), 'α', 'route', 'α·tanh(gain)'], rows), '',
              '## 数据支持的范围', '']
    by_kind = {variant['kind']: name for name, variant in context['variants'].items()}
    stage = selected[1]
    text, identifier, shuffled = ((by_kind[kind], stage) for kind in ('text', 'id', 'shuffled'))
    for other, title in ((identifier, 'ID 继续训练/集成对照'), (shuffled, '同结构打乱文本')):
        left, right = get(text, 'ndcg50'), get(other, 'ndcg50')
        differences = [a - b for a, b in zip(left['per_seed_scores'], right['per_seed_scores'])]
        lines += [f"在共同获选的 {STAGE_LABEL[stage]} 阶段，真实文本相对{title}的测试 N@50 均值差为 {fmt(statistics.mean(differences), True)}，{sum(d > 0 for d in differences)}/5 个 seed 为正。这里只描述固定对照中的方向与一致性，显著性和不确定性见下表。", '']
    cold_diff = get(text, 'ndcg50', 'cold')['mean'] - get(identifier, 'ndcg50', 'cold')['mean']
    lines += [f"预先声明的 cold 用户组中，真实文本相对同阶段 ID 对照的测试 N@50 均值差为 {fmt(cold_diff, True)}。全体用户与 cold 用户的结果应共同解释，不能只保留有利用户组。", '']
    router_rows = []
    for kind in ('id', 'text', 'shuffled'):
        name = by_kind[kind]
        left, right = get((name, 'router'), 'ndcg50'), get((name, 'expert'), 'ndcg50')
        differences = [a - b for a, b in zip(left['per_seed_scores'], right['per_seed_scores'])]
        zero_epoch = sum(selections[seed, name, 'router']['selected_epoch'] == 0 for seed in seeds)
        router_rows.append([KIND_LABEL[kind], fmt(statistics.mean(differences), True),
                            f'{sum(d > 0 for d in differences)}/5', f'{zero_epoch}/5'])
    lines += [table(['专家类型', '测试 N@50：relative − global', '正差 seed', 'router 选择 epoch0'], router_rows), '',
              'router 可以保留初始 global 强度，验证选择后的非负验证差有选择规则本身的影响，不能当作泛化证据。选择 epoch0 只表示参数更新未被验证选中；它与下文 gain=0 导致门控梯度退化是不同情况。', '',
              '## 配对差与不确定性', '',
              '差值方向为前者减后者；每个 seed 与自身对应基线配对。CI 是固定这些评估用户、只重采样训练 seed 的描述性 bootstrap 区间，五 seed 时不稳定。exact sign-flip 假设零假设下配对差对称，五 seed 枚举 32 种符号；其非零最小双侧 p 为 **0.0625**，本检验无法达到 0.05。不能将此推广为所有五样本检验的结论，也不能把未显著当成等效。', '',
              f"Holm p 直接使用完整 JSON 比较族（{len(reports['test']['paired_comparisons'])} 项）的校正，未只对展示子集重算；bootstrap 次数 {reports['test']['bootstrap_samples']}、种子 {reports['test']['bootstrap_seed']}。生成器复核配对差、exact p 和完整族 Holm；CI 沿用独立审计结果。", '',
              comparison_table(context, ('all',), METRICS[:2]), '',
              '<details>', '<summary>全部/冷用户/暖用户：R@10、N@10、R@50、N@50 的必要对照全表</summary>', '',
              comparison_table(context, GROUPS, METRICS), '', '</details>', '',
              'ID 是相同优化与选择配方的强继续训练/集成对照，ID 表与文本投影参数量、初始化分数并不严格匹配。Text−Shuffled 更直接检验正确物品—文本对应关系。relative−global 包含额外 router 阶段训练、校准和自适应幅度的整体效果，不能单独证明某个诊断特征的因果作用。', '',
              '## 校准、参数与训练预算', '',
              f"专家最多 {cfg['epochs']} 轮、至少 {cfg['min_epochs']} 轮；每 {cfg['eval_interval']} 轮验证，连续 {cfg['patience']} 次无改进后早停，epoch 0 可选。router 最多 {cfg['router_epochs']} 轮、至少 {cfg['router_min_epochs']} 轮，学习率 {cfg['router_lr']}。batch={cfg['batch_size']}，幅度网格={cfg['alpha_grid']}，expert routes={cfg['routes']}。三种专家共用下表配方，未按测试结果单独调参。", '']
    if context.get('single'):
        anchor_val = get(SINGLE, 'ndcg50', split='validation')['mean']
        method_val = get(selected, 'ndcg50', split='validation')['mean']
        baseline_lines = ['## 附加单分支 CF-CE 对照', '',
            f'单分支 CF-CE 的平均验证 N@50 为 {fmt(anchor_val)}，原七行验证所选模型为 {fmt(method_val)}，后者减前者为 {fmt(method_val - anchor_val, True)}。单分支基线有自己覆盖五种子的验证 epoch 锁；它没有被悄悄并入原七行选择，更没有根据测试重选原方法。', '',
            '该对照保持本地 CF 编码器、起始强 CF、分类尺度16、full-train 掩码池及共同优化/早停预算，只继续训练单分支。它不含独立分支、冻结 CF 分数残差或专家辅助损失，因此与残差模型的差值仍是整套配方比较，不是严格融合单因素实验；也不是官方 SASRec/RecBole CE 的完整复现。', '',
            f"以下附加比较固定为原七行共同验证所选模型减单分支 CF-CE；全部4用户组×4指标，共16项，不按测试结果挑选。差值CI枚举全部 5^5＝3125 种有序 seed-bootstrap 重采样，分位数作线性插值；exact p 仍枚举32种符号。这里 Holm 按附加16项与原{len(reports['test']['paired_comparisons'])}项组成的合并族校正；上表核心比较保留其原审计族的 Holm。两类 CI 均仅描述固定用户条件下的 seed 不确定性。", '',
            single_comparison_table(context), '']
        insertion = lines.index('## 校准、参数与训练预算')
        lines[insertion:insertion] = baseline_lines
    rows = []
    for name, variant in cfg_variant_order(context):
        sample = selections[seeds[0], name, 'expert']
        router = selections[seeds[0], name, 'router']
        parameter_keys = ('expert_parameter_count', 'item_adapter_parameter_count', 'expert_encoder_parameter_count')
        for seed in seeds:
            require(all(selections[seed, name, 'expert'][key] == sample[key] for key in parameter_keys), '跨 seed 参数量不一致')
            require(selections[seed, name, 'router']['active_trainable_parameter_count'] == 5, 'router 应仅训练五参数')
        rows.append([KIND_LABEL[variant['kind']], variant['objective'], variant['score_scale'], variant['lr'],
                     variant['auxiliary_weight'], variant['alignment_weight'], sample['item_adapter_parameter_count'],
                     sample['expert_encoder_parameter_count'], sample['active_trainable_parameter_count'],
                     router['active_trainable_parameter_count']])
    lines += [table(['专家', '目标', '专家尺度', 'lr', '辅助权重', '身份权重', '物品适配参数', '编码器参数', '专家阶段训练参数', 'router训练参数'], rows), '',
              '下表逐一披露全部 30 个被选模型。完成轮数/优化步数属于该阶段；router 从对应验证选出的 global checkpoint 继续，不能把其预算当作零成本。验证检查数包含 epoch 0，“候选/次”包含完整 α 网格。', '']
    rows = []
    for seed in seeds:
        for model in context['models'][1:]:
            row = selections[seed, *model]
            rows.append([seed, label(context, model), row['selected_epoch'], row['epochs_completed'],
                         row['optimizer_steps'], row['validation_checks_including_epoch0'], row['candidates_per_validation'],
                         row['calibration']['alpha'], row['calibration']['route'], fmt(row['tanh_gain'], True),
                         fmt(row['effective_alpha_times_tanh_gain'], True)])
    lines += [table(['seed', '模型', '选择轮', '完成轮', '优化步', '验证检查', '候选/次', 'α', 'route', 'tanh(gain)', 'α·tanh(gain)'], rows), '']
    if context.get('single'):
        rows = []
        for seed in seeds:
            row = context['single']['selections'][seed]
            rows.append([seed, row['selected_epoch'], row['epochs_completed'], row['optimizer_steps'],
                         fmt(row['validation_ndcg50']), row.get('active_trainable_parameter_count', '见基线训练记录')])
        lines += ['单分支 CF-CE 的独立 epoch 选择与实际训练预算：', '',
                  table(['seed', '选择轮', '完成轮', '优化步', '验证 N@50', '训练参数'], rows), '']
    degenerate = [f"{seed}/{name}" for seed in seeds for name in context['variants']
                  if abs(selections[seed, name, 'expert']['tanh_gain']) < 1e-15]
    if degenerate:
        lines += ['零 gain 的 global checkpoint：`' + ', '.join(degenerate) + '`。router 阶段冻结 gain，因此这些任务的残差/门控梯度退化为零，不能据此判断相对门控无效。', '']
    zero = [f"{seed}/{name}/{stage}" for (seed, name, stage), row in selections.items()
            if row['calibration']['alpha'] == 0]
    lines += [f'最终校准选择 α=0 的模型数量：{len(zero)}/30。α=0 精确恢复冻结 CF；非零残差被验证选中后仍可能在测试退化。汇总没有完整墙钟时间或峰值显存，不能据此声称运行速度/显存优势。', '',
              '## 改造依据、实现与论文同步', '',
              '- [PAD（SIGIR 2025）](https://arxiv.org/abs/2412.04107)：保留模态专属专家、最后阶段去掉 MMD，启发独立 CF 能力和分数融合；本轮未复现其完整三专家。',
              '- [LLM-ESR（NeurIPS 2024）](https://arxiv.org/abs/2405.20646)：双视图拼接点积等价于分数相加；原模型仍共享序列编码器，本轮独立克隆是自己的组合适配。',
              '- [UniSRec（KDD 2022）](https://arxiv.org/abs/2206.05941)：去中心轻量适配和温度分类目标提供设计依据；本轮 train-seen affine ridge 并非其原算法。',
              '- [RLMRec（WWW 2024）](https://arxiv.org/abs/2310.15950)：同 ID 跨视图对应对齐，不是匿名边际 MMD。身份对齐仅为可选消融；本次实际权重以配置表为准。',
              '- [IISAN（SIGIR 2024）](https://arxiv.org/abs/2404.02059)及[IISAN-Versa（TKDE 2025）](https://doi.org/10.1109/TKDE.2025.3608071)：借鉴缓存冻结特征和旁路适配原则；本轮只有最终文本向量，不是其多层图文适配完整复现。', '',
              f"实际配方说明：{cfg['experiment_note']}", '',
              'CF anchor 冻结并固定为 eval；专家编码器从 anchor 克隆后独立训练。文本适配器由 train-seen centered affine ridge 初始化，最终向量各自归一化，拼接后不再次归一化。分数为 `s_cf + α·tanh(gain)·user_gate·item_gate·s_expert`；专家独立辅助监督使零 gain 起点仍有训练信号。', '',
              '本轮 CE 使用全目录基础池，排除该用户 full-train 其他正例、恢复当前监督正例、排除 PAD，分块并在反向重计算。它不是标准 RecBole CE，也不是严格 prefix-only 负池；没有使用 validation/test 正例。若 pilot 的 CE 与 BPR 同时使用不同分数尺度，应写成配方比较，不能宣称纯 CE 单因素效果。', '',
              'Relative gate 使用最近已观察转移的 CF−expert probe-loss 均值/标准差、当前无目标熵差与 log-prefix；当前待预测目标和未来正例不进入 gate。前向时间边界成立，但专家由完整训练集拟合，历史误差仍 in-sample；不能宣称校准过的样本外可靠性或因果估计。', '',
              '论文须重写共享融合公式、冻结/训练参数和阶段、diagnostics 定义及信息边界；若采用可选同 ID InfoNCE，应移除旧 marginal-MMD/characteristic-kernel 理论对新模型的直接解释。实验表应同步真实数据统计、时间划分、全目录多正例指标、用户 cold/warm、CE 负池和验证预算。新方法只能按真实完成的对照强度陈述，不能延续旧消融结论。', '',
              '## 输入与审计链', '']
    identity = reports['test']['common_provenance']['identity']
    lines += [f"固定数据：{len(identity['user_order'])} 用户、{identity['items']} 个目录物品。完整 JSON 还保留 middle 用户组、Hit 指标和其他配对比较；本报告按请求展示 R/N 与 all/cold/warm。", '',
              '- 验证汇总：' + link(context['paths']['validation']) + '；SHA256：`' + sha256(context['paths']['validation']) + '`。',
              '- 测试汇总：' + link(context['paths']['test']) + '；SHA256：`' + sha256(context['paths']['test']) + '`。',
              '- 共同验证锁：' + link(context['lock_path']) + '；SHA256：`' + reports['test']['selection_lock_sha256'] + '`。',
              '- 冻结记录指纹：`' + reports['test']['selection_lock_fingerprint'] + '`。', '',
              '生成器只读取上述完成文件，复核聚合、配对统计和共同锁，不代替前置 checkpoint/逐用户排名审计；没有训练或测试模型调用。完整 provenance 与原始产物校验值见 JSON。', '',
              '报告生成器 SHA256：`' + sha256(Path(__file__).resolve()) + '`。', '']
    if context.get('single'):
        single = context['single']
        lines += ['附加基线审计文件：', '',
            '- 验证汇总：' + link(single['paths']['validation']) + '；SHA256：`' + sha256(single['paths']['validation']) + '`。',
            '- 测试汇总：' + link(single['paths']['test']) + '；SHA256：`' + sha256(single['paths']['test']) + '`。',
            '- 五种子 epoch 验证锁：' + link(single['lock_path']) + '；SHA256：`' + sha256(single['lock_path']) + '`。', '']
    return '\n'.join(lines)


def cfg_variant_order(context):
    return [(name, context['variants'][name]) for name, stage in context['models'][1:] if stage == 'expert']


def render_latex(context):
    lines = [r'% Generated only from audited combined_validation.json / combined_test.json.',
             r'% Requires booktabs. Values are mean +/- sample SD across five seeds.',
             r'% User groups are stacked to avoid shrinking thirteen wide mean/SD columns.',
             r'\begin{table*}[t]', r'\centering',
             r'\caption{ML-1M results (mean $\pm$ sample standard deviation over five training seeds). '
             r'$\dagger$ marks the model chosen by common mean validation NDCG@50. '
             r'Cold/warm refer to user training-history groups, not item frequency.}',
             r'\label{tab:ml1m_research_v3}', r'\begin{tabular}{lcccc}', r'\toprule',
             r'Model & R@10 & N@10 & R@50 & N@50 \\']
    for group in GROUPS:
        group_title = {'all': 'Overall', 'cold': 'Cold users', 'warm': 'Warm users'}[group]
        lines += [r'\midrule', r'\multicolumn{5}{c}{' + group_title + r'} \\', r'\midrule']
        for model in display_models(context):
            if model == ANCHOR:
                title = 'Strong CF anchor'
            elif model == SINGLE:
                title = 'Single-branch CF-CE (local)'
            else:
                kind = context['variants'][model[0]]['kind']
                title = {'id': 'ID', 'text': 'Text', 'shuffled': 'Shuffled text'}[kind] + ' / ' + STAGE_LABEL[model[1]]
            if model == context['chosen']:
                title = r'\textbf{' + title + r'}$^{\dagger}$'
            values = ['$' + mean_sd(context['metrics']['test'][*model, group, metric], True) + '$'
                      for metric in METRICS]
            lines.append(title + ' & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    return '\n'.join(lines)


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='\n',
                                         dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replica-root', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, required=True)
    parser.add_argument('--repo-md', type=Path, required=True)
    parser.add_argument('--table-tex', type=Path)
    parser.add_argument('--single-ce-report', type=Path, help='Audited five-seed single CF-CE test aggregate')
    parser.add_argument('--single-ce-validation-report', type=Path, help='Its validation aggregate; default: same directory/combined_validation.json')
    parser.add_argument('--report-date', help='Report date in Asia/Shanghai, YYYY-MM-DD; default: current UTC+8 date')
    args = parser.parse_args(argv)
    context = load_audited(args.replica_root)
    require(args.single_ce_report is not None or args.single_ce_validation_report is None,
            '--single-ce-validation-report 需要同时提供 --single-ce-report')
    if args.single_ce_report is not None:
        load_single_audited(context, args.single_ce_report, args.single_ce_validation_report)
    if args.report_date is not None:
        require(datetime.strptime(args.report_date, '%Y-%m-%d').strftime('%Y-%m-%d') == args.report_date,
                '报告日期必须采用 YYYY-MM-DD')
        context['report_date'] = args.report_date
    outputs = [args.output_md.resolve(), args.repo_md.resolve()]
    if args.table_tex is not None:
        outputs.append(args.table_tex.resolve())
    require(len(set(outputs)) == len(outputs), '各输出路径必须不同')
    require(all(p.suffix.lower() == '.md' for p in outputs[:2])
            and (len(outputs) == 2 or outputs[2].suffix.lower() == '.tex'), '输出扩展名必须是 .md/.tex')
    inputs = set(context['paths'].values()) | {context['lock_path'], Path(__file__).resolve()}
    if context.get('single'):
        inputs.update(context['single']['paths'].values())
        inputs.add(context['single']['lock_path'])
    require(not set(outputs) & inputs, '输出不得覆盖输入或生成器')
    markdown, latex = render_markdown(context), render_latex(context)
    # Rendering and all validation finish before the first output is touched.
    for path in outputs[:2]:
        write_atomic(path, markdown)
    if len(outputs) == 3:
        write_atomic(outputs[2], latex)
    print('已由完整审计结果生成中文报告' + ('及 LaTeX 表格' if len(outputs) == 3 else '') + '：')
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()
