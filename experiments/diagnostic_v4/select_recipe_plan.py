"""Freeze validation-selected recipes plus matched diagnostic controls."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import ace2026_experiment as ace
from experiments.diagnostic_v4 import fusion_experiment as fusion
from fafirec.runner import digest, file_digest, json_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=Path('runs/diagnostic_v4_fusion_pilot'))
    parser.add_argument('--output', type=Path, default=Path('configs/diagnostic_v4_selected.json'))
    args = parser.parse_args()
    output, pilot = args.output.resolve(), args.pilot.resolve()
    ace.require(not output.exists(), 'Recipe plan is immutable')
    selection = ace.read(pilot / 'selection.json')
    ace.require(selection['source_hashes'] == fusion.source_identity()
        and selection['selection_split'] == 'validation' and selection['test_evaluated'] is False,
        'Pilot source/selection identity differs')
    for path, checksum in selection['artifact_sha256'].items():
        ace.require(file_digest(path) == checksum, 'Pilot artifact changed: ' + path)
    candidates = selection['candidates']
    def best(predicate):
        return sorted([r for r in candidates if predicate(r)], key=lambda r: (-r['score'], r['variant']))[0]
    global_row = best(lambda r: r['mode'] == 'global' and r['alignment_weight'] == 0.)
    diagnostic = best(lambda r: r['mode'] in ('diagnostic', 'candidate') and r['alignment_weight'] == 0.)
    uniform = best(lambda r: r['mode'] == 'global' and r['alignment_weight'] > 0.)
    requested = [
        ('ace_coverage', best(lambda r: r['mode'] == 'global' and r['alpha'] == 0. and r['beta'] == 0. and r['alignment_weight'] == 0.)),
        ('popularity_only', best(lambda r: r['mode'] == 'global' and r['alpha'] == 0. and r['alignment_weight'] == 0.)),
        ('global_fusion', global_row), ('diagnostic_fusion', diagnostic), ('uniform_alignment', uniform)]
    # Exact hyperparameter matches isolate changing the controller or alignment.
    for name, mode, l2, mu in (
        ('uniform_without_alignment', 'global', None, 0.),
        ('diagnostic_matched', diagnostic['mode'], diagnostic['l2'], 0.),
        ('diagnostic_alignment', diagnostic['mode'], diagnostic['l2'], uniform['alignment_weight'])):
        requested.append((name, best(lambda r: r['mode'] == mode and r['l2'] == l2
            and r['alignment_weight'] == mu and r['alpha'] == uniform['alpha'] and r['beta'] == uniform['beta'])))
    recipes, aliases, seen = [], {}, {}
    for name, row in requested:
        parameters = {k: row[k] for k in ('mode', 'l2', 'alpha', 'beta', 'alignment_weight')}
        key = digest(parameters)
        if key in seen:
            aliases[name] = seen[key]
        else:
            seen[key] = name
            recipes.append({'name': name, **parameters, 'pilot_variant': row['variant'],
                'pilot_validation_ndcg50': row['score']})
    plan = {'protocol': 'fafirec_v4_recipe_plan_v1', 'seed_ids': [2025, 2026, 2027, 2028, 2029],
        'recipes': recipes, 'aliases': aliases, 'source_hashes': fusion.source_identity(),
        'selection_split': 'validation', 'new_test_used': False, 'historical_test_known': True,
        'selection_rule': 'One common recipe with highest five-seed mean validation NDCG@50; name breaks exact ties.',
        'pilot_recipe_selection_seed': 2025, 'pilot_candidate_count': selection['candidate_count'],
        'artifact_sha256': {str(pilot / 'selection.json'): file_digest(pilot / 'selection.json'),
            str(pilot / 'plan.json'): file_digest(pilot / 'plan.json'), str(Path(__file__).resolve()): file_digest(__file__)}}
    plan['fingerprint'] = digest(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(output, plan)
    print('Frozen recipes:', ', '.join(r['name'] for r in recipes))
    for row in recipes:
        print(row['name'], row['pilot_validation_ndcg50'])


if __name__ == '__main__':
    main()
