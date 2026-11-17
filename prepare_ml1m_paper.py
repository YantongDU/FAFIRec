"""Prepare MovieLens-1M with rating>=4 and per-user chronological 80/10/10.

The manuscript specifies the ratio but does not release split indices. This
entry point declares stable timestamp ordering and integer rounding; it does
not force the historical manuscript's reported user count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value, label):
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f'{label} must be a positive integer') from None
    if result < 1:
        raise ValueError(f'{label} must be a positive integer')
    return str(result)


def _group_manifest(records):
    order = sorted(range(len(records)), key=lambda i: (len(records[i]['train']), i))
    tail = min(max(1, int(len(records) * .1)), len(records) // 2)
    cold, warm = order[:tail], order[-tail:] if tail else []
    middle = order[tail:len(order)-tail] if tail else order
    result = {'rule': 'stable full-training-count rank; bottom/top 10%; ascending user ID breaks ties',
              'tail_users': tail, 'overlap': 0}
    for group, indices in (('cold', cold), ('middle', middle), ('warm', warm)):
        values = [len(records[i]['train']) for i in indices]
        result[group] = {'users': len(indices),
                         'training_count_range': [min(values), max(values)] if values else None}
    return result


def prepare(input_dir, output_dir, ordering_confirmed_by_user=False):
    """Create a new version-2 bundle; existing nonempty outputs are protected."""
    source, output = Path(input_dir), Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Output must be a new or empty directory')
    ratings_path, movies_path = source / 'ratings.dat', source / 'movies.dat'
    for path in (ratings_path, movies_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    items, item_mapping = [], {}
    with movies_path.open(encoding='latin-1') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                raw, remainder = line.rstrip('\r\n').split('::', 1)
                title, genres = remainder.rsplit('::', 1)
            except ValueError:
                raise ValueError(f'movies.dat:{line_number}: expected ID::title::genres') from None
            raw = _identifier(raw, 'MovieID')
            if raw in item_mapping:
                raise ValueError(f'Duplicate metadata MovieID {raw}')
            if not title.strip():
                raise ValueError('Movie titles must be nonempty')
            item_mapping[raw] = len(items) + 1
            items.append({'id': len(items) + 1, 'raw_id': raw, 'title': title,
                          'genres': genres.split('|') if genres else [],
                          'text': f'{title} | {genres}'})
    if not items:
        raise ValueError('Movie metadata catalog is empty')

    events, source_users = {}, set()
    counts = {'source_events': 0, 'rating_filtered_events': 0,
              'positive_events': 0, 'source_users': 0,
              'users_dropped_no_positive': 0, 'users_dropped_short': 0,
              'users_dropped_no_training_negative': 0, 'users': 0,
              'interactions': 0, 'training_events': 0, 'validation_events': 0,
              'test_events': 0, 'positive_events_dropped_with_users': 0}
    with ratings_path.open(encoding='ascii') as stream:
        for row_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            parts = line.strip().split('::')
            if len(parts) != 4:
                raise ValueError(f'ratings.dat:{row_number}: expected user::movie::rating::timestamp')
            user, raw = _identifier(parts[0], 'UserID'), _identifier(parts[1], 'MovieID')
            try:
                rating, timestamp = int(parts[2]), int(parts[3])
            except ValueError:
                raise ValueError(f'ratings.dat:{row_number}: rating and timestamp must be integers') from None
            if not 1 <= rating <= 5:
                raise ValueError('MovieLens ratings must lie in 1..5')
            if raw not in item_mapping:
                raise ValueError(f'Observed MovieID {raw} is absent from metadata')
            source_users.add(user)
            counts['source_events'] += 1
            if rating < 4:
                counts['rating_filtered_events'] += 1
                continue
            counts['positive_events'] += 1
            events.setdefault(user, []).append((timestamp, row_number, item_mapping[raw]))
    counts['source_users'] = len(source_users)
    counts['users_dropped_no_positive'] = len(source_users - set(events))
    records = []
    for user in sorted(events, key=int):
        sequence = [item for _, _, item in sorted(events[user])]
        total = len(sequence)
        n_train, n_validation = (8 * total) // 10, total // 10
        n_test = total - n_train - n_validation
        if n_train < 2 or n_validation < 1 or n_test < 1:
            counts['users_dropped_short'] += 1
            counts['positive_events_dropped_with_users'] += total
            continue
        train = sequence[:n_train]
        if len(set(train)) == len(items):
            counts['users_dropped_no_training_negative'] += 1
            counts['positive_events_dropped_with_users'] += total
            continue
        validation = sequence[n_train:n_train+n_validation]
        test = sequence[n_train+n_validation:]
        records.append({'user': user, 'train': train, 'validation': validation, 'test': test})
        counts['training_events'] += len(train)
        counts['validation_events'] += len(validation)
        counts['test_events'] += len(test)
        counts['interactions'] += total
    counts['users'] = len(records)
    if not records:
        raise ValueError('No eligible users after positive filtering and 80/10/10 partition requirements')

    users_path = source / 'users.dat'
    source_files = [ratings_path, movies_path] + ([users_path] if users_path.is_file() else [])
    manifest = {
        'format_version': 2, 'dataset': 'movielens1m',
        'split': {
            'strategy': 'per-user chronological 80/10/10 event partitions',
            'ratios': [0.8, 0.1, 0.1],
            'rounding': 'n_train=(8*n)//10; n_validation=n//10; n_test=n-n_train-n_validation',
            'ordering': 'timestamp ascending, then original ratings.dat row order',
            'ordering_choice_source': ('user-confirmed chronological split' if ordering_confirmed_by_user
                                       else 'explicit chronological execution convention'),
            'chronology_verified': True, 'user_order': 'ascending numeric UserID',
            'validation_history': 'complete training segment only',
            'test_history': 'complete training plus complete validation segments',
            'targets': 'complete validation/test event segments; never only the final item',
            'evaluation_relevance': 'unique item IDs within the held-out segment',
            'diagnostics': 'final training transition only',
            'repeated_events': 'retained; relevance item identities deduplicated only at evaluation',
        },
        'filtering': {'minimum_rating_inclusive': 4, 'criterion': 'rating >= 4 (rating > 3)',
                      'min_training_events': 2, 'min_validation_events': 1, 'min_test_events': 1,
                      'requires_training_unseen_negative': True,
                      'iterative_k_core': False, 'deduplicate_events': False,
                      'historical_user_count_forced': False},
        'catalog': {'items': len(items), 'ordering': 'movies.dat metadata row order',
                    'scope': 'all metadata items, including items without retained training feedback',
                    'text': 'static movie title and genres; no user behavior summaries'},
        'user_profiles': {'source_available': users_path.is_file(),
                          'source': 'users.dat' if users_path.is_file() else None,
                          'consumed_by_revised_model': False,
                          'note': 'Static demographics availability recorded; profile values are not read into model features.'},
        'grouping': _group_manifest(records), 'counts': counts,
        'sources': [{'name': path.name, 'sha256': sha256(path), 'bytes': path.stat().st_size}
                    for path in source_files],
        'protocol_status': 'manuscript ratios with declared chronological ordering/rounding; not historical-table reproduction',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ml1m-ratio-', dir=output.parent) as temporary:
        staging = Path(temporary)
        for name, values in (('items.jsonl', items), ('interactions.jsonl', records)):
            with (staging / name).open('w', encoding='utf-8', newline='\n') as stream:
                for value in values:
                    stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        manifest['files'] = {name: sha256(staging / name) for name in ('items.jsonl', 'interactions.jsonl')}
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        output.mkdir(parents=True, exist_ok=True)
        for name in ('items.jsonl', 'interactions.jsonl', 'manifest.json'):
            with (output / name).open('xb') as destination:
                destination.write((staging / name).read_bytes())
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Raw ml-1m directory containing ratings.dat and movies.dat')
    parser.add_argument('--output', required=True, help='New or empty prepared-data directory')
    parser.add_argument('--ordering-confirmed-by-user', action='store_true',
                        help='Record explicit user confirmation of the chronological split convention')
    args = parser.parse_args(argv)
    manifest = prepare(args.input, args.output, args.ordering_confirmed_by_user)
    print(json.dumps({'output': str(Path(args.output).resolve()), 'counts': manifest['counts'],
                      'split': manifest['split'], 'catalog': manifest['catalog']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
