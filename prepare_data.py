"""Create auditable next-item bundles; never infer chronology from ratings alone.

This program uses only Python's standard library. A temporary SQLite spool keeps
large event collections on disk while sorting one user's history at a time.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import math
import sqlite3
import tempfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _timestamp(value):
    if value is None or not str(value).strip():
        return None
    value = str(value).strip()
    try:
        number = float(value)
    except ValueError:
        try:
            date = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError as error:
            raise ValueError(f'Invalid timestamp {value!r}; use Unix numeric or ISO-8601 time') from error
        number = date.replace(tzinfo=date.tzinfo or dt.timezone.utc).timestamp()
    if not math.isfinite(number):
        raise ValueError('Timestamps must be finite')
    return number


def _rating(value):
    if value is None or not str(value).strip():
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Ratings must be finite')
    return result


def _metadata_csv(path, source):
    """Yield external IDs and static metadata, without behavioral summaries."""
    if source == 'book-crossing':
        with Path(path).open(encoding='latin-1', newline='') as stream:
            for row in csv.DictReader(stream, delimiter=';'):
                yield str(row['ISBN']), ' | '.join(str(row.get(k, '')).strip()
                                                  for k in ('Book-Title', 'Book-Author', 'Publisher')
                                                  if str(row.get(k, '')).strip())
    else:
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                identifier = row.get('item', row.get('raw_id', row.get('asin')))
                if identifier is None:
                    raise ValueError('Metadata CSV needs item (or raw_id/asin) and text (or title) columns')
                yield str(identifier), str(row.get('text', row.get('title', '')))


def _metadata(path, source):
    if path is None:
        return
    path = Path(path)
    if source == 'movielens1m' and path.suffix.lower() == '.dat':
        with path.open(encoding='latin-1') as stream:
            for line in stream:
                if not line.strip():
                    continue
                raw_id, title, genres = line.rstrip('\r\n').split('::', 2)
                yield raw_id, f'{title} | {genres}'
    elif source == 'netflix' and path.name.lower().startswith('movie_titles'):
        with path.open(encoding='latin-1', newline='') as stream:
            for row in csv.reader(stream):
                if len(row) < 3:
                    raise ValueError('Netflix movie_titles file needs ID, year, title')
                yield row[0], f'{",".join(row[2:])} ({row[1]})'
    else:
        yield from _metadata_csv(path, source)


def _choose_file(root, alternatives):
    root = Path(root)
    if root.is_file():
        return root
    for name in alternatives:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Expected one of {alternatives} under {root}')


def _sources(args):
    source = args.source
    root = Path(args.input)
    metadata = Path(args.metadata or args.items) if args.metadata or args.items else None
    if source == 'prime-pantry':
        event_files = [_choose_file(root, ['Prime_Pantry_cleaned_history.csv'])]
        if metadata is None:
            metadata = event_files[0].parent / 'metadata_Prime_Pantry.csv'
    elif source == 'movielens1m':
        event_files = [_choose_file(root, ['ratings.dat'])]
        if metadata is None:
            metadata = event_files[0].parent / 'movies.dat'
    elif source == 'book-crossing':
        event_files = [_choose_file(root, ['BX-Book-Ratings.csv'])]
        default = event_files[0].parent / 'BX-Books.csv'
        if metadata is None and default.is_file():
            metadata = default
    elif source == 'netflix':
        if root.is_file():
            event_files = [root]
        else:
            directory = root / 'training_set' if (root / 'training_set').is_dir() else root
            event_files = sorted(directory.glob('mv_*.txt'))
            if not event_files:
                event_files = sorted(directory.glob('combined_data_*.txt'))
            if not event_files:
                raise FileNotFoundError('Netflix input needs training_set/mv_*.txt or combined_data_*.txt')
        default = (root if root.is_dir() else root.parent) / 'movie_titles.txt'
        if metadata is None and default.is_file():
            metadata = default
    else:
        event_files = [_choose_file(root, ['interactions.csv'])]
    if metadata is not None and not metadata.is_file():
        raise FileNotFoundError(metadata)
    return event_files, metadata


def _events(args, files):
    """Yield user, raw item ID, timestamp-or-None, rating-or-None."""
    source = args.source
    pantry_users = set()
    for path in files:
        if source == 'prime-pantry':
            with path.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream):
                    if row['user'] in pantry_users:
                        raise ValueError('Duplicate complete user history in Prime Pantry input')
                    pantry_users.add(row['user'])
                    history = ast.literal_eval(row['history'])
                    if not isinstance(history, list) or any(not isinstance(item, str) for item in history):
                        raise ValueError('Pantry history must be a literal list of string item identifiers')
                    for item in history:
                        yield str(row['user']), item, None, None
        elif source == 'movielens1m':
            with path.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    if line.strip():
                        user, item, rating, stamp = line.rstrip('\r\n').split('::')
                        yield user, item, _timestamp(stamp), _rating(rating)
        elif source == 'netflix':
            item = None
            with path.open(encoding='utf-8-sig', newline='') as stream:
                for line in stream:
                    value = line.strip()
                    if not value:
                        continue
                    if value.endswith(':'):
                        item = value[:-1]
                    else:
                        if item is None:
                            raise ValueError('Netflix rating appears before its movie-ID header')
                        user, rating, stamp = next(csv.reader([value]))
                        yield user, item, _timestamp(stamp), _rating(rating)
        elif source == 'book-crossing':
            with path.open(encoding='latin-1', newline='') as stream:
                for row in csv.DictReader(stream, delimiter=';'):
                    yield str(row['User-ID']), str(row['ISBN']), None, _rating(row['Book-Rating'])
        else:
            with path.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream):
                    if args.user_column not in row or args.item_column not in row:
                        raise ValueError('Interaction CSV needs the configured user and item columns')
                    yield (str(row[args.user_column]), str(row[args.item_column]),
                           _timestamp(row.get(args.timestamp_column)), _rating(row.get(args.rating_column)))


def prepare(args):
    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Refusing to overwrite a nonempty output: {output}')
    if args.min_rating is not None and not math.isfinite(args.min_rating):
        raise ValueError('--min-rating must be finite')
    if args.source in ('prime-pantry', 'book-crossing') and not args.allow_unverified_order:
        raise ValueError(f'{args.source} has no timestamps; explicitly pass --allow-unverified-order')
    event_files, metadata_path = _sources(args)
    catalog, mapping = [], {}
    def add_item(raw, text=''):
        if not raw or raw == '<PAD>':
            raise ValueError('Empty or reserved raw item ID')
        if raw in mapping:
            raise ValueError(f'Duplicate item metadata ID: {raw}')
        mapping[raw] = len(catalog) + 1
        catalog.append({'id': len(catalog) + 1, 'raw_id': raw, 'text': text})
    if metadata_path is not None:
        for raw, text in _metadata(metadata_path, args.source):
            add_item(raw, text)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {'source_events': 0, 'rating_filtered_events': 0, 'users_dropped_short': 0,
              'users_dropped_no_training_negative': 0, 'users': 0, 'interactions': 0}
    any_missing_time = False
    with tempfile.TemporaryDirectory(prefix='.fafirec-prepare-', dir=output.parent) as directory:
        staging = Path(directory)
        database = sqlite3.connect(staging / 'events.sqlite')
        try:
            database.execute('CREATE TABLE events (user TEXT NOT NULL, item INTEGER NOT NULL, stamp REAL, ordinal INTEGER NOT NULL)')
            pending = []
            for ordinal, (user, raw, stamp, rating) in enumerate(_events(args, event_files)):
                counts['source_events'] += 1
                if not user or not raw:
                    raise ValueError('Empty user or item identifier')
                if args.min_rating is not None:
                    if rating is None:
                        raise ValueError('--min-rating requires an observed rating on every source event')
                    if rating < args.min_rating:
                        counts['rating_filtered_events'] += 1
                        continue
                if raw not in mapping:
                    if metadata_path is not None:
                        raise ValueError(f'Observed item {raw!r} is absent from the supplied metadata catalog')
                    add_item(raw)
                if stamp is None:
                    any_missing_time = True
                pending.append((user, mapping[raw], stamp, ordinal))
                if len(pending) >= 10000:
                    database.executemany('INSERT INTO events VALUES (?, ?, ?, ?)', pending)
                    pending.clear()
            if pending:
                database.executemany('INSERT INTO events VALUES (?, ?, ?, ?)', pending)
            database.commit()
            if any_missing_time and not args.allow_unverified_order:
                raise ValueError('Missing timestamps; explicitly use --allow-unverified-order to retain file order')
            if not catalog:
                raise ValueError('No catalog items remain')
            # With any unknown timestamp, retain input order for *every* user;
            # never mix numeric chronology with a fabricated timestamp for gaps.
            order = 'user, ordinal' if any_missing_time else 'user, stamp, ordinal'
            database.execute(f'CREATE INDEX event_order ON events ({order})')
            def write_user(stream, user, history):
                if len(history) < 4:
                    counts['users_dropped_short'] += 1
                    return
                train = history[:-2]
                if len(set(train)) >= len(catalog):
                    counts['users_dropped_no_training_negative'] += 1
                    return
                record = {'user': user, 'train': train, 'validation': history[-2], 'test': history[-1]}
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                counts['users'] += 1
                counts['interactions'] += len(history)
            with (staging / 'interactions.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
                last_user, history = None, []
                for user, item in database.execute(f'SELECT user, item FROM events ORDER BY {order}'):
                    if last_user is not None and user != last_user:
                        write_user(stream, last_user, history)
                        history = []
                    last_user = user
                    history.append(item)
                if last_user is not None:
                    write_user(stream, last_user, history)
            if counts['users'] == 0:
                raise ValueError('No eligible users after filtering and diagnostic/negative requirements')
        finally:
            database.close()
        with (staging / 'items.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            for item in catalog:
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
        chronology = not any_missing_time
        manifest = {
            'format_version': 1, 'dataset': args.source,
            'split': {'strategy': 'leave-two-out after filtering; at least four observations per retained user',
                      'ordering': 'timestamp ascending; stable file-order ties' if chronology else 'provided file/history order; chronology UNVERIFIED',
                      'chronology_verified': chronology,
                      'timestamp_resolution': 'as supplied; tied timestamps follow source file order',
                      'validation_history': 'training only', 'test_history': 'training plus validation event',
                      'diagnostics': 'final training transition only; no validation/test targets',
                      'repeated_events': 'retained; repeated target identity may occur earlier'},
            'filtering': {'minimum_rating_inclusive': args.min_rating,
                          'rating_zero': 'retained when threshold permits; no implicit/explicit reinterpretation',
                          'min_training_events': 2, 'min_total_events': 4,
                          'requires_training_unseen_negative': True},
            'catalog': {'ordering': 'metadata file order' if metadata_path is not None else 'first surviving event appearance',
                        'scope': 'all supplied metadata rows, including zero-training-feedback items' if metadata_path is not None else 'all surviving observed item identities',
                        'items': len(catalog),
                        'text': 'supplied static metadata only; no behavioral user summaries',
                        'empty_text_items': sum(not item['text'].strip() for item in catalog)},
            'counts': counts,
            'sources': [{'name': str(path.relative_to(Path(args.input))) if Path(args.input).is_dir() and path.is_relative_to(Path(args.input)) else path.name,
                         'sha256': sha256(path), 'bytes': path.stat().st_size}
                        for path in event_files + ([metadata_path] if metadata_path is not None else [])],
            'files': {name: sha256(staging / name) for name in ('items.jsonl', 'interactions.jsonl')},
            'protocol_status': 'new explicit protocol; not a claim of reproducing historical paper tables',
        }
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        output.mkdir(parents=True, exist_ok=True)
        for name in ('items.jsonl', 'interactions.jsonl', 'manifest.json'):
            # Exclusive creation also prevents concurrent preparations clobbering
            # one another after the initial empty-directory check.
            with (output / name).open('xb') as destination, (staging / name).open('rb') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    destination.write(chunk)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, choices=['prime-pantry', 'movielens1m', 'netflix', 'book-crossing', 'csv'])
    parser.add_argument('--input', required=True, help='Source directory or ratings/history file')
    parser.add_argument('--output', required=True, help='New or empty prepared-data directory')
    parser.add_argument('--metadata', '--items', dest='metadata', help='Optional metadata file; source-specific defaults are used when present')
    parser.set_defaults(items=None)
    parser.add_argument('--min-rating', type=float, default=None, help='Inclusive threshold; omitted means retain observed events of every rating')
    parser.add_argument('--allow-unverified-order', action='store_true', help='Allow source order where timestamps are absent; recorded as UNVERIFIED')
    parser.add_argument('--user-column', default='user')
    parser.add_argument('--item-column', default='item')
    parser.add_argument('--timestamp-column', default='timestamp')
    parser.add_argument('--rating-column', default='rating')
    return parser


def main():
    args = build_parser().parse_args()
    manifest = prepare(args)
    print(json.dumps({'output': str(Path(args.output).resolve()),
                      'dataset': manifest['dataset'], 'counts': manifest['counts'],
                      'split': manifest['split']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
