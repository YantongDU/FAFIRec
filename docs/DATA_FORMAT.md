# Prepared data and feature artifacts

`prepare_data.py` converts supplied raw files into one explicit next-item protocol.
It does **not** reconstruct the historical manuscript tables or silently infer a
missing chronological ordering. All file hashes are SHA-256, streamed from disk.

## Bundle format

A prepared directory contains:

- `items.jsonl`: one object per item, with contiguous `id` values `1..N`, unique
  string `raw_id`, and static metadata `text`. ID `0` is reserved for padding.
- `interactions.jsonl`: one object per user with string `user`, list `train`, and
  scalar `validation` and `test` item IDs. Each user has at least two training
  events and an item outside its **full** training set for negative sampling.
- `manifest.json`: `format_version: 1`, source paths/names and hashes, exact
  rating filter, ordering convention, inclusion/drop counts, metadata catalog
  scope, and hashes of both JSONL files in its `files` mapping.

Example item: `{"id": 1, "raw_id": "B00005BPJO", "text": "Milk chocolate"}`.
Example user: `{"user": "u1", "train": [1, 3], "validation": 2, "test": 4}`.

Sorting is timestamp ascending, with source file order breaking ties. The final
two surviving observations become validation and test targets. Validation uses
only training history; test appends the validation observation to that history.
Repeated item identities/events are retained. Their treatment in candidate
filtering is an explicit evaluation decision, not a preprocessing rewrite.

Metadata order determines item IDs where a catalog is supplied. All metadata
rows are retained, including items with no training feedback; the manifest
states this candidate scope. Without metadata, item IDs follow first surviving
event appearance and text is empty. An observed item missing from supplied
metadata is an error, not silently dropped. Clean/complete the catalog explicitly
if the raw release contains unmatched items.

Users with fewer than four surviving observations, or no training-unseen
negative, are excluded with counts in the manifest. No iterative k-core filter
or automatic deduplication is applied. The minimum rating is inclusive and must
be requested explicitly; absent a threshold, every observed rating is retained.
Book-Crossing zero ratings are not silently interpreted as explicit preferences.

The loader validates file hashes, item mapping, scalar targets, user uniqueness,
and diagnostic eligibility. Cold/warm reporting groups are disjoint bottom/top
10% of users by **full training count**, with prepared user order breaking ties.
They are descriptive relative groups, not an absolute few-shot definition.

## Raw-source adapters

```bash
# Repository fixture: ordered histories have no timestamps.
python prepare_data.py --source prime-pantry --input dataset \
  --output prepared/prime-pantry --allow-unverified-order

# ratings.dat + movies.dat; threshold is a new declared choice.
python prepare_data.py --source movielens1m --input /data/ml-1m \
  --output prepared/ml1m --min-rating 4

# training_set/mv_*.txt or combined_data_*.txt; movie_titles.txt when available.
python prepare_data.py --source netflix --input /data/netflix \
  --output prepared/netflix --min-rating 4

# No timestamps in this source: source-file order cannot prove chronology.
python prepare_data.py --source book-crossing --input /data/book-crossing \
  --output prepared/book-crossing --allow-unverified-order

# General interaction CSV: user,item,timestamp,rating; metadata CSV: item,text.
python prepare_data.py --source csv --input /data/interactions.csv \
  --metadata /data/items.csv --output prepared/custom --min-rating 4
```

`--metadata`/`--items` overrides source-specific catalog defaults. General CSV
column names can be changed with `--user-column`, `--item-column`,
`--timestamp-column`, and `--rating-column`. Timestamps accept finite Unix
numbers or ISO-8601; naive ISO times are interpreted as UTC. Missing timestamps
require `--allow-unverified-order`. If any surviving event lacks a timestamp,
**all** histories retain source order, and the manifest records chronology as
unverified. No mixed timestamp/guessed-order chronology is invented.

The Netflix adapter streams event rows into a temporary SQLite spool and sorts
histories on disk, holding one user sequence at a time. Allow sufficient disk
space for the source-sized spool and its index. Raw data are never downloaded
automatically. A nonempty output directory is never overwritten.

## Strict frozen embeddings

The default loader mode is `embeddings`, requiring `--text-embeddings`. There is
no lexical fallback. Supply a PyTorch floating-point tensor of shape
`[N+1, feature_dim]` whose row zero is all zero, plus a same-stem manifest:

```json
{
  "row_ids": ["<PAD>", "B00005BPJO", "B0000DIWNI"],
  "embedding_sha256": "SHA-256 of the exact .pt artifact",
  "base_checkpoint": "actual checkpoint used",
  "adapter": "actual encoder adapter used",
  "text_source": "static item metadata; no held-out behavioral targets"
}
```

For `features.pt`, the manifest must be `features.manifest.json`. Its `row_ids`
must exactly match `['<PAD>'] + raw_id values in items.jsonl order`. The loader
rejects missing manifests, incorrect ordering or hashes, non-finite tensors,
wrong shapes, non-floating values, or a nonzero padding row. Loading uses
`weights_only=True`; only plain tensor artifacts are supported. It does not guess
whether a missing row should be prepended.

Mapping/hash verification establishes artifact identity, not the truth of an
encoder provenance claim. Record real model/adapter revisions and metadata
construction in the manifest. Upstream metadata must not contain held-out user
behavior; the loader cannot infer that from prose alone.

## Explicit lexical smoke mode

`--feature-mode lexical` uses deterministic signed hashing of `[a-z0-9]+`
metadata tokens into a fixed-dimensional normalized bag of words. Every run
manifest labels these as **NOT LLM features** and records empty-text counts.
This is a pipeline check, not a substitute for formal semantic features.

Training pairs are shifted by one event and left padded. Negative items are
sampled outside the user's complete training set, not its truncated input.
Validation/test targets are not consulted when constructing the negative pool.
The supported experiment runner uses `num_workers=0` to keep per-user negative
sampling and loader shuffling reproducible.
