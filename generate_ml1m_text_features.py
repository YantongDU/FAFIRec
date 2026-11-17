"""Generate mapped, genuine Qwen3 item embeddings for the ML-1M experiment.

The original manuscript did not name a reproducible text checkpoint. This is
an explicit Qwen3-Embedding-0.6B choice, not a reproduction of PAD/LLM2Vec
embeddings. Only static prepared item text is encoded; no user targets enter.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
WEIGHT_NAME = "model.safetensors"
WEIGHT_SIZE = 1191586416
WEIGHT_SHA256 = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
DEFAULT_MODEL_DIR = ROOT / "data/models/qwen3_embedding_06b"
SMALL_ASSETS = {
    "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "merges.txt", "special_tokens_map.json", "added_tokens.json",
    "generation_config.json", "README.md", "LICENSE",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def resolve_url(name, suffix=""):
    return f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{urllib.parse.quote(name)}{suffix}"


def _read_url(url, *, headers=None, timeout=45):
    request = urllib.request.Request(url, headers={"User-Agent": "FAFIRec-ML1M-research", **(headers or {})})
    return urllib.request.urlopen(request, timeout=timeout)


def download_assets(model_dir, workers=8, chunk_bytes=8 * 1024 * 1024):
    """Anonymous HTTPS downloads; validate exact ranges, sizes, and official hashes."""
    model_dir = Path(model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
    with _read_url(api) as response:
        metadata = json.load(response)
    if metadata.get("sha") != REVISION or metadata.get("gated"):
        raise ValueError("The fixed official public model revision was not returned")
    entries = {entry["rfilename"]: entry for entry in metadata["siblings"]}
    weight = entries[WEIGHT_NAME]
    lfs = weight.get("lfs") or {}
    if weight.get("size") != WEIGHT_SIZE or lfs.get("sha256") != WEIGHT_SHA256:
        raise ValueError("Official weight metadata does not match the audited revision")
    json_dump(model_dir / "official_model_info.json", metadata)
    checksums = {}
    for name in sorted(SMALL_ASSETS & set(entries)):
        entry = entries[name]
        with _read_url(resolve_url(name)) as response:
            value = response.read()
        if len(value) != entry.get("size"):
            raise ValueError(f"Asset size mismatch: {name}")
        blob = hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest()
        if entry.get("lfs"):
            if hashlib.sha256(value).hexdigest() != entry["lfs"]["sha256"]:
                raise ValueError(f"LFS content hash mismatch: {name}")
        elif entry.get("blobId") and blob != entry["blobId"]:
            raise ValueError(f"Git blob hash mismatch: {name}")
        path = model_dir / name
        if path.exists() and path.read_bytes() != value:
            raise FileExistsError(f"Existing model asset differs from the fixed revision: {path}")
        if not path.exists():
            path.write_bytes(value)
        checksums[name] = sha256_file(path)
    for required in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        if required not in checksums:
            raise ValueError(f"Required native Transformers asset absent: {required}")

    target = model_dir / WEIGHT_NAME
    if target.exists():
        if target.stat().st_size != WEIGHT_SIZE or sha256_file(target) != WEIGHT_SHA256:
            raise ValueError("Existing model weight file fails its official size/hash")
    else:
        parts_dir = model_dir / "download_parts"
        parts_dir.mkdir(exist_ok=True)
        starts = list(range(0, WEIGHT_SIZE, chunk_bytes))

        def download_part(start):
            stop = min(start + chunk_bytes, WEIGHT_SIZE) - 1
            path = parts_dir / f"part_{start:012d}"
            expected_size = stop - start + 1
            if path.exists() and path.stat().st_size == expected_size:
                return expected_size
            for attempt in range(4):
                try:
                    suffix = f"?download=true&fafirec_range={start}-{stop}"
                    with _read_url(resolve_url(WEIGHT_NAME, suffix), headers={"Range": f"bytes={start}-{stop}"}, timeout=75) as response:
                        wanted = f"bytes {start}-{stop}/{WEIGHT_SIZE}"
                        if response.status != 206 or response.headers.get("Content-Range") != wanted:
                            raise ValueError(f"Server did not honor exact range {wanted}")
                        value = response.read()
                    if len(value) != expected_size:
                        raise ValueError("Incomplete range body")
                    temporary = path.with_suffix(".tmp")
                    temporary.write_bytes(value)
                    os.replace(temporary, path)
                    return expected_size
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(min(2 ** attempt, 8))

        completed, last_report = 0, -1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(download_part, start) for start in starts]
            for future in as_completed(futures):
                completed += future.result()
                report = int(100 * completed / WEIGHT_SIZE) // 5
                if report > last_report:
                    print(f"Model download {completed}/{WEIGHT_SIZE} bytes ({100 * completed / WEIGHT_SIZE:.1f}%)", flush=True)
                    last_report = report
        temporary = model_dir / (WEIGHT_NAME + ".partial")
        digest = hashlib.sha256()
        with temporary.open("wb") as stream:
            for start in starts:
                with (parts_dir / f"part_{start:012d}").open("rb") as part:
                    for value in iter(lambda: part.read(1024 * 1024), b""):
                        stream.write(value)
                        digest.update(value)
        if temporary.stat().st_size != WEIGHT_SIZE or digest.hexdigest() != WEIGHT_SHA256:
            raise ValueError("Assembled model does not match the official SHA-256")
        os.replace(temporary, target)
    checksums[WEIGHT_NAME] = WEIGHT_SHA256
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "gated": False,
                "license": metadata.get("cardData", {}).get("license"),
                "weight_bytes": WEIGHT_SIZE, "weight_sha256": WEIGHT_SHA256,
                "files_sha256": checksums, "official_metadata_url": api,
                "download": "anonymous HTTPS; exact Content-Range validation; final official SHA-256"}
    json_dump(model_dir / "asset_manifest.json", manifest)
    print("Official model assets verified", flush=True)
    return manifest


def read_items(data_dir):
    data_dir = Path(data_dir)
    source = data_dir / "items.jsonl"
    data_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    if data_manifest["files"]["items.jsonl"] != sha256_file(source):
        raise ValueError("Prepared static item file fails its data-manifest hash")
    items = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not items or [row["id"] for row in items] != list(range(1, len(items) + 1)):
        raise ValueError("Item IDs must be contiguous 1..N in exact feature-row order")
    raw = [row["raw_id"] for row in items]
    if len(set(raw)) != len(raw) or any(not isinstance(x, str) or not x or x == "<PAD>" for x in raw):
        raise ValueError("Raw item identifiers must be unique nonempty strings")
    if any(not isinstance(row.get("text"), str) or not row["text"].strip() for row in items):
        raise ValueError("A real nonempty static metadata string is required for every item")
    return items, data_manifest


def encode(args):
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from torch.nn import functional as F
    import transformers
    from transformers import AutoModel, AutoTokenizer
    from packaging.version import Version
    if Version(transformers.__version__) < Version("4.51.0"):
        raise RuntimeError("Qwen3 requires Transformers >=4.51.0 in a separate feature environment")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this feature environment has no available CUDA")
    if args.device == "cpu" and args.dtype != "float32":
        raise ValueError("Use explicit float32 for this CPU feature-generation workflow")
    model_dir = args.model_dir.resolve()
    asset = json.loads((model_dir / "asset_manifest.json").read_text(encoding="utf-8"))
    if asset.get("revision") != REVISION or asset.get("weight_sha256") != WEIGHT_SHA256:
        raise ValueError("The local model asset manifest does not match the audited revision")
    for name, checksum in asset["files_sha256"].items():
        if sha256_file(model_dir / name) != checksum:
            raise ValueError(f"Local model asset changed: {name}")
    items, item_data_manifest = read_items(args.data_dir)
    initial_items_hash = item_data_manifest["files"]["items.jsonl"]
    initial_data_manifest_hash = sha256_file(args.data_dir / "manifest.json")
    initial_generator_hash = sha256_file(__file__)
    if not args.probe and (args.output.exists() or args.output.with_suffix(".manifest.json").exists()):
        raise FileExistsError("Use a new feature-output path; existing artifacts are not overwritten")
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False, padding_side="left")
    load_options = {"local_files_only": True, "trust_remote_code": False,
                    "torch_dtype": getattr(torch, args.dtype)}
    if args.device == "cuda":
        # Directly materialize mapped safetensors on GPU. This avoids a full
        # CPU model allocation on memory-constrained hosts (requires Accelerate).
        load_options.update({"device_map": {"": 0}, "low_cpu_mem_usage": True,
                             "attn_implementation": "sdpa"})
    else:
        load_options["attn_implementation"] = "eager"
    model = AutoModel.from_pretrained(model_dir, **load_options)
    model.config.use_cache = False
    model = model.eval()
    if args.device == "cpu":
        model = model.to("cpu")
    texts = [row["text"] for row in items]
    if args.probe:
        texts = texts[:min(4, len(texts))]
    lengths = [len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]) for text in texts]
    vectors = torch.zeros(len(texts) + 1, model.config.hidden_size, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch = tokenizer(texts[start:start + args.batch_size], padding=True, truncation=True,
                              max_length=args.max_length, return_tensors="pt")
            batch = {key: value.to(args.device) for key, value in batch.items()}
            outputs = model(**batch, use_cache=False)
            # Left padding guarantees the last column is a valid final token.
            if not bool(batch["attention_mask"][:, -1].all()):
                raise RuntimeError("Last-token pooling requires the declared left-padding convention")
            pooled = outputs.last_hidden_state[:, -1].float()
            values = F.normalize(pooled, p=2, dim=-1).cpu()
            if values.shape != (len(batch["input_ids"]), model.config.hidden_size) or not bool(torch.isfinite(values).all()):
                raise ValueError("The encoder returned invalid vectors")
            if not torch.allclose(values.norm(dim=-1), torch.ones(len(values)), rtol=1e-4, atol=1e-5):
                raise ValueError("Expected finite unit-normalized semantic embeddings")
            vectors[start + 1:start + 1 + len(values)] = values
            if args.probe or start == 0 or (start // args.batch_size + 1) % 20 == 0 or start + len(values) == len(texts):
                print(f"Encoded {start + len(values)}/{len(texts)} static movie descriptions; {time.monotonic() - started:.1f}s", flush=True)
    if sha256_file(args.data_dir / "items.jsonl") != initial_items_hash or sha256_file(args.data_dir / "manifest.json") != initial_data_manifest_hash:
        raise RuntimeError("Prepared item mapping changed while features were being generated")
    if sha256_file(__file__) != initial_generator_hash:
        raise RuntimeError("Generator source changed during this run")
    manifest = {"encoder": MODEL_ID, "model_revision": REVISION, "model_weight_sha256": WEIGHT_SHA256,
                "model_files_sha256": asset["files_sha256"], "pooling": "last valid token with left padding",
                "normalization": "L2 after float32 conversion", "feature_type": "genuine frozen decoder-LLM item embeddings; Qwen3",
                "text_source": "prepared static MovieLens title (including release year) and genres only",
                "query_instruction": None, "heldout_behavior_used": False,
                "max_length": args.max_length, "truncated_items": sum(n > args.max_length for n in lengths),
                "untruncated_token_lengths": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
                "batch_size": args.batch_size, "device": args.device, "inference_dtype": args.dtype,
                "attention_implementation": load_options["attn_implementation"],
                "saved_dtype": "float32", "shape": list(vectors.shape),
                "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
                "tokenizers_version": importlib.metadata.version("tokenizers"),
                "generator_sha256": initial_generator_hash, "items_sha256": initial_items_hash,
                "data_manifest_sha256": initial_data_manifest_hash,
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "elapsed_seconds": time.monotonic() - started, "created_utc": datetime.now(timezone.utc).isoformat(),
                "provenance_note": "The manuscript lacked an exact text checkpoint. This is a declared encoder choice, not PAD/LLM2Vec reproduction."}
    if args.probe:
        manifest["probe_items"] = len(texts)
        json_dump(model_dir / "probe.json", manifest)
        print(json.dumps({"probe": "passed", "items": len(texts), "shape": list(vectors.shape),
                          "seconds": manifest["elapsed_seconds"]}), flush=True)
        return manifest
    manifest["row_ids"] = ["<PAD>"] + [row["raw_id"] for row in items]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(vectors, temporary)
    os.replace(temporary, args.output)
    manifest["embedding_sha256"] = sha256_file(args.output)
    json_dump(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps({"output": str(args.output), "shape": list(vectors.shape),
                      "embedding_sha256": manifest["embedding_sha256"],
                      "truncated_items": manifest["truncated_items"], "seconds": manifest["elapsed_seconds"]}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/ml1m_paper_ratio")
    parser.add_argument("--output", type=Path, default=ROOT / "data/ml1m_qwen3_06b.pt")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    args = parser.parse_args()
    if min(args.download_workers, args.batch_size, args.max_length, args.threads) < 1:
        parser.error("Worker counts, batch size, length and threads must be positive")
    if args.download_only:
        download_assets(args.model_dir, workers=args.download_workers)
    else:
        encode(args)


if __name__ == "__main__":
    main()
