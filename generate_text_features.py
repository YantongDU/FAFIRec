"""Generate explicitly mapped LLM2Vec features for a prepared item catalog."""
import argparse
import json
from pathlib import Path

from fafirec.data import file_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Actual base checkpoint path or Hugging Face ID")
    parser.add_argument("--adapter", required=True, help="Matching LLM2Vec adapter path or ID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args(argv)
    import torch
    from llm2vec import LLM2Vec
    if min(args.batch_size, args.chunk_size) < 1:
        parser.error("batch and chunk sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled environment is required for this device")
    if args.output.exists() or args.output.with_suffix('.manifest.json').exists():
        raise FileExistsError("Refusing to overwrite a feature artifact")
    source = args.data_dir / "items.jsonl"
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest["files"]["items.jsonl"] != file_sha256(source):
        raise ValueError("Item metadata differs from the prepared data manifest")
    items = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not items or [r["id"] for r in items] != list(range(1, len(items) + 1)):
        raise ValueError("Item rows must be contiguous and nonempty")
    encoder = LLM2Vec.from_pretrained(args.model, peft_model_name_or_path=args.adapter,
                                    device_map=args.device, torch_dtype=getattr(torch, args.dtype))
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(items), args.chunk_size):
            texts = [r["text"] for r in items[start:start + args.chunk_size]]
            values = torch.as_tensor(encoder.encode(texts, batch_size=args.batch_size)).detach().float().cpu()
            if values.ndim != 2 or len(values) != len(texts) or not torch.isfinite(values).all():
                raise ValueError("Encoder returned invalid features")
            chunks.append(values)
            print(f"Encoded {min(start + len(texts), len(items))}/{len(items)} items", flush=True)
    values = torch.cat(chunks)
    values = torch.cat([torch.zeros(1, values.shape[1]), values])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(values, args.output)
    feature_manifest = {"row_ids": ["<PAD>"] + [r["raw_id"] for r in items],
                        "embedding_sha256": file_sha256(args.output), "shape": list(values.shape),
                        "base_checkpoint": args.model, "adapter": args.adapter,
                        "items_sha256": file_sha256(source), "text_source": "prepared item metadata only",
                        "dtype": args.dtype, "encoder": "LLM2Vec", "torch_version": str(torch.__version__)}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(feature_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
