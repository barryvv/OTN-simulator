#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def _load_meta(meta_path: Path) -> list[dict]:
    rows = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Query FAISS index built from simulator runs.")
    ap.add_argument("--index-dir", default="rag_index")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--as-prompt", action="store_true")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    meta = _load_meta(index_dir / "meta.jsonl")
    if not meta:
        raise SystemExit("Meta is empty.")

    model = SentenceTransformer(args.model)
    q = model.encode([args.query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    scores, idxs = index.search(q, args.top_k)

    hits = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        if idx < 0 or idx >= len(meta):
            continue
        row = meta[idx]
        hits.append((rank, float(score), row))

    if args.as_prompt:
        blocks = []
        for rank, score, row in hits:
            header = f"[{rank}] {row['path']} (score={score:.4f}, chunk={row['chunk']})"
            blocks.append(header + "\n" + row["text"])
        print("\n\n".join(blocks))
        return

    for rank, score, row in hits:
        print(f"[{rank}] score={score:.4f} path={row['path']} chunk={row['chunk']}")
        preview = row["text"].replace("\n", " ").strip()
        if len(preview) > 280:
            preview = preview[:280] + "..."
        print(f"  {preview}\n")


if __name__ == "__main__":
    main()
