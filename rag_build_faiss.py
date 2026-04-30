#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def _iter_files(root: Path, exts: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        return []
    step = max(1, size - max(0, overlap))
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FAISS index from run corpus.")
    ap.add_argument("--corpus-dir", default="rag_corpus", help="Folder with run files")
    ap.add_argument("--outdir", default="rag_index", help="Output folder for index+meta")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--chunk-size", type=int, default=1200)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument(
        "--extensions",
        default=".csv,.txt,.md",
        help="Comma-separated file extensions to include",
    )
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exts = {e.strip().lower() for e in args.extensions.split(",") if e.strip()}
    files = list(_iter_files(corpus_dir, exts))
    if not files:
        raise SystemExit(f"No files found in {corpus_dir} with {sorted(exts)}")

    model = SentenceTransformer(args.model)
    texts: list[str] = []
    meta: list[dict] = []

    for path in files:
        text = _read_text(path)
        if not text:
            continue
        for idx, chunk in enumerate(_chunk_text(text, args.chunk_size, args.chunk_overlap)):
            meta.append(
                {
                    "id": len(texts),
                    "path": str(path),
                    "chunk": idx,
                    "text": chunk,
                }
            )
            texts.append(chunk)

    if not texts:
        raise SystemExit("No text chunks produced.")

    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(outdir / "index.faiss"))
    with (outdir / "meta.jsonl").open("w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (outdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "extensions": sorted(exts),
                "corpus_dir": str(corpus_dir),
                "total_chunks": len(texts),
            },
            f,
            indent=2,
        )

    print(f"Wrote {len(texts)} chunks → {outdir}")


if __name__ == "__main__":
    main()
