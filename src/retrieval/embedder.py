"""
src/retrieval/embedder.py

Text Embedder + FAISS Index Builder.

Encodes clinical notes (and optionally MKG linearized facts + guidelines)
into a FAISS vector index for fast approximate nearest-neighbor retrieval.

Supported embedding models:
    - BAAI/bge-small-en-v1.5       (recommended: fast, strong on medical text)
    - sentence-transformers/all-MiniLM-L6-v2  (fallback: smaller, lighter)

Outputs (saved to embeddings/):
    embeddings/notes_index.faiss   <- FAISS index
    embeddings/notes_chunks.parquet <- chunk metadata (text, hadm_id, source, etc.)

Usage:
    # Build index from notes
    python src/retrieval/embedder.py

    # Use alternate model
    python src/retrieval/embedder.py --model sentence-transformers/all-MiniLM-L6-v2

    # Limit chunks for quick testing
    python src/retrieval/embedder.py --max-chunks 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# -- Config --------------------------------------------------------------------

LAKE         = Path("data/lakehouse")
NOTES_FILE   = LAKE / "notes.parquet"
OUT_DIR      = Path("embeddings")
INDEX_FILE   = OUT_DIR / "notes_index.faiss"
CHUNKS_FILE  = OUT_DIR / "notes_chunks.parquet"

DEFAULT_MODEL   = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE      = 256    # words per chunk
CHUNK_OVERLAP   = 32     # word overlap between chunks
BATCH_SIZE      = 64     # sentences per embedding batch
MAX_NOTE_WORDS  = 2000   # truncate very long notes before chunking
MAX_CHUNKS      = None   # set via --max-chunks for quick tests


# -- Text chunking -------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = None,
               overlap: int = None) -> list[str]:
    """
    Split text into overlapping word-level chunks.
    Returns list of chunk strings.
    """
    # Use globals if not explicitly provided
    chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
    overlap = overlap if overlap is not None else CHUNK_OVERLAP

    words = text.split()
    if not words:
        return []

    # Truncate very long notes
    if len(words) > MAX_NOTE_WORDS:
        words = words[:MAX_NOTE_WORDS]

    chunks: list[str] = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        if i + chunk_size >= len(words):
            break

    return chunks


# -- Notes loading -------------------------------------------------------------

def load_notes(max_chunks: int | None = None) -> pd.DataFrame:
    """
    Load notes from Parquet, chunk each note, return a flat DataFrame of chunks.

    Returns DataFrame with columns:
        chunk_id, hadm_id, subject_id, source, category, chartdate, text
    """
    if not NOTES_FILE.exists():
        raise FileNotFoundError(
            f"Notes file not found: {NOTES_FILE.resolve()}\n"
            "Expected: data/lakehouse/notes.parquet"
        )

    print(f"[INFO] Loading notes from {NOTES_FILE}...")
    df_notes = pd.read_parquet(NOTES_FILE)
    print(f"[INFO] Loaded {len(df_notes):,} notes")

    # Detect column names (handle variants)
    cols = {c.lower(): c for c in df_notes.columns}

    text_col     = cols.get("text", cols.get("note_text", cols.get("content")))
    hadm_col     = cols.get("hadm_id", cols.get("hadmid"))
    subject_col  = cols.get("subject_id", cols.get("subjectid"))
    category_col = cols.get("category", cols.get("note_type"))
    date_col     = cols.get("chartdate", cols.get("charttime", cols.get("chart_date")))

    if text_col is None:
        raise ValueError(
            f"No text column found in notes. Available: {list(df_notes.columns)}"
        )

    print(f"[INFO] Columns detected: text={text_col}, hadm={hadm_col}, "
          f"category={category_col}, date={date_col}")

    # Drop rows with empty text
    df_notes = df_notes[df_notes[text_col].notna()].copy()
    df_notes = df_notes[df_notes[text_col].str.strip().str.len() > 20]
    print(f"[INFO] Notes after filtering empty: {len(df_notes):,}")

    # Build chunk records
    print("[INFO] Chunking notes...")
    records: list[dict] = []
    chunk_id = 0

    for _, row in df_notes.iterrows():
        text    = str(row[text_col])
        hadm_id = int(row[hadm_col]) if hadm_col and pd.notna(row.get(hadm_col)) else None
        subj_id = int(row[subject_col]) if subject_col and pd.notna(row.get(subject_col)) else None
        cat     = str(row[category_col]) if category_col and pd.notna(row.get(category_col)) else "unknown"
        date    = str(row[date_col])     if date_col and pd.notna(row.get(date_col)) else None

        chunks = chunk_text(text)
        for chunk in chunks:
            records.append({
                "chunk_id":   chunk_id,
                "hadm_id":    hadm_id,
                "subject_id": subj_id,
                "source":     "mimic_notes",
                "category":   cat,
                "chartdate":  date,
                "text":       chunk,
            })
            chunk_id += 1

        if max_chunks and chunk_id >= max_chunks:
            print(f"[INFO] max_chunks={max_chunks} reached — stopping early.")
            break

    df_chunks = pd.DataFrame(records)
    print(f"[INFO] Total chunks: {len(df_chunks):,} from {len(df_notes):,} notes")
    return df_chunks


# -- Embedding -----------------------------------------------------------------

def load_model(model_name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    # FIXED: Replaced deprecated get_sentence_embedding_dimension()
    dim = model.get_embedding_dimension() 
    print(f"[INFO] Embedding dimension: {dim}")
    return model


def embed_chunks(
    model: SentenceTransformer,
    texts: list[str],
    model_name: str,  # FIXED: Passed model_name explicitly
    batch_size: int = BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode list of text strings -> numpy float32 array of shape (N, dim).
    Uses BGE query prefix if the model supports it.
    """
    # FIXED: Check model_name string directly instead of using private _model_config
    use_prefix = "bge" in model_name.lower()
    prefix = "Represent this clinical text for retrieval: " if use_prefix else ""

    if prefix:
        texts_to_encode = [prefix + t for t in texts]
    else:
        texts_to_encode = texts

    print(f"[INFO] Encoding {len(texts):,} chunks "
          f"(batch_size={batch_size}, prefix={'yes' if prefix else 'no'})...")

    t0 = time.time()
    embeddings = model.encode(
        texts_to_encode,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # cosine similarity via inner product
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    rate = len(texts) / elapsed
    print(f"[INFO] Encoded {len(texts):,} chunks in {elapsed:.1f}s "
          f"({rate:.0f} chunks/s)")
    return embeddings.astype(np.float32)


# -- FAISS index ---------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP (inner product) index.
    Since embeddings are L2-normalized, inner product == cosine similarity.

    For > 100k chunks, consider IndexIVFFlat for speed.
    """
    n, dim = embeddings.shape
    print(f"[INFO] Building FAISS IndexFlatIP: {n:,} vectors x {dim}d")

    if n > 100_000:
        # IVF index for large collections
        nlist   = min(int(np.sqrt(n)), 256)
        quantizer = faiss.IndexFlatIP(dim)
        index     = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        print(f"[INFO] Using IndexIVFFlat with nlist={nlist} (n={n:,} > 100k)")
        index.train(embeddings)
        index.nprobe = min(nlist, 16)
    else:
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    print(f"[INFO] FAISS index built: {index.ntotal:,} vectors")
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[INFO] FAISS index saved: {path} ({size_mb:.1f} MB)")


def save_chunks(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[INFO] Chunk metadata saved: {path} ({len(df):,} rows)")


# -- Sanity check --------------------------------------------------------------

def sanity_check(
    index: faiss.Index,
    df_chunks: pd.DataFrame,
    model: SentenceTransformer,
    model_name: str, # FIXED: Passed model_name explicitly
    n_queries: int = 3,
) -> None:
    """Run a few test queries against the built index."""
    print("\n[INFO] Sanity check — test queries:")
    test_queries = [
        "What is the primary diagnosis for this patient?",
        "Type 2 diabetes mellitus management",
        "Acute kidney injury lab findings",
    ][:n_queries]

    # FIXED: Use model_name parameter directly
    use_prefix = "bge" in model_name.lower()
    query_prefix = "Represent this clinical question for retrieval: " if use_prefix else ""

    for query in test_queries:
        encoded = model.encode(
            [query_prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        D, I = index.search(encoded, k=3)

        print(f"\n  Query: '{query}'")
        for rank, (score, idx) in enumerate(zip(D[0], I[0])):
            if idx < 0 or idx >= len(df_chunks):
                continue
            chunk_text_preview = df_chunks.iloc[idx]["text"][:120].replace("\n", " ")
            hadm_id   = df_chunks.iloc[idx].get("hadm_id", "?")
            category  = df_chunks.iloc[idx].get("category", "?")
            print(f"    #{rank+1} score={score:.4f} hadm={hadm_id} "
                  f"cat={category}")
            print(f"        '{chunk_text_preview}...'")


# -- Main ----------------------------------------------------------------------

def main() -> None:
    global CHUNK_SIZE, CHUNK_OVERLAP
    
    parser = argparse.ArgumentParser(
        description="Build FAISS index over MIMIC-IV clinical notes."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace embedding model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--max-chunks", type=int, default=None,
        help="Max chunks to embed (useful for quick tests, e.g. --max-chunks 500)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Embedding batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Words per chunk (default: {CHUNK_SIZE})"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=CHUNK_OVERLAP,
        help=f"Word overlap between chunks (default: {CHUNK_OVERLAP})"
    )
    parser.add_argument(
        "--no-sanity", action="store_true",
        help="Skip sanity-check queries after building index"
    )
    args = parser.parse_args()

    CHUNK_SIZE    = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap

    t_total = time.time()

    # -- Step 1: Load + chunk notes ---------------------------------------------
    df_chunks = load_notes(max_chunks=args.max_chunks)

    if df_chunks.empty:
        print("[ERROR] No chunks produced. Check notes.parquet content.")
        return

    # -- Step 2: Load embedding model -------------------------------------------
    model = load_model(args.model)

    # -- Step 3: Embed ----------------------------------------------------------
    embeddings = embed_chunks(
        model,
        df_chunks["text"].tolist(),
        model_name=args.model, # FIXED: Pass args.model here
        batch_size=args.batch_size,
    )

    # -- Step 4: Build + save FAISS index ---------------------------------------
    index = build_faiss_index(embeddings)
    save_index(index, INDEX_FILE)
    save_chunks(df_chunks, CHUNKS_FILE)

    # -- Step 5: Sanity check ---------------------------------------------------
    if not args.no_sanity:
        sanity_check(index, df_chunks, model, model_name=args.model) # FIXED: Pass args.model here

    # -- Summary ----------------------------------------------------------------
    elapsed = time.time() - t_total
    print("\n" + "=" * 60)
    print("FAISS INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"  Chunks indexed : {index.ntotal:,}")
    print(f"  Embedding dim  : {index.d}")
    print(f"  Model used     : {args.model}")
    print(f"  Index file     : {INDEX_FILE.resolve()}")
    print(f"  Chunks file    : {CHUNKS_FILE.resolve()}")
    print(f"  Total time     : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()