"""
text_index.py — Chunk clinical notes and build a FAISS vector index for text retrieval.

This creates the T branch of the unified retrieval pipeline:
  query -> embed -> FAISS top-k chunks

It saves:
  - data/text_chunks.parquet
  - data/faiss.index
  - data/text_chunk_meta.csv

Assumptions:
  - clinical notes live in a table named `notes`
  - columns: hadm_id, subject_id, charttime, category, text
"""

from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE = Path("data")
BASE.mkdir(exist_ok=True)

CHUNKS_PATH = BASE / "text_chunks.parquet"
META_PATH = BASE / "text_chunk_meta.csv"
INDEX_PATH = BASE / "faiss.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40
TOP_K_DEFAULT = 5


def load_notes() -> pd.DataFrame:
    candidates = [
        Path("data/lakehouse/notes.parquet"),
        Path("data/lakehouse/notes.csv"),
        Path("lakehouse/notes.parquet"),
        Path("lakehouse/notes.csv"),
        Path("data/notes.parquet"),
        Path("data/notes.csv"),
    ]
    for p in candidates:
        if p.exists():
            print(f"Loading notes from: {p}")
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(
        "Could not find notes.parquet or notes.csv in data/lakehouse/, lakehouse/, or data/"
    )


def clean_text(text: str) -> str:
    text = str(text).replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = clean_text(text).split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]
        if len(chunk) < 25:
            continue
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, row in df.iterrows():
        text = row.get("text", "")
        if pd.isna(text) or not str(text).strip():
            continue
        chunks = chunk_text(str(text))
        for j, ch in enumerate(chunks):
            rows.append({
                "chunk_id": f"{row.get('subject_id', 'na')}_{row.get('hadm_id', 'na')}_{i}_{j}",
                "subject_id": row.get("subject_id", None),
                "hadm_id": row.get("hadm_id", None),
                "charttime": row.get("charttime", None),
                "category": row.get("category", None),
                "chunk_text": ch,
                "source_row": int(i),
            })
    return pd.DataFrame(rows)


def build_index(chunks_df: pd.DataFrame):
    model = SentenceTransformer(MODEL_NAME)
    texts = chunks_df["chunk_text"].tolist()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, str(INDEX_PATH))
    chunks_df.to_parquet(CHUNKS_PATH, index=False)
    chunks_df.drop(columns=["chunk_text"]).to_csv(META_PATH, index=False)

    return index, model


def search(query: str, model, index, chunks_df: pd.DataFrame, top_k: int = TOP_K_DEFAULT):
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        row = chunks_df.iloc[idx]
        results.append({
            "score": float(score),
            "chunk_id": row["chunk_id"],
            "hadm_id": row["hadm_id"],
            "subject_id": row["subject_id"],
            "category": row["category"],
            "text": row["chunk_text"],
        })
    return results


def main():
    notes = load_notes()
    print(f"Loaded notes: {len(notes)} rows")
    chunks = build_chunks(notes)
    print(f"Built chunks: {len(chunks)}")
    if chunks.empty:
        raise ValueError("No chunks were created from notes")
    index, model = build_index(chunks)
    print(f"FAISS index size: {index.ntotal}")
    print(f"Saved chunks to: {CHUNKS_PATH}")
    print(f"Saved metadata to: {META_PATH}")
    print(f"Saved index to: {INDEX_PATH}")

    # quick smoke test
    sample_query = "chest pain and troponin"
    results = search(sample_query, model, index, chunks, top_k=3)
    print("\nSmoke test results:")
    for r in results:
        print(f"- score={r['score']:.3f} hadm_id={r['hadm_id']} category={r['category']}")
        print(r["text"][:220])
        print()


if __name__ == "__main__":
    main()