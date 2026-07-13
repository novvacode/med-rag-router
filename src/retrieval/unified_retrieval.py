"""
unified_retrieval.py — T / TE / TEK retrieval wrapper.

Modes:
  T   = text-only FAISS retrieval
  TE  = text + EHR snapshot
  TEK = text + EHR snapshot + KG facts

This module is the single retrieval entry point for Phase 3 and later router experiments.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ehr.ehr_snapshot import format_ehr_snapshot
from mkg.retrieval import retrieve_kg_context

BASE = Path("data")
CHUNKS_PATH = BASE / "text_chunks.parquet"
INDEX_PATH = BASE / "faiss.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


class TextRetriever:
    def __init__(self, chunks_path=CHUNKS_PATH, index_path=INDEX_PATH, model_name=MODEL_NAME):
        self.chunks = pd.read_parquet(chunks_path)
        self.index = faiss.read_index(str(index_path))
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = TOP_K):
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, idxs = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            row = self.chunks.iloc[idx]
            results.append({
                "score": float(score),
                "chunk_id": row["chunk_id"],
                "hadm_id": row["hadm_id"],
                "subject_id": row["subject_id"],
                "category": row["category"],
                "text": row["chunk_text"],
            })
        return results


def format_text_context(chunks: list) -> str:
    if not chunks:
        return "No relevant clinical notes found."
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Note {i} | score={c['score']:.3f} | hadm_id={c['hadm_id']}] {c['text']}")
    return "\n\n".join(parts)


def retrieve_context(question: str, hadm_id: int | None = None, mode: str = "T", top_k: int = TOP_K):
    """
    Returns a dict with keys: mode, text_context, ehr_context, kg_context, combined_context, retrieved_notes.
    """
    retriever = TextRetriever()
    note_chunks = retriever.retrieve(question, top_k=top_k)

    text_context = format_text_context(note_chunks)
    ehr_context = ""
    kg_context = ""

    if mode in ("TE", "TEK"):
        if hadm_id is None:
            raise ValueError("hadm_id is required for TE/TEK modes")
        ehr_context = format_ehr_snapshot(int(hadm_id))

    if mode == "TEK":
        if hadm_id is None:
            raise ValueError("hadm_id is required for TEK mode")
        snapshot = ehr_context
        dx_part = ""
        if "Diagnoses:" in snapshot:
            dx_part = snapshot.split("Diagnoses:", 1)[1].split("Labs:", 1)[0].strip().rstrip(".")
        diagnosis_texts = [d.strip() for d in dx_part.split(",") if d.strip()] if dx_part else []
        kg_context = retrieve_kg_context(diagnosis_texts)

    combined_parts = []
    if text_context:
        combined_parts.append(f"NOTES:\n{text_context}")
    if ehr_context:
        combined_parts.append(f"EHR:\n{ehr_context}")
    if kg_context:
        combined_parts.append(f"KNOWLEDGE:\n{kg_context}")

    combined_context = "\n\n".join(combined_parts)

    return {
        "mode": mode,
        "text_context": text_context,
        "ehr_context": ehr_context,
        "kg_context": kg_context,
        "combined_context": combined_context,
        "retrieved_notes": note_chunks,
    }


if __name__ == "__main__":
    q = "patient with chest pain and elevated troponin"
    demo_hadm = 21351702
    for mode in ["T", "TE", "TEK"]:
        print(f"\n=== MODE {mode} ===")
        ctx = retrieve_context(q, hadm_id=demo_hadm, mode=mode, top_k=3)
        print("--- TEXT ---")
        print(ctx["text_context"][:1800])
        if ctx["ehr_context"]:
            print("\n--- EHR ---")
            print(ctx["ehr_context"][:1800])
        if ctx["kg_context"]:
            print("\n--- KNOWLEDGE ---")
            print(ctx["kg_context"][:1800])
        print("\n--- COMBINED ---")
        print(ctx["combined_context"][:2500])