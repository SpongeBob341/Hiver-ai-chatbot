"""
RAG Index Builder
------------------
Builds the BM25 + FAISS retrieval index over stitched microsofthelps threads.

V1 design decisions (knowledgebase §11, §13.3):
  • Granularity:  Full Thread — entire stitched conversation (≤6 turns)
  • Embedder:     all-MiniLM-L6-v2 (384-dim, 512-token window)
  • Bare-URL:     Option B — bare-URL tweets excluded from retrieval (flagged)
  • Stitching:    Phase 1 thread stitcher applied before embedding
  • BM25:         rank-bm25 (BM25Okapi, tokenized on lowercased whitespace split)
  • FAISS:        IndexFlatIP (inner product on L2-normalized vectors = cosine sim)

Output files (saved to Google Drive in Colab):
  bm25_index.pkl         — serialized BM25Okapi object
  faiss_index.bin        — FAISS flat inner product index
  thread_metadata.json   — [{thread_id, text, intent_label, escalation_flag, ...}]

Run this script once offline; load in notebook via HybridRetriever.load().
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from data.prepare_golden_set import CLUSTER_TO_CLASS, get_escalation_category
except ImportError:
    from prepare_golden_set import CLUSTER_TO_CLASS, get_escalation_category


# ---------------------------------------------------------------------------
# Bare-URL detection (mirrors Phase 1 logic)
# ---------------------------------------------------------------------------

RE_URL     = re.compile(r"https?://\S+|www\.\S+")
RE_MENTION = re.compile(r"@\w+")


def is_bare_url_tweet(text: str) -> bool:
    """Return True if the tweet is only URLs/mentions with no substantive text."""
    stripped = RE_MENTION.sub("", str(text)).strip()
    stripped = RE_URL.sub("", stripped).strip()
    return len(stripped) < 5   # less than 5 non-URL chars = bare-URL tweet


# ---------------------------------------------------------------------------
# Thread stitcher (re-implemented here for standalone use)
# ---------------------------------------------------------------------------

RE_PART_START = re.compile(r"\b1\s*/\s*\d+\b|\b1\s+of\s+\d+\b", re.IGNORECASE)
RE_PART_CONT  = re.compile(r"\b[2-9]\s*/\s*\d+\b|\b[2-9]\s+of\s+\d+\b", re.IGNORECASE)
RE_CONT_LABEL = re.compile(r"\(cont\.?\)", re.IGNORECASE)


def stitch_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive same-author multi-part tweets."""
    if not turns:
        return []

    stitched = []
    i = 0
    while i < len(turns):
        current = dict(turns[i])
        text = current.get("text", "")
        is_start = RE_PART_START.search(text) or RE_CONT_LABEL.search(text)

        if is_start:
            parts = [text]
            j = i + 1
            while j < len(turns):
                nxt = turns[j]
                nxt_text = nxt.get("text", "")
                is_cont = RE_PART_CONT.search(nxt_text) or RE_CONT_LABEL.search(nxt_text)
                same_author = nxt.get("author_id") == current.get("author_id")
                if same_author and is_cont:
                    parts.append(nxt_text)
                    j += 1
                else:
                    break

            if len(parts) > 1:
                cleaned = []
                for p in parts:
                    p = RE_PART_START.sub("", p)
                    p = RE_PART_CONT.sub("", p)
                    p = RE_CONT_LABEL.sub("", p).strip()
                    cleaned.append(p)
                current["text"] = " ".join(cleaned)
                current["is_stitched"] = True
                i = j
            else:
                i += 1
        else:
            i += 1

        stitched.append(current)
    return stitched


def prepare_thread_text(turns: list[dict[str, Any]], max_turns: int = 6) -> str:
    """Stitch turns, cap at max_turns, format as embedding-ready string."""
    stitched = stitch_turns(turns)
    stitched = stitched[:max_turns]
    parts = []
    for t in stitched:
        author = t.get("inbound", None)
        if author is None:
            handle = "customer" if t.get("inbound", True) else "microsofthelps"
        else:
            handle = "customer" if author else "microsofthelps"
        parts.append(f"{handle}: {t.get('text', '')}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(
    threads_df:         pd.DataFrame,
    intent_labels_df:   Optional[pd.DataFrame] = None,
    thread_id_col:      str = "thread_id",
    text_col:           str = "text",
    author_col:         str = "inbound",       # True = customer, False = brand
    turn_pos_col:       str = "turn_pos",
    cluster_col:        str = "cluster_final",
    max_turns:          int = 6,
    embedder_name:      str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size:         int = 64,
    output_dir:         str = "/content/drive/MyDrive/hiver_index/",
    device:             Optional[str] = None,
) -> dict[str, Any]:
    """
    Build BM25 + FAISS index from thread DataFrame.

    Args:
        threads_df:       DataFrame with all tweet turns. Must have thread_id_col,
                          text_col, and optionally author_col, turn_pos_col, cluster_col.
        intent_labels_df: Optional DataFrame mapping thread_id → intent label.
                          If None, cluster_col in threads_df is used.
        thread_id_col:    Thread identifier column.
        text_col:         Tweet text column.
        author_col:       Column indicating if tweet is inbound (customer) or outbound (brand).
                          True/1 = customer, False/0 = microsofthelps.
        turn_pos_col:     Turn position column (for ordering).
        cluster_col:      Intent cluster column (from k9 split taxonomy).
        max_turns:        Maximum turns per thread before embedding (6 = V1 lock).
        embedder_name:    SentenceTransformer model name.
        batch_size:       Embedding batch size.
        output_dir:       Directory to save index files.
        device:           "cuda" or "cpu".

    Returns:
        Dict with bm25_index, faiss_index, metadata, stats.
    """
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[IndexBuilder] Building RAG index from {len(threads_df)} tweet turns")

    # Auto-detect cluster column if specified one is missing
    if cluster_col not in threads_df.columns:
        for cand in ["cluster_final", "cluster_id", "cluster", "cluster_k9"]:
            if cand in threads_df.columns:
                cluster_col = cand
                break

    # If intent_labels_df provided, map cluster_final onto threads_df
    if intent_labels_df is not None and thread_id_col in intent_labels_df.columns:
        lbl_cluster_col = "cluster_final" if "cluster_final" in intent_labels_df.columns else "cluster_id"
        if lbl_cluster_col in intent_labels_df.columns:
            cmap = dict(zip(intent_labels_df[thread_id_col].astype(str), intent_labels_df[lbl_cluster_col]))
            threads_df = threads_df.copy()
            threads_df["_mapped_cluster"] = threads_df[thread_id_col].astype(str).map(cmap)
            cluster_col = "_mapped_cluster"

    # --- Step 1: Group turns by thread ---
    if turn_pos_col in threads_df.columns:
        threads_df = threads_df.sort_values([thread_id_col, turn_pos_col])

    grouped = threads_df.groupby(thread_id_col)
    print(f"[IndexBuilder] Found {grouped.ngroups} unique threads")

    # --- Step 2: Build thread docs ---
    metadata: list[dict[str, Any]] = []
    thread_texts: list[str] = []
    n_bare_url_excluded = 0
    n_stitched = 0

    for thread_id, group in grouped:
        rows = group.to_dict("records")

        # Filter bare-URL brand turns (Option B: exclude from generative retrieval)
        filtered_rows = []
        for row in rows:
            text = str(row.get(text_col, ""))
            is_brand_turn = not row.get(author_col, True)  # inbound=True means customer
            if is_brand_turn and is_bare_url_tweet(text):
                n_bare_url_excluded += 1
                continue   # exclude bare-URL brand tweets
            filtered_rows.append(row)

        if not filtered_rows:
            continue  # skip threads that have no usable content after filtering

        # Stitch and prepare embedding text
        turns_for_stitch = [
            {
                "text":      str(r.get(text_col, "")),
                "author_id": r.get(author_col, True),   # True=customer, False=brand
                "inbound":   r.get(author_col, True),
            }
            for r in filtered_rows
        ]

        thread_text = prepare_thread_text(turns_for_stitch, max_turns=max_turns)
        if not thread_text.strip():
            continue

        # Track stitching
        stitched = stitch_turns(turns_for_stitch)
        if any(t.get("is_stitched") for t in stitched):
            n_stitched += 1

        # Get intent label
        first_row = rows[0]
        intent_label = ""
        if cluster_col in first_row and pd.notna(first_row.get(cluster_col)):
            raw_cluster = str(first_row.get(cluster_col, ""))
            intent_label = CLUSTER_TO_CLASS.get(raw_cluster, "Windows General")
        else:
            intent_label = "Windows General"

        # Escalation flag
        first_text = str(first_row.get(text_col, ""))
        esc_category = get_escalation_category(first_text)

        # Brand turns only (for reply grounding context)
        brand_turns = [
            str(r.get(text_col, ""))
            for r in filtered_rows
            if not r.get(author_col, True)  # False/0 = brand turn
        ]

        meta = {
            "thread_id":       str(thread_id),
            "text":            thread_text,
            "intent_label":    intent_label,
            "escalation_flag": esc_category is not None,
            "escalation_category": esc_category or "",
            "num_turns":       len(filtered_rows),
            "brand_turns":     brand_turns,
            "index_position":  len(metadata),
        }
        metadata.append(meta)
        thread_texts.append(thread_text)

    print(f"[IndexBuilder] Indexed threads:           {len(metadata)}")
    print(f"[IndexBuilder] Bare-URL turns excluded:   {n_bare_url_excluded}")
    print(f"[IndexBuilder] Threads with stitching:    {n_stitched}")

    # --- Step 3: Build BM25 index ---
    print("[IndexBuilder] Building BM25 index…")
    tokenized = [doc.lower().split() for doc in thread_texts]
    bm25_index = BM25Okapi(tokenized)

    bm25_path = out_path / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"[IndexBuilder] BM25 saved: {bm25_path}")

    # --- Step 4: Build FAISS index ---
    print(f"[IndexBuilder] Building FAISS index with {embedder_name}…")
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    embedder = SentenceTransformer(embedder_name, device=device)

    embeddings = embedder.encode(
        thread_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalize for cosine similarity via dot product
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    print(f"[IndexBuilder] Embeddings shape: {embeddings.shape} (dim={dim})")

    faiss_index = faiss.IndexFlatIP(dim)   # Inner product = cosine sim on normalized vecs
    faiss_index.add(embeddings)

    faiss_path = out_path / "faiss_index.bin"
    faiss.write_index(faiss_index, str(faiss_path))
    print(f"[IndexBuilder] FAISS saved: {faiss_path} ({faiss_index.ntotal} vectors)")

    # --- Step 5: Save metadata ---
    meta_path = out_path / "thread_metadata.json"
    # Ensure JSON-serializable (remove non-serializable types)
    for m in metadata:
        m["escalation_flag"] = bool(m["escalation_flag"])

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[IndexBuilder] Metadata saved: {meta_path}")

    # --- Step 6: Stats summary ---
    stats = {
        "n_threads_indexed":      len(metadata),
        "n_bare_url_excluded":    n_bare_url_excluded,
        "n_stitched_threads":     n_stitched,
        "faiss_dim":              dim,
        "faiss_vectors":          faiss_index.ntotal,
        "embedder":               embedder_name,
        "max_turns_cap":          max_turns,
        "intent_distribution":    {
            k: sum(1 for m in metadata if m["intent_label"] == k)
            for k in set(m["intent_label"] for m in metadata)
        },
        "n_escalation_threads":   sum(1 for m in metadata if m["escalation_flag"]),
    }

    stats_path = out_path / "index_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n[IndexBuilder] ✅ Index build complete!")
    print(f"  Threads indexed:    {stats['n_threads_indexed']}")
    print(f"  Escalation threads: {stats['n_escalation_threads']}")
    print(f"  FAISS dim:          {stats['faiss_dim']}")
    print(f"  Output dir:         {out_path}")

    return {
        "bm25_index":   bm25_index,
        "faiss_index":  faiss_index,
        "metadata":     metadata,
        "stats":        stats,
    }


# ---------------------------------------------------------------------------
# Colab entry point
# ---------------------------------------------------------------------------

def run_in_colab(
    threads_csv:       str = "/content/drive/MyDrive/hiver_data/ms_threads.csv",
    first_inbound_csv: Optional[str] = None,
    output_dir:        str = "/content/drive/MyDrive/hiver_index/",
    max_turns:         int = 6,
    batch_size:        int = 64,
) -> dict[str, Any]:
    """
    Colab-ready entry point. Call this cell in the notebook.

    Example:
        from data.build_rag_index import run_in_colab
        index_result = run_in_colab(threads_csv=..., first_inbound_csv=...)
    """
    print("[IndexBuilder] Loading thread data…")
    threads_df = pd.read_csv(threads_csv, low_memory=False)
    print(f"  threads_df: {threads_df.shape}")

    intent_df = None
    if first_inbound_csv and Path(first_inbound_csv).exists():
        print(f"[IndexBuilder] Loading intent mapping from {first_inbound_csv}…")
        intent_df = pd.read_csv(first_inbound_csv, low_memory=False)
    else:
        # Check if first_inbound_k9.csv is alongside threads_csv
        candidate_p = Path(threads_csv).parent / "microsofthelps_first_inbound_k9.csv"
        if candidate_p.exists():
            print(f"[IndexBuilder] Auto-detected intent mapping: {candidate_p}")
            intent_df = pd.read_csv(candidate_p, low_memory=False)

    return build_index(
        threads_df=threads_df,
        intent_labels_df=intent_df,
        max_turns=max_turns,
        batch_size=batch_size,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    result = run_in_colab(
        threads_csv="ms_threads.csv",
        output_dir="./hiver_index/",
    )
    print(result["stats"])
