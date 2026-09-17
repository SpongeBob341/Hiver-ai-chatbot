"""
Phase 3: Hybrid Trajectory Retrieval
--------------------------------------
BM25 (lexical) + FAISS (dense) retrieval over stitched full-thread corpus.

V1 Design decisions (locked, from knowledgebase §11, §13.1, §13.3):
  • Granularity:  Full Thread (≤6 turns, stitched)
  • Bare-URL:     Option B — bare-URL tweets flagged, excluded from retrieval
  • Retrieval:    BM25 (rank-bm25) + FAISS (all-MiniLM-L6-v2, 384-dim)
  • Scoping:      Filter by intent cluster before ranking (reduces noise)
  • Confidence:   Reciprocal Rank Fusion score → used by Phase 4 escalation gate

Index files (built by data/build_rag_index.py, saved to Google Drive / HF):
  • bm25_index.pkl      — serialized BM25Okapi object
  • faiss_index.bin     — FAISS flat L2 index
  • thread_metadata.json — [{thread_id, intent_label, text, escalation_flag, ...}]
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ThreadDoc:
    """Single indexed thread document."""
    thread_id:       str
    text:            str           # stitched, capped-at-6-turn string
    intent_label:    str
    escalation_flag: bool
    num_turns:       int
    brand_turns:     list[str]     # microsofthelps turns only (for reply grounding)
    metadata:        dict[str, Any] = field(default_factory=dict)

    def short_summary(self) -> str:
        """First ~200 chars for display."""
        return self.text[:200] + ("…" if len(self.text) > 200 else "")


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""
    thread:          ThreadDoc
    bm25_rank:       Optional[int]
    faiss_rank:      Optional[int]
    rrf_score:       float            # Reciprocal Rank Fusion score
    bm25_score:      Optional[float]
    faiss_distance:  Optional[float]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf(bm25_rank: Optional[int], faiss_rank: Optional[int], k: int = 60) -> float:
    """
    Compute RRF score for a document given its ranks in two retrieval lists.
    RRF(d) = Σ 1 / (k + rank_i)
    k=60 is the standard default (Cormack et al. 2009).
    """
    score = 0.0
    if bm25_rank is not None:
        score += 1.0 / (k + bm25_rank + 1)
    if faiss_rank is not None:
        score += 1.0 / (k + faiss_rank + 1)
    return score


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Hybrid BM25 + FAISS retriever over stitched microsofthelps threads.

    Usage:
        retriever = HybridRetriever.load("/content/drive/index/")
        results = retriever.retrieve(query, intent_label="Windows General", top_k=5)
        confidence = retriever.score_confidence(results)
    """

    def __init__(
        self,
        bm25_index,
        faiss_index,
        metadata: list[dict[str, Any]],
        embedder,
        rrf_k: int = 60,
    ) -> None:
        """
        Args:
            bm25_index:  BM25Okapi object (rank-bm25).
            faiss_index: FAISS index object.
            metadata:    List of thread metadata dicts, in the same order as the index.
            embedder:    SentenceTransformer model for query encoding.
            rrf_k:       RRF smoothing constant.
        """
        self.bm25      = bm25_index
        self.faiss     = faiss_index
        self.metadata  = metadata
        self.embedder  = embedder
        self.rrf_k     = rrf_k
        self._docs: list[ThreadDoc] = [
            ThreadDoc(
                thread_id=       m["thread_id"],
                text=            m["text"],
                intent_label=    m.get("intent_label", ""),
                escalation_flag= m.get("escalation_flag", False),
                num_turns=       m.get("num_turns", 0),
                brand_turns=     m.get("brand_turns", []),
                metadata=        m,
            )
            for m in metadata
        ]

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> "HybridRetriever":
        """
        Load a previously built index from disk.

        Args:
            index_dir:     Directory containing bm25_index.pkl, faiss_index.bin,
                           thread_metadata.json.
            embedder_name: SentenceTransformer model name or path.
            device:        "cuda", "cpu", or None (auto).
        """
        import faiss
        from sentence_transformers import SentenceTransformer

        index_dir = Path(index_dir)

        print(f"[HybridRetriever] Loading BM25 index from {index_dir / 'bm25_index.pkl'}")
        with open(index_dir / "bm25_index.pkl", "rb") as f:
            bm25_index = pickle.load(f)

        print(f"[HybridRetriever] Loading FAISS index from {index_dir / 'faiss_index.bin'}")
        faiss_index = faiss.read_index(str(index_dir / "faiss_index.bin"))

        print(f"[HybridRetriever] Loading metadata from {index_dir / 'thread_metadata.json'}")
        with open(index_dir / "thread_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print(f"[HybridRetriever] Loading embedder: {embedder_name}")
        embedder = SentenceTransformer(embedder_name, device=device)

        print(f"[HybridRetriever] ✅ Index loaded: {len(metadata)} threads")
        return cls(bm25_index, faiss_index, metadata, embedder)

    # ------------------------------------------------------------------
    def _bm25_retrieve(self, query: str, top_n: int) -> list[tuple[int, float]]:
        """Return [(doc_idx, score)] from BM25."""
        tokens = query.lower().split()
        if not tokens or not self._docs:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in ranked[:top_n] if 0 <= idx < len(self._docs)]

    def _faiss_retrieve(self, query: str, top_n: int) -> list[tuple[int, float]]:
        """Return [(doc_idx, score)] from FAISS."""
        if not self._docs or getattr(self.faiss, "ntotal", 0) <= 0:
            return []
        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)
        top_n = min(top_n, self.faiss.ntotal)
        if top_n <= 0:
            return []
        distances, indices = self.faiss.search(query_vec, top_n)
        hits: list[tuple[int, float]] = []
        for idx, dist in zip(indices[0].tolist(), distances[0].tolist()):
            if 0 <= idx < len(self._docs):
                hits.append((idx, dist))
        return hits

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        intent_label: Optional[str] = None,
        top_k: int = 5,
        bm25_candidates: int = 50,
        faiss_candidates: int = 50,
    ) -> list[RetrievalResult]:
        """
        Hybrid BM25 + FAISS retrieval with Reciprocal Rank Fusion.

        Args:
            query:            Incoming customer tweet (normalized).
            intent_label:     If provided, filter candidates to same intent class.
                              This scopes search to relevant historical threads.
            top_k:            Number of results to return after RRF fusion.
            bm25_candidates:  BM25 candidate pool size before RRF.
            faiss_candidates: FAISS candidate pool size before RRF.

        Returns:
            List of RetrievalResult, sorted by RRF score descending.
        """
        # --- Fetch raw candidates ---
        bm25_hits  = self._bm25_retrieve(query, bm25_candidates)
        faiss_hits = self._faiss_retrieve(query, faiss_candidates)

        # Build rank lookup: {doc_idx: rank}
        bm25_rank_map  = {idx: rank for rank, (idx, _) in enumerate(bm25_hits)}
        faiss_rank_map = {idx: rank for rank, (idx, _) in enumerate(faiss_hits)}
        faiss_dist_map = {idx: dist for idx, dist in faiss_hits}
        bm25_score_map = {idx: score for idx, score in bm25_hits}

        # Union of all candidate indices
        all_candidate_ids = set(bm25_rank_map.keys()) | set(faiss_rank_map.keys())

        # --- Apply intent filter (with fallback to all candidates if empty) ---
        candidate_ids = all_candidate_ids
        if intent_label:
            scoped_ids = {
                idx for idx in all_candidate_ids
                if self._docs[idx].intent_label == intent_label
            }
            if scoped_ids:
                candidate_ids = scoped_ids

        # --- RRF fusion ---
        scored: list[tuple[float, int]] = []
        for idx in candidate_ids:
            rrf_score = _rrf(
                bm25_rank_map.get(idx),
                faiss_rank_map.get(idx),
                k=self.rrf_k,
            )
            scored.append((rrf_score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_results = scored[:top_k]

        # --- Build result objects ---
        results: list[RetrievalResult] = []
        for rrf_score, idx in top_results:
            results.append(
                RetrievalResult(
                    thread=         self._docs[idx],
                    bm25_rank=      bm25_rank_map.get(idx),
                    faiss_rank=     faiss_rank_map.get(idx),
                    rrf_score=      rrf_score,
                    bm25_score=     bm25_score_map.get(idx),
                    faiss_distance= faiss_dist_map.get(idx),
                )
            )

        return results

    # ------------------------------------------------------------------
    def score_confidence(self, results: list[RetrievalResult]) -> float:
        """
        Compute an overall retrieval confidence score in [0, 1].

        Method: normalized RRF score of the top result.
        The max possible RRF score for top-1 (ranks 0,0) = 2 * 1/(60+1) ≈ 0.0328.
        We scale linearly: confidence = min(top_rrf / max_rrf, 1.0).

        Returns:
            Float in [0, 1]. Used by Phase 4 escalation gate.
        """
        if not results:
            return 0.0

        top_rrf = results[0].rrf_score
        max_possible_rrf = 2.0 / (self.rrf_k + 1)   # both lists rank it #1

        return min(top_rrf / max_possible_rrf, 1.0)

    # ------------------------------------------------------------------
    def format_context_for_drafter(
        self,
        results: list[RetrievalResult],
        max_threads: int = 3,
    ) -> str:
        """
        Format retrieved threads as a context block for the reply drafter (Phase 5).

        Returns a structured string:
            [EXAMPLE 1]
            Customer: ...
            microsofthelps: ...
            [EXAMPLE 2]
            ...
        """
        context_parts: list[str] = []
        for i, result in enumerate(results[:max_threads]):
            doc = result.thread
            context_parts.append(
                f"[EXAMPLE {i+1}] (Relevance: {result.rrf_score:.4f})\n"
                f"{doc.text}\n"
            )
        return "\n---\n".join(context_parts)
