"""
Evaluation Metrics
------------------
Automated metrics for all three system components:

  1. Intent Classification — Macro/Micro F1, Precision, Recall, Confusion Matrix
  2. Escalation Decision   — Precision, Recall, False Negative Rate (FNR)
  3. Retrieval Relevance   — Hit@k, MRR (Mean Reciprocal Rank)
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. Intent Classification Metrics
# ---------------------------------------------------------------------------

INTENT_LABELS = [
    "Windows General",
    "Windows Update & Install",
    "Office & Productivity",
    "Account & Sign-In",
    "Surface Hardware",
    "BSOD & Crash / Distress",
    "Xbox & Gaming",
    "Support Channel Navigation",
]


def compute_classification_metrics(
    y_true: list[int | str],
    y_pred: list[int | str],
    labels: Optional[list] = None,
    label_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Compute classification metrics for the intent classifier.

    Args:
        y_true:       Ground-truth intent labels (int ids or string labels).
        y_pred:       Predicted intent labels.
        labels:       Label ids to include (default: all).
        label_names:  Human-readable names for the labels.

    Returns:
        Dict with macro_f1, micro_f1, weighted_f1, per_class metrics,
        confusion_matrix, and classification_report string.
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    label_names = label_names or INTENT_LABELS

    macro_f1    = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    micro_f1    = f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)

    macro_precision = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    macro_recall    = recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)

    cm     = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    # Per-class breakdown
    per_class_f1    = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_prec  = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_rec   = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    # Get unique labels in order
    unique_labels = labels if labels is not None else sorted(set(list(y_true) + list(y_pred)))
    per_class: dict[str, dict] = {}
    for i, label in enumerate(unique_labels):
        name = label_names[label] if isinstance(label, int) and label < len(label_names) else str(label)
        if i < len(per_class_f1):
            per_class[name] = {
                "f1":        round(float(per_class_f1[i]), 4),
                "precision": round(float(per_class_prec[i]), 4),
                "recall":    round(float(per_class_rec[i]), 4),
            }

    return {
        "macro_f1":          round(float(macro_f1), 4),
        "micro_f1":          round(float(micro_f1), 4),
        "weighted_f1":       round(float(weighted_f1), 4),
        "macro_precision":   round(float(macro_precision), 4),
        "macro_recall":      round(float(macro_recall), 4),
        "per_class":         per_class,
        "confusion_matrix":  cm.tolist(),
        "classification_report": report,
        "n_samples":         len(y_true),
    }


# ---------------------------------------------------------------------------
# 2. Escalation Decision Metrics
# ---------------------------------------------------------------------------

def compute_escalation_metrics(
    y_true_escalate: list[bool | int],
    y_pred_escalate: list[bool | int],
) -> dict[str, float]:
    """
    Compute binary escalation decision metrics.

    CRITICAL: False Negative Rate (FNR) is the most important metric here.
    Missing a real escalation is high-cost (e.g., missing a data loss or
    billing fraud case).

    Args:
        y_true_escalate: Ground-truth escalation flags (True/1 = should escalate).
        y_pred_escalate: Predicted escalation flags.

    Returns:
        Dict with precision, recall, f1, fnr, fpr, accuracy.
    """
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    precision = precision_score(y_true_escalate, y_pred_escalate, zero_division=0)
    recall    = recall_score(y_true_escalate, y_pred_escalate, zero_division=0)
    f1        = f1_score(y_true_escalate, y_pred_escalate, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_true_escalate, y_pred_escalate, labels=[0, 1]
    ).ravel()

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # False Negative Rate (CRITICAL)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False Positive Rate
    accuracy = (tp + tn) / len(y_true_escalate)

    n_true_escalations = sum(int(x) for x in y_true_escalate)
    n_pred_escalations = sum(int(x) for x in y_pred_escalate)

    return {
        "precision":           round(float(precision), 4),
        "recall":              round(float(recall), 4),
        "f1":                  round(float(f1), 4),
        "fnr":                 round(float(fnr), 4),     # ← most critical
        "fpr":                 round(float(fpr), 4),
        "accuracy":            round(float(accuracy), 4),
        "tp":                  int(tp),
        "fp":                  int(fp),
        "fn":                  int(fn),                  # ← missed escalations
        "tn":                  int(tn),
        "n_true_escalations":  n_true_escalations,
        "n_pred_escalations":  n_pred_escalations,
        "n_samples":           len(y_true_escalate),
    }


# ---------------------------------------------------------------------------
# 3. Retrieval Relevance Metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    queries:       list[str],
    retrieved_ids: list[list[str]],   # retrieved thread_ids per query (ordered by rank)
    relevant_ids:  list[list[str]],   # ground-truth relevant thread_ids per query
    k_values:      list[int] = [1, 3, 5],
) -> dict[str, float]:
    """
    Compute retrieval relevance metrics.

    Args:
        queries:       List of query strings (for reporting).
        retrieved_ids: For each query, ordered list of retrieved thread_ids.
        relevant_ids:  For each query, list of thread_ids considered relevant.
        k_values:      k values for Hit@k.

    Returns:
        Dict with hit@k for each k, MRR, and per-query details.
    """
    assert len(retrieved_ids) == len(relevant_ids), "Mismatch in retrieved/relevant lists"

    n = len(retrieved_ids)
    hit_at_k: dict[int, float] = {k: 0.0 for k in k_values}
    reciprocal_ranks: list[float] = []

    per_query: list[dict] = []
    for i, (ret_ids, rel_ids) in enumerate(zip(retrieved_ids, relevant_ids)):
        rel_set = set(rel_ids)

        # Hit@k
        hits: dict[int, bool] = {}
        for k in k_values:
            hits[k] = any(r in rel_set for r in ret_ids[:k])
            if hits[k]:
                hit_at_k[k] += 1

        # Reciprocal Rank
        rr = 0.0
        for rank, ret_id in enumerate(ret_ids, start=1):
            if ret_id in rel_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        per_query.append({
            "query_idx": i,
            "query":     queries[i][:80] if queries else "",
            "hits":      hits,
            "rr":        round(rr, 4),
        })

    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    return {
        **{f"hit@{k}": round(hit_at_k[k] / n, 4) for k in k_values},
        "mrr":       round(mrr, 4),
        "n_queries": n,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------

def print_classification_report(metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📊 INTENT CLASSIFICATION METRICS")
    print("=" * 60)
    print(f"  Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"  Micro F1:      {metrics['micro_f1']:.4f}")
    print(f"  Weighted F1:   {metrics['weighted_f1']:.4f}")
    print(f"  Precision:     {metrics['macro_precision']:.4f}")
    print(f"  Recall:        {metrics['macro_recall']:.4f}")
    print(f"  Samples:       {metrics['n_samples']}")
    print("\nPer-class breakdown:")
    for cls_name, m in metrics["per_class"].items():
        print(f"  {cls_name:<35} F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
    print("\nFull Classification Report:")
    print(metrics["classification_report"])


def print_escalation_report(metrics: dict[str, float]) -> None:
    print("\n" + "=" * 60)
    print("🚨 ESCALATION DECISION METRICS")
    print("=" * 60)
    print(f"  Precision:             {metrics['precision']:.4f}")
    print(f"  Recall:                {metrics['recall']:.4f}")
    print(f"  F1:                    {metrics['f1']:.4f}")
    print(f"  ⚠️  FNR (CRITICAL):    {metrics['fnr']:.4f}  ← missed escalations")
    print(f"  FPR:                   {metrics['fpr']:.4f}")
    print(f"  Accuracy:              {metrics['accuracy']:.4f}")
    print(f"  TP / FP / FN / TN:    {metrics['tp']} / {metrics['fp']} / {metrics['fn']} / {metrics['tn']}")
    print(f"  True escalations:      {metrics['n_true_escalations']} / {metrics['n_samples']}")


def print_retrieval_report(metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🔍 RETRIEVAL RELEVANCE METRICS")
    print("=" * 60)
    for key, val in metrics.items():
        if key.startswith("hit@") or key == "mrr":
            print(f"  {key.upper():<12}: {val:.4f}")
    print(f"  Queries evaluated: {metrics['n_queries']}")
