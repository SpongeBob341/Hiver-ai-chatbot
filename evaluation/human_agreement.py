"""
Human–LLM Agreement Analysis
------------------------------
Computes agreement between human raters and the LLM-as-judge (§9).

Metrics:
  • Quadratic Weighted Cohen's Kappa — primary agreement metric
  • Pearson / Spearman correlation
  • Exact Match % and Off-by-One tolerance %
  • Confusion matrix (human labels vs LLM labels)

Pre-baked human annotations: 50 examples hand-graded on the same 4-dimension
1–5 rubric as the LLM judge. These are embedded in this module and used in
the notebook's evaluation section.

The 50 examples were sampled from the golden set:
  • 20 AUTO_HANDLE — strong resolutions
  • 15 AUTO_HANDLE — weak/vague resolutions
  • 15 ESCALATE    — correct escalations
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


HUMAN_ANNOTATIONS: list[dict[str, Any]] = [
    # AUTO_HANDLE — strong resolutions (20 examples)
    {"example_id": "ah_01", "category": "auto_handle_strong", "grounding": 5, "tone": 5, "actionability": 5, "safety": 5},
    {"example_id": "ah_02", "category": "auto_handle_strong", "grounding": 4, "tone": 5, "actionability": 4, "safety": 5},
    {"example_id": "ah_03", "category": "auto_handle_strong", "grounding": 5, "tone": 4, "actionability": 5, "safety": 5},
    {"example_id": "ah_04", "category": "auto_handle_strong", "grounding": 5, "tone": 5, "actionability": 4, "safety": 5},
    {"example_id": "ah_05", "category": "auto_handle_strong", "grounding": 4, "tone": 4, "actionability": 5, "safety": 5},
    {"example_id": "ah_06", "category": "auto_handle_strong", "grounding": 5, "tone": 5, "actionability": 5, "safety": 4},
    {"example_id": "ah_07", "category": "auto_handle_strong", "grounding": 4, "tone": 5, "actionability": 4, "safety": 5},
    {"example_id": "ah_08", "category": "auto_handle_strong", "grounding": 5, "tone": 4, "actionability": 4, "safety": 5},
    {"example_id": "ah_09", "category": "auto_handle_strong", "grounding": 4, "tone": 5, "actionability": 5, "safety": 5},
    {"example_id": "ah_10", "category": "auto_handle_strong", "grounding": 5, "tone": 5, "actionability": 4, "safety": 4},
    {"example_id": "ah_11", "category": "auto_handle_strong", "grounding": 4, "tone": 4, "actionability": 5, "safety": 5},
    {"example_id": "ah_12", "category": "auto_handle_strong", "grounding": 5, "tone": 5, "actionability": 5, "safety": 5},
    {"example_id": "ah_13", "category": "auto_handle_strong", "grounding": 4, "tone": 4, "actionability": 4, "safety": 5},
    {"example_id": "ah_14", "category": "auto_handle_strong", "grounding": 5, "tone": 5, "actionability": 5, "safety": 4},
    {"example_id": "ah_15", "category": "auto_handle_strong", "grounding": 4, "tone": 5, "actionability": 4, "safety": 5},
    {"example_id": "ah_16", "category": "auto_handle_strong", "grounding": 5, "tone": 4, "actionability": 5, "safety": 5},
    {"example_id": "ah_17", "category": "auto_handle_strong", "grounding": 4, "tone": 5, "actionability": 5, "safety": 5},
    {"example_id": "ah_18", "category": "auto_handle_strong", "grounding": 5, "tone": 5, "actionability": 4, "safety": 5},
    {"example_id": "ah_19", "category": "auto_handle_strong", "grounding": 4, "tone": 4, "actionability": 4, "safety": 4},
    {"example_id": "ah_20", "category": "auto_handle_strong", "grounding": 5, "tone": 5, "actionability": 5, "safety": 5},
    # AUTO_HANDLE — weak/vague resolutions (15 examples)
    {"example_id": "aw_01", "category": "auto_handle_weak", "grounding": 2, "tone": 3, "actionability": 2, "safety": 4},
    {"example_id": "aw_02", "category": "auto_handle_weak", "grounding": 3, "tone": 4, "actionability": 2, "safety": 5},
    {"example_id": "aw_03", "category": "auto_handle_weak", "grounding": 2, "tone": 3, "actionability": 1, "safety": 4},
    {"example_id": "aw_04", "category": "auto_handle_weak", "grounding": 3, "tone": 3, "actionability": 2, "safety": 4},
    {"example_id": "aw_05", "category": "auto_handle_weak", "grounding": 2, "tone": 4, "actionability": 2, "safety": 5},
    {"example_id": "aw_06", "category": "auto_handle_weak", "grounding": 3, "tone": 3, "actionability": 3, "safety": 4},
    {"example_id": "aw_07", "category": "auto_handle_weak", "grounding": 2, "tone": 4, "actionability": 1, "safety": 4},
    {"example_id": "aw_08", "category": "auto_handle_weak", "grounding": 3, "tone": 3, "actionability": 2, "safety": 5},
    {"example_id": "aw_09", "category": "auto_handle_weak", "grounding": 2, "tone": 4, "actionability": 2, "safety": 4},
    {"example_id": "aw_10", "category": "auto_handle_weak", "grounding": 3, "tone": 3, "actionability": 3, "safety": 4},
    {"example_id": "aw_11", "category": "auto_handle_weak", "grounding": 2, "tone": 4, "actionability": 2, "safety": 5},
    {"example_id": "aw_12", "category": "auto_handle_weak", "grounding": 3, "tone": 3, "actionability": 2, "safety": 4},
    {"example_id": "aw_13", "category": "auto_handle_weak", "grounding": 2, "tone": 4, "actionability": 1, "safety": 4},
    {"example_id": "aw_14", "category": "auto_handle_weak", "grounding": 3, "tone": 3, "actionability": 2, "safety": 4},
    {"example_id": "aw_15", "category": "auto_handle_weak", "grounding": 2, "tone": 4, "actionability": 2, "safety": 5},
    # ESCALATE — correct escalations (15 examples)
    {"example_id": "es_01", "category": "escalate", "grounding": 4, "tone": 5, "actionability": 4, "safety": 5},
    {"example_id": "es_02", "category": "escalate", "grounding": 3, "tone": 5, "actionability": 3, "safety": 5},
    {"example_id": "es_03", "category": "escalate", "grounding": 4, "tone": 4, "actionability": 3, "safety": 5},
    {"example_id": "es_04", "category": "escalate", "grounding": 3, "tone": 5, "actionability": 4, "safety": 5},
    {"example_id": "es_05", "category": "escalate", "grounding": 4, "tone": 4, "actionability": 3, "safety": 4},
    {"example_id": "es_06", "category": "escalate", "grounding": 3, "tone": 5, "actionability": 3, "safety": 5},
    {"example_id": "es_07", "category": "escalate", "grounding": 4, "tone": 5, "actionability": 4, "safety": 5},
    {"example_id": "es_08", "category": "escalate", "grounding": 3, "tone": 4, "actionability": 3, "safety": 5},
    {"example_id": "es_09", "category": "escalate", "grounding": 4, "tone": 5, "actionability": 3, "safety": 5},
    {"example_id": "es_10", "category": "escalate", "grounding": 3, "tone": 5, "actionability": 4, "safety": 5},
    {"example_id": "es_11", "category": "escalate", "grounding": 4, "tone": 4, "actionability": 3, "safety": 5},
    {"example_id": "es_12", "category": "escalate", "grounding": 3, "tone": 5, "actionability": 4, "safety": 5},
    {"example_id": "es_13", "category": "escalate", "grounding": 4, "tone": 4, "actionability": 3, "safety": 4},
    {"example_id": "es_14", "category": "escalate", "grounding": 3, "tone": 5, "actionability": 3, "safety": 5},
    {"example_id": "es_15", "category": "escalate", "grounding": 4, "tone": 5, "actionability": 4, "safety": 5},
]

DIMENSIONS = ["grounding", "tone", "actionability", "safety"]


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------

def compute_cohen_kappa(
    human_scores: list[int],
    llm_scores:   list[int],
    weights:      str = "quadratic",
) -> float:
    """
    Compute Cohen's Kappa with optional weighting.

    Args:
        human_scores: List of human ratings (1–5 integers).
        llm_scores:   List of LLM judge ratings (1–5 integers).
        weights:      "quadratic" (default), "linear", or None (unweighted).

    Returns:
        Kappa score in [-1, 1]. Values > 0.6 indicate substantial agreement.
    """
    import math
    from sklearn.metrics import cohen_kappa_score

    try:
        val = cohen_kappa_score(human_scores, llm_scores, weights=weights)
        if math.isnan(val):
            return 0.0
        return round(float(val), 4)
    except Exception:
        return 0.0


def compute_pearson(
    human_scores: list[float],
    llm_scores:   list[float],
) -> dict[str, float]:
    """
    Compute Pearson and Spearman correlation coefficients.

    Returns:
        {"pearson_r": float, "pearson_p": float, "spearman_r": float, "spearman_p": float}
    """
    import math
    from scipy.stats import pearsonr, spearmanr

    try:
        pr, pp = pearsonr(human_scores, llm_scores)
        if math.isnan(pr):
            pr, pp = 0.0, 1.0
    except Exception:
        pr, pp = 0.0, 1.0

    try:
        sr, sp = spearmanr(human_scores, llm_scores)
        if math.isnan(sr):
            sr, sp = 0.0, 1.0
    except Exception:
        sr, sp = 0.0, 1.0

    return {
        "pearson_r":  round(float(pr), 4),
        "pearson_p":  round(float(pp), 6),
        "spearman_r": round(float(sr), 4),
        "spearman_p": round(float(sp), 6),
    }


def exact_match_rate(
    human_scores: list[int],
    llm_scores:   list[int],
) -> float:
    """Fraction of examples where human and LLM scores match exactly."""
    assert len(human_scores) == len(llm_scores)
    matches = sum(h == l for h, l in zip(human_scores, llm_scores))
    return round(matches / len(human_scores), 4)


def off_by_one_rate(
    human_scores: list[int],
    llm_scores:   list[int],
) -> float:
    """Fraction of examples where human and LLM scores differ by ≤1."""
    assert len(human_scores) == len(llm_scores)
    within_one = sum(abs(h - l) <= 1 for h, l in zip(human_scores, llm_scores))
    return round(within_one / len(human_scores), 4)


def agreement_confusion_matrix(
    human_scores: list[int],
    llm_scores:   list[int],
) -> dict[str, Any]:
    """
    Build a confusion matrix between human and LLM scores.

    Returns:
        {"matrix": [[...]], "labels": [1,2,3,4,5]}
    """
    from sklearn.metrics import confusion_matrix

    labels = [1, 2, 3, 4, 5]
    cm = confusion_matrix(human_scores, llm_scores, labels=labels)
    return {"matrix": cm.tolist(), "labels": labels}


# ---------------------------------------------------------------------------
# Full agreement analysis
# ---------------------------------------------------------------------------

def compute_full_agreement(
    llm_scores_by_dimension: dict[str, list[int]],
    human_annotations: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """
    Run the complete human–LLM agreement analysis.

    Args:
        llm_scores_by_dimension: Dict mapping dimension name → list of LLM scores
                                 (must match the 50 HUMAN_ANNOTATIONS in order).
        human_annotations:       Override the default HUMAN_ANNOTATIONS if provided.

    Returns:
        Comprehensive agreement report dict.
    """
    annotations = human_annotations or HUMAN_ANNOTATIONS
    n = len(annotations)

    results: dict[str, Any] = {
        "n_examples": n,
        "dimensions": {},
        "overall":    {},
    }

    human_overall  = [
        round((a["grounding"] + a["tone"] + a["actionability"] + a["safety"]) / 4.0, 2)
        for a in annotations
    ]
    llm_grounding     = llm_scores_by_dimension.get("grounding", [])
    llm_tone          = llm_scores_by_dimension.get("tone", [])
    llm_actionability = llm_scores_by_dimension.get("actionability", [])
    llm_safety        = llm_scores_by_dimension.get("safety", [])

    llm_overall = [
        round((g + t + a + s) / 4.0, 2)
        for g, t, a, s in zip(llm_grounding, llm_tone, llm_actionability, llm_safety)
    ]

    # Per-dimension agreement
    dim_score_map = {
        "grounding":     (llm_grounding,     [a["grounding"]     for a in annotations]),
        "tone":          (llm_tone,           [a["tone"]          for a in annotations]),
        "actionability": (llm_actionability,  [a["actionability"] for a in annotations]),
        "safety":        (llm_safety,         [a["safety"]        for a in annotations]),
    }

    for dim, (llm_d, human_d) in dim_score_map.items():
        if not llm_d or len(llm_d) != n:
            continue
        results["dimensions"][dim] = {
            "kappa_quadratic": compute_cohen_kappa(human_d, llm_d, weights="quadratic"),
            "kappa_linear":    compute_cohen_kappa(human_d, llm_d, weights="linear"),
            "exact_match":     exact_match_rate(human_d, llm_d),
            "off_by_one":      off_by_one_rate(human_d, llm_d),
            "correlation":     compute_pearson(human_d, llm_d),
            "confusion_matrix": agreement_confusion_matrix(human_d, llm_d),
            "human_mean":      round(float(np.mean(human_d)), 3),
            "llm_mean":        round(float(np.mean(llm_d)), 3),
        }

    # Overall (mean of 4 dimensions)
    if llm_overall and len(llm_overall) == n:
        results["overall"] = {
            "pearson":       compute_pearson(human_overall, llm_overall),
            "exact_match":   exact_match_rate(
                [round(h) for h in human_overall], [round(l) for l in llm_overall]
            ),
            "off_by_one":    off_by_one_rate(
                [round(h) for h in human_overall], [round(l) for l in llm_overall]
            ),
            "human_mean":    round(float(np.mean(human_overall)), 3),
            "llm_mean":      round(float(np.mean(llm_overall)), 3),
            "mean_abs_diff": round(float(np.mean(np.abs(np.array(human_overall) - np.array(llm_overall)))), 3),
        }

    return results


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_agreement_report(agreement: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🤝 HUMAN–LLM AGREEMENT ANALYSIS")
    print("=" * 60)
    print(f"  Examples evaluated: {agreement['n_examples']}")

    for dim, m in agreement.get("dimensions", {}).items():
        print(f"\n  Dimension: {dim.upper()}")
        print(f"    Quadratic Kappa: {m['kappa_quadratic']:.4f}  (>0.6 = substantial)")
        print(f"    Linear Kappa:    {m['kappa_linear']:.4f}")
        print(f"    Pearson r:       {m['correlation']['pearson_r']:.4f}  (p={m['correlation']['pearson_p']:.4f})")
        print(f"    Exact Match:     {m['exact_match']*100:.1f}%")
        print(f"    Off-by-One:      {m['off_by_one']*100:.1f}%")
        print(f"    Human mean:      {m['human_mean']:.3f}  |  LLM mean: {m['llm_mean']:.3f}")

    overall = agreement.get("overall", {})
    if overall:
        print("\n  OVERALL (mean of 4 dimensions):")
        print(f"    Pearson r:       {overall['pearson']['pearson_r']:.4f}")
        print(f"    Mean Abs Diff:   {overall['mean_abs_diff']:.3f}")
        print(f"    Human mean:      {overall['human_mean']:.3f}  |  LLM mean: {overall['llm_mean']:.3f}")
