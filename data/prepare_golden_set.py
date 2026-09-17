"""
Golden Evaluation Set Builder
-------------------------------
Produces a stratified 250-thread golden evaluation set from the microsofthelps corpus.

Sampling strategy (knowledgebase §10):
  Windows General   (C1_general_os + C8)      → 60 threads
  Windows Update    (C6 + C7)                 → 40 threads
  Office            (C1_office_365)            → 25 threads
  Account & Sign-In (C5 + C1_email_account)   → 25 threads
  Surface Hardware  (C4)                       → 20 threads
  BSOD & Distress   (C2 + C0)                 → 30 threads
  Xbox & Gaming     (C1_xbox_gaming)           → 15 threads
  Escalation overlay (stratified over above)  → +35 threads
                                                ─────────────
                                                 ~250 threads

Output files (saved to Google Drive in Colab):
  golden_set_250.csv   — thread-level, with ground-truth intent + escalation label
  golden_set_labels.csv — compact label summary for evaluation harness

Run this script as a notebook cell; data loaded from Google Drive or mounted CSV.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Intent cluster → production class mapping
# Mirrors knowledgebase §7.2 (7 collapsed classes + escalation overlay)
# ---------------------------------------------------------------------------

CLUSTER_TO_CLASS: dict[str, str] = {
    "C1_general_os":  "Windows General",
    "C8":             "Windows General",
    "C1_store_app":   "Windows General",        # too small, merged
    "C6":             "Windows Update & Install",
    "C7":             "Windows Update & Install",
    "C1_office_365":  "Office & Productivity",
    "C5":             "Account & Sign-In",
    "C1_email_account": "Account & Sign-In",
    "C4":             "Surface Hardware",
    "C2":             "BSOD & Crash / Distress",
    "C0":             "BSOD & Crash / Distress",
    "C1_xbox_gaming": "Xbox & Gaming",
    "C3":             "Support Channel Navigation",  # routing class
}

# Number of examples to sample per production class
CLASS_SAMPLE_SIZES: dict[str, int] = {
    "Windows General":            60,
    "Windows Update & Install":   40,
    "Office & Productivity":      25,
    "Account & Sign-In":          25,
    "Surface Hardware":           20,
    "BSOD & Crash / Distress":    30,
    "Xbox & Gaming":              15,
    "Support Channel Navigation":  0,   # routing only, not evaluated in golden set
}

# Escalation overlay sizes (stratified across categories)
ESCALATION_OVERLAY: dict[str, int] = {
    "BSOD / System Crash":       15,
    "Payment / Billing Dispute": 10,
    "Account Lockout / Hack":     7,
    "Data Loss / Corruption":     3,  # sparse — take all available
}


# ---------------------------------------------------------------------------
# Escalation keyword scanner (mirrors Phase 4 escalation gate)
# ---------------------------------------------------------------------------

_ESC_PATTERNS: dict[str, list[re.Pattern]] = {
    "BSOD / System Crash": [re.compile(p, re.IGNORECASE) for p in [
        r"\bblue\s*screen\b", r"\bbsod\b", r"\bblack\s*screen\b",
        r"\bsystem\s*crash\b", r"\bcrash(?:ed|ing)?\b", r"\bfreezing\b",
        r"\b0x[0-9a-fA-F]{6,}\b",
    ]],
    "Payment / Billing Dispute": [re.compile(p, re.IGNORECASE) for p in [
        r"\bbill(?:ing|ed)?\b", r"\bcharged?\b", r"\bpayment\b",
        r"\brefund\b", r"\bfraud\b", r"\bcredit\s*card\b",
    ]],
    "Account Lockout / Hack": [re.compile(p, re.IGNORECASE) for p in [
        r"\bhacked?\b", r"\bcompromised\b", r"\blocke?d?\s*out\b",
        r"\bphish(?:ing)?\b", r"\bunauthori[sz]ed\s*access\b",
    ]],
    "Data Loss / Corruption": [re.compile(p, re.IGNORECASE) for p in [
        r"\bdata\s*loss\b", r"\bfiles?\s*(?:deleted|lost|gone)\b",
        r"\bdata\s*corrupted?\b", r"\bdisk\s*(?:error|fail)\b",
    ]],
}


def get_escalation_category(text: str) -> Optional[str]:
    """Return the first escalation category that matches, or None."""
    for category, patterns in _ESC_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                return category
    return None


# ---------------------------------------------------------------------------
# Golden set builder
# ---------------------------------------------------------------------------

def build_golden_set(
    threads_df:         pd.DataFrame,
    first_inbound_df:   pd.DataFrame,
    cluster_col:        str = "cluster_final",
    thread_id_col:      str = "thread_id",
    text_col:           str = "text",
    seed:               int = 42,
    output_dir:         Optional[str] = None,
) -> pd.DataFrame:
    """
    Build the 250-thread stratified golden evaluation set.

    Args:
        threads_df:       Full thread DataFrame (ms_threads.csv or equivalent).
                          Expected columns: thread_id, text (per turn), cluster_final.
        first_inbound_df: First inbound tweet per thread (microsofthelps_first_inbound_k9.csv).
                          Expected columns: thread_id, text, cluster_id.
        cluster_col:      Column with the cluster/intent label.
        thread_id_col:    Column with thread identifier.
        text_col:         Column with tweet text.
        seed:             Random seed for reproducibility.
        output_dir:       If provided, save golden_set_250.csv and golden_set_labels.csv here.

    Returns:
        golden_set_df: DataFrame with 250 rows, one per sampled thread.
    """
    rng = np.random.default_rng(seed)

    # --- Step 1: Map cluster IDs to production class names ---
    if cluster_col not in first_inbound_df.columns:
        for cand in ["cluster_final", "cluster_id", "cluster", "cluster_k9"]:
            if cand in first_inbound_df.columns:
                cluster_col = cand
                break
        else:
            raise ValueError(f"Column '{cluster_col}' not found. Available: {first_inbound_df.columns.tolist()}")

    inbound = first_inbound_df.copy()
    inbound["intent_class"] = inbound[cluster_col].map(CLUSTER_TO_CLASS).fillna("Windows General")

    # --- Step 2: Add escalation labels ---
    inbound["escalation_category"] = inbound[text_col].apply(get_escalation_category)
    inbound["is_escalation"] = inbound["escalation_category"].notna()

    print(f"[GoldenSet] Corpus: {len(inbound)} threads")
    print(f"[GoldenSet] Escalation threads: {inbound['is_escalation'].sum()} "
          f"({inbound['is_escalation'].mean()*100:.1f}%)")

    # --- Step 3: Sample by production class ---
    selected_ids: set[str] = set()
    class_samples: list[pd.DataFrame] = []

    for cls, n in CLASS_SAMPLE_SIZES.items():
        if n == 0:
            continue
        pool = inbound[inbound["intent_class"] == cls]
        if len(pool) == 0:
            print(f"  [!] Class '{cls}' has 0 threads — skipping.")
            continue

        # Prefer multi-turn threads (≥3 turns) where possible
        multi_turn_pool = pool[pool[thread_id_col].isin(
            threads_df.groupby(thread_id_col).size()[
                threads_df.groupby(thread_id_col).size() >= 3
            ].index
        )] if thread_id_col in threads_df.columns else pool

        if len(multi_turn_pool) >= n:
            sampled = multi_turn_pool.sample(n=n, random_state=seed, replace=False)
        else:
            sampled = pool.sample(n=min(n, len(pool)), random_state=seed, replace=False)
            print(f"  [!] Class '{cls}': only {len(pool)} available, sampled {len(sampled)}")

        selected_ids.update(sampled[thread_id_col].tolist())
        class_samples.append(sampled)
        print(f"  ✅ {cls}: {len(sampled)} threads sampled")

    base_df = pd.concat(class_samples, ignore_index=True)

    # --- Step 4: Escalation overlay ---
    # Sample additional escalation threads not already in base set
    not_selected = inbound[~inbound[thread_id_col].isin(selected_ids)]
    overlay_samples: list[pd.DataFrame] = []

    for category, n in ESCALATION_OVERLAY.items():
        esc_pool = not_selected[
            not_selected["escalation_category"] == category
        ]
        if len(esc_pool) == 0:
            print(f"  [!] Escalation overlay '{category}': 0 threads available")
            continue
        n_take = min(n, len(esc_pool))
        sampled = esc_pool.sample(n=n_take, random_state=seed, replace=False)
        overlay_samples.append(sampled)
        selected_ids.update(sampled[thread_id_col].tolist())
        print(f"  🔴 Escalation overlay '{category}': {n_take} threads")

    if overlay_samples:
        overlay_df = pd.concat(overlay_samples, ignore_index=True)
        golden_df = pd.concat([base_df, overlay_df], ignore_index=True)
    else:
        golden_df = base_df

    # Drop duplicates (a thread may appear in both class sample and escalation overlay)
    golden_df = golden_df.drop_duplicates(subset=[thread_id_col])
    print(f"\n[GoldenSet] ✅ Total: {len(golden_df)} threads in golden set")

    # --- Step 5: Add full thread text ---
    if thread_id_col in threads_df.columns:
        # Aggregate all turns per thread into a single string
        thread_texts = (
            threads_df
            .sort_values([thread_id_col, "turn_pos"] if "turn_pos" in threads_df.columns else thread_id_col)
            .groupby(thread_id_col)[text_col]
            .apply(lambda turns: "\n".join(turns.astype(str)))
            .reset_index()
            .rename(columns={text_col: "full_thread_text"})
        )
        golden_df = golden_df.merge(thread_texts, on=thread_id_col, how="left")

    # --- Step 6: Final column selection ---
    keep_cols = [
        thread_id_col,
        text_col,           # first inbound tweet text
        "intent_class",
        "is_escalation",
        "escalation_category",
    ]
    if "full_thread_text" in golden_df.columns:
        keep_cols.append("full_thread_text")

    golden_df = golden_df[[c for c in keep_cols if c in golden_df.columns]].copy()
    golden_df = golden_df.reset_index(drop=True)
    golden_df["example_id"] = [f"ex_{i:04d}" for i in range(len(golden_df))]

    # --- Step 7: Save ---
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        golden_path = out_path / "golden_set_250.csv"
        golden_df.to_csv(golden_path, index=False)
        print(f"[GoldenSet] Saved: {golden_path}")

        # Compact label CSV for evaluation harness
        label_cols = [thread_id_col, "example_id", "intent_class", "is_escalation", "escalation_category"]
        label_df = golden_df[[c for c in label_cols if c in golden_df.columns]]
        label_path = out_path / "golden_set_labels.csv"
        label_df.to_csv(label_path, index=False)
        print(f"[GoldenSet] Saved: {label_path}")

    # --- Step 8: Summary ---
    print("\n[GoldenSet] Class distribution:")
    print(golden_df["intent_class"].value_counts().to_string())
    print(f"\n[GoldenSet] Escalation breakdown:")
    print(golden_df[golden_df["is_escalation"]]["escalation_category"].value_counts().to_string())

    return golden_df


# ---------------------------------------------------------------------------
# Colab entry point
# ---------------------------------------------------------------------------

def run_in_colab(
    threads_csv:       str = "/content/drive/MyDrive/hiver_data/ms_threads.csv",
    first_inbound_csv: str = "/content/drive/MyDrive/hiver_data/microsofthelps_first_inbound_k9.csv",
    output_dir:        str = "/content/drive/MyDrive/hiver_data/golden_set/",
    seed:              int = 42,
) -> pd.DataFrame:
    """
    Colab-ready entry point. Call this cell in the notebook.

    Example:
        from data.prepare_golden_set import run_in_colab
        golden_df = run_in_colab()
    """
    print("[GoldenSet] Loading data…")
    threads_df    = pd.read_csv(threads_csv, low_memory=False)
    first_inbound = pd.read_csv(first_inbound_csv, low_memory=False)
    print(f"  threads_df:    {threads_df.shape}")
    print(f"  first_inbound: {first_inbound.shape}")

    golden_df = build_golden_set(
        threads_df=threads_df,
        first_inbound_df=first_inbound,
        output_dir=output_dir,
        seed=seed,
    )
    return golden_df


if __name__ == "__main__":
    # For local testing (replace paths as needed)
    import sys
    golden = run_in_colab(
        threads_csv=       "ms_threads.csv",
        first_inbound_csv= "microsofthelps_first_inbound_k9.csv",
        output_dir=        "./golden_set_output/",
    )
    print(golden.head())
