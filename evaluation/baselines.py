"""
Baseline Comparisons
---------------------
Three baselines required by the assignment (knowledgebase §8):

  Baseline 0 — Trivial Baseline:
    • Classification: Majority class → always "Windows General"
    • Reply:          Static canned template
    • Escalation:     Keyword-only trigger (no LLM, no context)

  Baseline 1 — Simple Baseline:
    • Classification: TF-IDF + Logistic Regression (7 classes)
    • Reply:          Zero-shot Llama 3.3 70B via Groq (no RAG grounding)
    • Escalation:     Naive LLM prompt "Should this be escalated? Yes or No."

  Comparison output feeds directly into the results table in the README.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Shared output dataclass (mirrors AgentOutput for easy comparison)
# ---------------------------------------------------------------------------

@dataclass
class BaselineOutput:
    intent:       str
    action:       str    # "AUTO_HANDLE" or "ESCALATE"
    public_reply: str
    baseline_id:  str    # "baseline_0" or "baseline_1"


# ---------------------------------------------------------------------------
# BASELINE 0 — Trivial Baseline
# ---------------------------------------------------------------------------

# Static canned reply (used by Baseline 0)
CANNED_REPLY = (
    "Hi! We're sorry to hear that. Please visit https://support.microsoft.com "
    "or DM us for further assistance. We're here to help! 🙏"
)

# Hard escalation keywords for Baseline 0
BASELINE0_ESCALATION_KEYWORDS = [
    "blue screen", "bsod", "black screen", "crash", "crashed",
    "billing", "charged", "payment", "refund", "fraud",
    "hacked", "locked out", "account locked", "compromised",
    "data loss", "deleted files", "drive failure",
]

MAJORITY_CLASS = "Windows General"


class MajorityClassBaseline:
    """
    Baseline 0: Trivial majority-class classifier.

    Classification: Always predicts "Windows General" (~37% accuracy ceiling).
    Reply:          Always returns the same static canned template.
    Escalation:     Keyword trigger only (BSOD/crash/billing/locked).
    """

    def predict_intent(self, text: str) -> str:
        return MAJORITY_CLASS

    def predict_escalation(self, text: str) -> str:
        lower = text.lower()
        for kw in BASELINE0_ESCALATION_KEYWORDS:
            if kw in lower:
                return "ESCALATE"
        return "AUTO_HANDLE"

    def run(self, tweet_text: str) -> BaselineOutput:
        intent = self.predict_intent(tweet_text)
        action = self.predict_escalation(tweet_text)
        return BaselineOutput(
            intent=intent,
            action=action,
            public_reply=CANNED_REPLY,
            baseline_id="baseline_0",
        )

    def run_batch(self, tweets: list[str]) -> list[BaselineOutput]:
        return [self.run(t) for t in tweets]


# ---------------------------------------------------------------------------
# BASELINE 1 — TF-IDF + Logistic Regression Classifier
# ---------------------------------------------------------------------------

class TFIDFLogisticBaseline:
    """
    Baseline 1 — Classification component:
    TF-IDF (8k features, 1–2 grams, min_df=3) + Logistic Regression.

    Matches the method used in the EDA/intent taxonomy phase
    (knowledgebase §7.1 methodology).
    """

    TFIDF_PARAMS = dict(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
    )
    LR_PARAMS = dict(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )

    def __init__(self) -> None:
        self._is_fitted = False
        self.pipeline = None
        self.label_encoder = None

    def fit(
        self,
        X_train: list[str],
        y_train: list[str | int],
    ) -> "TFIDFLogisticBaseline":
        """Fit TF-IDF + LR pipeline on training data."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_train)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**self.TFIDF_PARAMS)),
            ("lr",    LogisticRegression(**self.LR_PARAMS)),
        ])
        self.pipeline.fit(X_train, y_encoded)
        self._is_fitted = True
        print(f"[TFIDFLogisticBaseline] ✅ Fitted on {len(X_train)} examples.")
        return self

    def predict(self, texts: list[str]) -> list[str]:
        """Return predicted intent label strings."""
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .predict()")
        encoded = self.pipeline.predict(texts)
        return list(self.label_encoder.inverse_transform(encoded))

    def predict_proba(self, texts: list[str]) -> Any:
        """Return probability matrix over all classes."""
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .predict()")
        return self.pipeline.predict_proba(texts)


# ---------------------------------------------------------------------------
# BASELINE 1 — Zero-shot Groq reply drafter (no RAG)
# ---------------------------------------------------------------------------

ZERO_SHOT_SYSTEM = (
    "You are a Microsoft customer support agent on Twitter (@MicrosoftHelps). "
    "Reply to the customer's tweet in a helpful, empathetic, professional tone. "
    "Keep your reply under 280 characters. Be specific and actionable."
)

ZERO_SHOT_USER = "Customer tweet: {tweet}\n\nReply:"

NAIVE_ESCALATION_SYSTEM = (
    "You are a triage agent for Microsoft customer support on Twitter. "
    "Your only job is to decide whether the following customer tweet should be "
    "escalated to a human agent or can be auto-handled. Respond with exactly one word: "
    "ESCALATE or AUTO_HANDLE."
)

NAIVE_ESCALATION_USER = "Customer tweet: {tweet}\n\nDecision (ESCALATE or AUTO_HANDLE):"


class ZeroShotGroqBaseline:
    """
    Baseline 1 — Reply drafter and escalation components:
    Uses Groq-hosted Llama 3.3 70B without RAG grounding.

    This is the 'unconstrained zero-shot LLM' baseline.
    Paired with TFIDFLogisticBaseline for classification.
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        groq_api_key:      Optional[str] = None,
        model:             str = DEFAULT_MODEL,
        sleep_between:     float = 0.3,
    ) -> None:
        import os
        self.api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.model   = model
        self.sleep   = sleep_between

        if not self.api_key:
            raise ValueError("Set GROQ_API_KEY env var or pass groq_api_key=")

        from groq import Groq
        self._client = Groq(api_key=self.api_key)

    def _call(self, system: str, user: str, max_tokens: int = 100) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERROR: {e}]"

    def generate_reply(self, tweet: str) -> str:
        """Zero-shot reply without any RAG context."""
        return self._call(
            system=ZERO_SHOT_SYSTEM,
            user=ZERO_SHOT_USER.format(tweet=tweet),
            max_tokens=100,
        )

    def predict_escalation(self, tweet: str) -> str:
        """Naive LLM escalation prompt — 'ESCALATE or AUTO_HANDLE?'"""
        raw = self._call(
            system=NAIVE_ESCALATION_SYSTEM,
            user=NAIVE_ESCALATION_USER.format(tweet=tweet),
            max_tokens=10,
        )
        raw_upper = raw.strip().upper()
        if "ESCALATE" in raw_upper:
            return "ESCALATE"
        return "AUTO_HANDLE"

    def run(self, tweet_text: str, intent: str = "Unknown") -> BaselineOutput:
        action = self.predict_escalation(tweet_text)
        reply  = self.generate_reply(tweet_text)
        return BaselineOutput(
            intent=intent,
            action=action,
            public_reply=reply,
            baseline_id="baseline_1",
        )

    def run_batch(
        self,
        tweets: list[str],
        intents: Optional[list[str]] = None,
    ) -> list[BaselineOutput]:
        results: list[BaselineOutput] = []
        for i, tweet in enumerate(tweets):
            intent = (intents[i] if intents else "Unknown")
            results.append(self.run(tweet, intent))
            if i < len(tweets) - 1:
                time.sleep(self.sleep)
        return results


# ---------------------------------------------------------------------------
# Results comparison table printer
# ---------------------------------------------------------------------------

def print_baseline_comparison(
    system_metrics:     dict[str, Any],
    baseline0_metrics:  dict[str, Any],
    baseline1_metrics:  dict[str, Any],
    judge_system:       Optional[dict] = None,
    judge_baseline0:    Optional[dict] = None,
    judge_baseline1:    Optional[dict] = None,
) -> None:
    """
    Print a formatted comparison table for all three systems.
    Matches the report table format required by the assignment (knowledgebase §4).
    """
    header = f"{'Metric':<30} {'Baseline 0':>12} {'Baseline 1':>12} {'Our System':>12}"
    sep    = "─" * len(header)

    print("\n" + "=" * 70)
    print("📊 RESULTS vs. BASELINES")
    print("=" * 70)
    print(header)
    print(sep)

    def row(label, b0_key, b1_key, sys_key, src0=baseline0_metrics, src1=baseline1_metrics, src2=system_metrics):
        b0  = src0.get(b0_key, "—") if src0 else "—"
        b1  = src1.get(b1_key, "—") if src1 else "—"
        sys = src2.get(sys_key, "—") if src2 else "—"
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {label:<28} {fmt(b0):>12} {fmt(b1):>12} {fmt(sys):>12}")

    # Classification
    print(f"\n  {'── Intent Classification ──'}")
    row("Macro F1",        "macro_f1",     "macro_f1",     "macro_f1")
    row("Micro F1",        "micro_f1",     "micro_f1",     "micro_f1")
    row("Weighted F1",     "weighted_f1",  "weighted_f1",  "weighted_f1")

    # Escalation
    print(f"\n  {'── Escalation Decision ──'}")
    row("Precision",       "precision",    "precision",    "precision",
        src0=baseline0_metrics.get("escalation", {}),
        src1=baseline1_metrics.get("escalation", {}),
        src2=system_metrics.get("escalation", {}))
    row("Recall",          "recall",       "recall",       "recall",
        src0=baseline0_metrics.get("escalation", {}),
        src1=baseline1_metrics.get("escalation", {}),
        src2=system_metrics.get("escalation", {}))
    row("FNR (⚠️ lower=better)", "fnr",   "fnr",          "fnr",
        src0=baseline0_metrics.get("escalation", {}),
        src1=baseline1_metrics.get("escalation", {}),
        src2=system_metrics.get("escalation", {}))

    # LLM Judge (if provided)
    if judge_system:
        print(f"\n  {'── LLM-as-Judge (mean scores 1–5) ──'}")
        row("Grounding",       "mean_grounding",     "mean_grounding",     "mean_grounding",
            src0=judge_baseline0, src1=judge_baseline1, src2=judge_system)
        row("Actionability",   "mean_actionability", "mean_actionability", "mean_actionability",
            src0=judge_baseline0, src1=judge_baseline1, src2=judge_system)
        row("Tone",            "mean_tone",          "mean_tone",          "mean_tone",
            src0=judge_baseline0, src1=judge_baseline1, src2=judge_system)
        row("Overall",         "mean_overall",       "mean_overall",       "mean_overall",
            src0=judge_baseline0, src1=judge_baseline1, src2=judge_system)

    print(sep)
    print()
