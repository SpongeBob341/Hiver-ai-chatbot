"""
Phase 4: Escalation Gate
------------------------
Deterministic hard-rule escalation engine — no LLM judgment needed.

Decision table (from knowledgebase §13.1 Phase 4):
  BSOD / System Crash         → ESCALATE
  Payment / Billing Dispute   → ESCALATE
  Account Lockout / Hack      → ESCALATE
  Data Loss / Corruption      → ESCALATE
  Distress / Urgency (C0)     → ESCALATE
  Low retrieval confidence    → ESCALATE
  Everything else             → AUTO_HANDLE
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------

class Action(str, Enum):
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE    = "ESCALATE"


# ---------------------------------------------------------------------------
# Escalation trigger patterns
# ---------------------------------------------------------------------------

# 1. BSOD / System Crash
BSOD_PATTERNS = [
    r"\bblue\s*screen\b",
    r"\bbsod\b",
    r"\bblack\s*screen\b",
    r"\bsystem\s*crash\b",
    r"\bkernel\s*panic\b",
    r"\bcomputer\s*(?:won'?t|wont|does\s*not|doesn'?t)\s*(?:start|boot|turn\s*on)\b",
    r"\b(?:won'?t|wont|doesn'?t|does\s*not)\s*(?:start|boot)\b",
    r"\bstop\s*(?:code|error)\b",
    r"\b0x[0-9a-fA-F]{6,}\b",   # BSOD stop codes like 0x0000007E
    r"\bcrash(?:ed|ing)?\b",
    r"\bfroze?\b",
    r"\bfreezing\b",
    r"\bforce\s*restart\b",
    r"\bloop(?:ing)?\s*restart\b",
    r"\brestart\s*loop\b",
]

# 2. Payment / Billing Dispute
BILLING_PATTERNS = [
    r"\bbill(?:ing|ed)?\b",
    r"\bcharged?\b",
    r"\bpayment\b",
    r"\brefund\b",
    r"\bunauthori[sz]ed\s*(?:charge|transaction|payment)\b",
    r"\bfraud\b",
    r"\bsubscription\s*(?:cancel|charge|issue)\b",
    r"\bovercharged?\b",
    r"\bdouble\s*charged?\b",
    r"\bcredit\s*card\b",
    r"\bdebit\s*card\b",
    r"\binvoice\b",
    r"\bprice\s*(?:wrong|error|mistake)\b",
]

# 3. Account Lockout / Hack
LOCKOUT_PATTERNS = [
    r"\baccount\s*(?:hacked?|compromised|stolen|breached?|locked?)\b",
    r"\bhacked?\b",
    r"\bcompromised\b",
    r"\bunauthori[sz]ed\s*(?:access|login|sign.in)\b",
    r"\blocke?d?\s*out\b",
    r"\bcan'?t\s*(?:login|log\s*in|sign\s*in|access)\b",
    r"\bpassword\s*(?:reset|stolen|changed\s*without)\b",
    r"\bsomeone\s*else\s*(?:accessed?|using|in)\b",
    r"\bidentity\s*(?:theft|stolen)\b",
    r"\bphish(?:ing)?\b",
    r"\b2fa\s*(?:issue|problem|not\s*working)\b",
    r"\bauthenticator\s*(?:lost|not\s*working)\b",
]

# 4. Data Loss / Corruption
DATA_LOSS_PATTERNS = [
    r"\bdata\s*loss\b",
    r"\bdata\s*(?:lost|deleted|corrupted?|gone|disappeared?)\b",
    r"\bfiles?\s*(?:deleted|lost|gone|missing|corrupted?)\b",
    r"\bdeleted\s*(?:all|my)\s*files?\b",
    r"\bwiped?\b",
    r"\bformat(?:ted)?\b",
    r"\bdisk\s*(?:error|fail|failure|corrupted?)\b",
    r"\bcorrupted?\s*(?:drive|disk|file|data)\b",
    r"\bhard\s*drive\s*(?:fail|failed|crash|error)\b",
    r"\bssd\s*(?:fail|failed|crash|error)\b",
    r"\bbackup\s*(?:fail|lost|missing|corrupted?)\b",
    r"\bonedrive\s*(?:deleted|lost|missing|corrupted?|gone)\b",
]

# 5. Distress / Urgency (maps to C0: Urgent / Distress Escalation)
DISTRESS_PATTERNS = [
    r"\burgent(?:ly)?\b",
    r"\basap\b",
    r"\bhelp\s*asap\b",
    r"\bplease\s*help\b",
    r"\bdesperate(?:ly)?\b",
    r"\bemergency\b",
    r"\bcritical\b",
    r"\bimmediate(?:ly)?\b",
    r"\bso\s*frustrated\b",
    r"\bso\s*angry\b",
    r"\bthis\s*is\s*ridiculous\b",
    r"\bunacceptable\b",
    r"\bsick\s*and\s*tired\b",
    r"\bterrible\s*service\b",
    r"\bworst\s*(?:service|support|experience)\b",
    r"\bsuing\b",
    r"\blegal\s*action\b",
    r"\bescalate\b",
]

# Compile all patterns
_COMPILED_RULES: list[tuple[str, list[re.Pattern]]] = [
    ("BSOD / System Crash",        [re.compile(p, re.IGNORECASE) for p in BSOD_PATTERNS]),
    ("Payment / Billing Dispute",  [re.compile(p, re.IGNORECASE) for p in BILLING_PATTERNS]),
    ("Account Lockout / Hack",     [re.compile(p, re.IGNORECASE) for p in LOCKOUT_PATTERNS]),
    ("Data Loss / Corruption",     [re.compile(p, re.IGNORECASE) for p in DATA_LOSS_PATTERNS]),
    ("Distress / Urgency",         [re.compile(p, re.IGNORECASE) for p in DISTRESS_PATTERNS]),
]

# Retrieval confidence threshold below which we escalate
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.35


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class EscalationDecision:
    action:             Action
    reason:             str
    triggered_by:       Optional[str]     # category name that triggered escalation
    matched_pattern:    Optional[str]     # the specific pattern that matched
    confidence_score:   Optional[float]   # retrieval confidence if relevant
    internal_note:      str               # note for human agent handoff packet


# ---------------------------------------------------------------------------
# EscalationGate
# ---------------------------------------------------------------------------

class EscalationGate:
    """
    Deterministic hard-rule escalation engine.

    Usage:
        gate = EscalationGate()
        decision = gate.decide(tweet_text, intent_label, retrieval_confidence)
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        verbose: bool = False,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose

    def _scan_patterns(self, text: str) -> Optional[tuple[str, str]]:
        """
        Scan text against all escalation rule categories.

        Returns:
            (category_name, matched_pattern_string) if a match is found, else None.
        """
        for category, patterns in _COMPILED_RULES:
            for pattern in patterns:
                m = pattern.search(text)
                if m:
                    return category, m.group(0)
        return None

    def decide(
        self,
        tweet_text: str,
        intent_label: str,
        retrieval_confidence: Optional[float] = None,
    ) -> EscalationDecision:
        """
        Apply the escalation decision tree.

        Priority order:
        1. Hard keyword/regex trigger on tweet text → ESCALATE immediately
        2. Intent label is distress class (C0 / BSOD+Distress) → ESCALATE
        3. Low retrieval confidence → ESCALATE with static fallback
        4. Default → AUTO_HANDLE

        Args:
            tweet_text:            Raw or normalized incoming customer tweet.
            intent_label:          Predicted intent class (from Phase 2).
            retrieval_confidence:  Float in [0, 1] from Phase 3 retrieval score.

        Returns:
            EscalationDecision dataclass.
        """
        # --- Rule 1: Hard pattern scan ---
        match = self._scan_patterns(tweet_text)
        if match:
            category, matched_text = match
            return EscalationDecision(
                action=Action.ESCALATE,
                reason=f"Escalation trigger detected: {category}",
                triggered_by=category,
                matched_pattern=matched_text,
                confidence_score=retrieval_confidence,
                internal_note=(
                    f"Auto-escalated by pattern rule [{category}]. "
                    f"Matched: '{matched_text}'. "
                    f"Intent: {intent_label}. "
                    "Please review for sensitivity before responding."
                ),
            )

        # --- Rule 2: Intent-based escalation ---
        ESCALATION_INTENTS = {
            "BSOD & Crash / Distress",
            "bsod_distress",
            "C0",
            "C2",
        }
        if intent_label in ESCALATION_INTENTS:
            return EscalationDecision(
                action=Action.ESCALATE,
                reason=f"Intent class '{intent_label}' is a high-risk escalation category",
                triggered_by="Intent-based rule",
                matched_pattern=None,
                confidence_score=retrieval_confidence,
                internal_note=(
                    f"Escalated by intent class [{intent_label}]. "
                    "Customer tweet indicates a crash, BSOD, or distress scenario."
                ),
            )

        # --- Rule 3: Low retrieval confidence ---
        if (
            retrieval_confidence is not None
            and retrieval_confidence < self.confidence_threshold
        ):
            return EscalationDecision(
                action=Action.ESCALATE,
                reason=(
                    f"Retrieval confidence ({retrieval_confidence:.2f}) is below "
                    f"threshold ({self.confidence_threshold:.2f}) — "
                    "no grounded historical resolution found"
                ),
                triggered_by="Low retrieval confidence",
                matched_pattern=None,
                confidence_score=retrieval_confidence,
                internal_note=(
                    f"No sufficiently similar historical resolution found "
                    f"(confidence={retrieval_confidence:.2f}, threshold={self.confidence_threshold:.2f}). "
                    "Escalating to avoid hallucinated response. "
                    "Static reply: 'We're looking into this — a specialist will follow up shortly.'"
                ),
            )

        # --- Default: AUTO_HANDLE ---
        return EscalationDecision(
            action=Action.AUTO_HANDLE,
            reason="No escalation triggers detected; retrieval confidence sufficient",
            triggered_by=None,
            matched_pattern=None,
            confidence_score=retrieval_confidence,
            internal_note="Auto-handle. Proceed to grounded reply drafter.",
        )


# ---------------------------------------------------------------------------
# Convenience function for quick label checks (used in golden set builder)
# ---------------------------------------------------------------------------

def get_escalation_label(tweet_text: str) -> dict[str, Any]:
    """
    Returns a compact escalation label dict for annotation purposes.
    Used in golden set construction and evaluation harness.

    Returns:
        {
            "is_escalation": bool,
            "category": str | None,
            "matched_text": str | None,
        }
    """
    gate = EscalationGate()
    match = gate._scan_patterns(tweet_text)
    if match:
        category, matched_text = match
        return {"is_escalation": True, "category": category, "matched_text": matched_text}
    return {"is_escalation": False, "category": None, "matched_text": None}


# make Any available (referenced above without import guard)
from typing import Any  # noqa: E402 (needed at module bottom for get_escalation_label)
