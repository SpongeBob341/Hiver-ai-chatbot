"""
Phase 6: Deterministic Safety Guardrails
-----------------------------------------
Post-generation safety checks applied to every reply before delivery.

Checks:
  1. PII solicitation block  — detects patterns that ask customers to share
                               sensitive info in a public tweet.
  2. Link domain whitelist   — ensures any URLs in the reply point only to
                               approved Microsoft domains.
  3. Reply length enforcement — hard 280-character cap (Twitter limit).
  4. Trace logger            — structured JSON log for every pipeline run;
                               feeds the LLM-as-judge evaluation harness.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 1. PII Solicitation Block
# ---------------------------------------------------------------------------

# Patterns that indicate the reply is asking for sensitive info publicly.
# A support agent should NEVER ask for these in a public tweet.
_PII_SOLICITATION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:send|share|provide|give)\s+(?:us|me)\s+(?:your\s+)?(?:email|e-mail)\b",
        r"\b(?:your|ur)\s+(?:email|e-mail)\s+(?:address|id)\b",
        r"\b(?:send|share|provide|give)\s+(?:us|me)\s+(?:your\s+)?(?:phone|mobile|cell)\b",
        r"\b(?:your|ur)\s+(?:phone|mobile|cell)\s*(?:number|#|no\.?)\b",
        r"\b(?:send|share|provide|give)\s+(?:us|me)\s+(?:your\s+)?(?:password|passcode|pin)\b",
        r"\b(?:your\s+)?(?:account\s+)?(?:number|id)\s*:\s*\d+\b",
        r"\b(?:send|share|provide|give)\s+(?:us|me)\s+(?:your\s+)?credit\s+card\b",
        r"\b(?:send|share|provide|give)\s+(?:us|me)\s+(?:your\s+)?social\s+security\b",
        r"\b(?:send|share|provide|give)\s+(?:us|me)\s+(?:your\s+)?ssn\b",
        r"\b(?:dm|direct\s*message)\s+(?:us|me)\s+(?:your\s+)?(?:email|password|phone|account)\b",
    ]
]


def check_pii_solicitation(reply: str) -> dict[str, Any]:
    """
    Detect if the reply solicits PII from the customer.

    Returns:
        {
            "is_safe":       bool,   # True = safe (no PII solicitation)
            "violations":    list[str],
            "blocked_reply": str | None,  # sanitized replacement if unsafe
        }
    """
    violations = []
    for pattern in _PII_SOLICITATION_PATTERNS:
        m = pattern.search(reply)
        if m:
            violations.append(m.group(0))

    if violations:
        blocked_reply = (
            "We'd love to help! Please visit https://support.microsoft.com "
            "or contact us through our official support channels for account-specific assistance."
        )
        return {
            "is_safe": False,
            "violations": violations,
            "blocked_reply": blocked_reply,
        }

    return {"is_safe": True, "violations": [], "blocked_reply": None}


# ---------------------------------------------------------------------------
# 2. Link Domain Whitelist
# ---------------------------------------------------------------------------

APPROVED_DOMAINS: set[str] = {
    "support.microsoft.com",
    "aka.ms",
    "microsoft.com",
    "docs.microsoft.com",
    "learn.microsoft.com",
    "answers.microsoft.com",
    "techcommunity.microsoft.com",
    "msft.social",
    "xbox.com",
    "office.com",
    "onedrive.live.com",
    "go.microsoft.com",
    "windows.microsoft.com",
    "t.co",              # Twitter's shortener — pass through (original domain unknown)
    "bit.ly",            # generic shortener — flag but allow
}

_URL_PATTERN = re.compile(r"https?://(?:www\.)?([^/\s?#]+)", re.IGNORECASE)


def check_link_whitelist(reply: str) -> dict[str, Any]:
    """
    Verify all URLs in the reply point to approved Microsoft domains.

    Returns:
        {
            "is_safe":             bool,
            "found_urls":          list[str],
            "unapproved_domains":  list[str],
        }
    """
    found_urls: list[str] = []
    unapproved: list[str] = []

    for m in _URL_PATTERN.finditer(reply):
        full_url = m.group(0)
        domain   = m.group(1).lower()
        found_urls.append(full_url)

        # Check domain and any parent domain
        if not any(
            domain == approved or domain.endswith("." + approved)
            for approved in APPROVED_DOMAINS
        ):
            unapproved.append(domain)

    return {
        "is_safe": len(unapproved) == 0,
        "found_urls": found_urls,
        "unapproved_domains": unapproved,
    }


# ---------------------------------------------------------------------------
# 3. Reply Length Enforcement
# ---------------------------------------------------------------------------

TWITTER_CHAR_LIMIT = 280
ESCALATION_REPLY_LIMIT = 240  # leave headroom for @handle prefix


def enforce_length(reply: str, limit: int = TWITTER_CHAR_LIMIT) -> dict[str, Any]:
    """
    Enforce Twitter's character limit.

    Returns:
        {
            "is_within_limit":  bool,
            "char_count":       int,
            "truncated_reply":  str,  # unchanged if within limit, truncated if not
        }
    """
    char_count = len(reply)
    if char_count <= limit:
        return {
            "is_within_limit": True,
            "char_count": char_count,
            "truncated_reply": reply,
        }

    # Truncate at the last word boundary before the limit
    truncated = reply[:limit - 1]
    last_space = truncated.rfind(" ")
    if last_space > limit // 2:
        truncated = truncated[:last_space]
    truncated = truncated.rstrip(".,;:!?") + "…"

    return {
        "is_within_limit": False,
        "char_count": char_count,
        "truncated_reply": truncated,
    }


# ---------------------------------------------------------------------------
# 4. Full Guardrail Runner
# ---------------------------------------------------------------------------

def apply_guardrails(
    reply: str,
    action: str = "AUTO_HANDLE",
) -> dict[str, Any]:
    """
    Run all safety checks on a generated reply.

    Returns a guardrail result dict; use `final_reply` as the safe output.

    Args:
        reply:  Generated reply text.
        action: "AUTO_HANDLE" or "ESCALATE" (affects length limit).

    Returns:
        {
            "final_reply":      str,
            "pii_check":        dict,
            "link_check":       dict,
            "length_check":     dict,
            "all_safe":         bool,
            "applied_fixes":    list[str],  # list of modifications made
        }
    """
    applied_fixes: list[str] = []

    # --- PII Check ---
    pii = check_pii_solicitation(reply)
    if not pii["is_safe"]:
        reply = pii["blocked_reply"]
        applied_fixes.append(f"PII solicitation blocked; replaced with safe fallback. Violations: {pii['violations']}")

    # --- Link Whitelist ---
    link = check_link_whitelist(reply)
    if not link["is_safe"]:
        applied_fixes.append(
            f"Unapproved domains found: {link['unapproved_domains']}. Reply flagged for review."
        )
        # Don't block — flag for review but allow through (human will check)

    # --- Length Enforcement ---
    limit = ESCALATION_REPLY_LIMIT if action == "ESCALATE" else TWITTER_CHAR_LIMIT
    length = enforce_length(reply, limit=limit)
    if not length["is_within_limit"]:
        reply = length["truncated_reply"]
        applied_fixes.append(
            f"Reply truncated from {length['char_count']} to ≤{limit} characters."
        )
        # Re-run length check on truncated version
        length = enforce_length(reply, limit=limit)

    all_safe = pii["is_safe"] and link["is_safe"] and length["is_within_limit"]

    return {
        "final_reply":   reply,
        "pii_check":     pii,
        "link_check":    link,
        "length_check":  length,
        "all_safe":      all_safe,
        "applied_fixes": applied_fixes,
    }


# ---------------------------------------------------------------------------
# 5. Trace Logger
# ---------------------------------------------------------------------------

class TraceLogger:
    """
    Structured JSON trace logger for every pipeline run.
    Feeds the LLM-as-judge evaluation harness (§9).

    Each trace record contains:
      - run_id:          Unique UUID for this inference run
      - timestamp:       ISO 8601 UTC
      - input_tweet:     Raw customer tweet
      - phase outputs:   intent, retrieval, escalation, reply, guardrails
      - metadata:        model versions, latencies (if provided)
    """

    def __init__(
        self,
        log_dir: Optional[str | Path] = None,
        echo: bool = False,
    ) -> None:
        """
        Args:
            log_dir: Directory to write JSONL trace log. If None, in-memory only.
            echo:    If True, print each trace to stdout.
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.echo = echo
        self._records: list[dict[str, Any]] = []

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = self.log_dir / "pipeline_traces.jsonl"
        else:
            self._log_path = None

    def log(
        self,
        input_tweet:        str,
        intent:             Optional[str]       = None,
        intent_confidence:  Optional[float]     = None,
        retrieved_threads:  Optional[list]      = None,
        retrieval_score:    Optional[float]     = None,
        escalation_action:  Optional[str]       = None,
        escalation_reason:  Optional[str]       = None,
        draft_reply:        Optional[str]       = None,
        final_reply:        Optional[str]       = None,
        guardrail_result:   Optional[dict]      = None,
        metadata:           Optional[dict]      = None,
    ) -> str:
        """
        Log a complete pipeline trace. Returns the run_id.
        """
        run_id = str(uuid.uuid4())
        record: dict[str, Any] = {
            "run_id":             run_id,
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "input_tweet":        input_tweet,
            "intent":             intent,
            "intent_confidence":  intent_confidence,
            "num_retrieved":      len(retrieved_threads) if retrieved_threads else 0,
            "retrieval_score":    retrieval_score,
            "escalation_action":  escalation_action,
            "escalation_reason":  escalation_reason,
            "draft_reply":        draft_reply,
            "final_reply":        final_reply,
            "guardrail_result":   guardrail_result,
            "metadata":           metadata or {},
        }
        self._records.append(record)

        if self._log_path:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.echo:
            print(json.dumps(record, indent=2, ensure_ascii=False))

        return run_id

    def get_records(self) -> list[dict[str, Any]]:
        """Return all logged records (in-memory)."""
        return list(self._records)

    def to_dataframe(self):
        """Convert trace log to a pandas DataFrame for analysis."""
        try:
            import pandas as pd
            return pd.DataFrame(self._records)
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")
