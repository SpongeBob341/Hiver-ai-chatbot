"""
Phase 1: Intake, Normalization & Thread Stitching
--------------------------------------------------
• normalize_tweet(text)                    — lowercase, de-slang, mention/URL cleanup
• extract_entities(text)                   — device, OS, component extraction
• stitch_multipart(turns)                  — merge 1/2 … n/n same-author consecutive turns
• prepare_thread_for_embedding(turns, k=6) — returns a single string ready for embedding
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Regex patterns — multi-part tweet detection
# ---------------------------------------------------------------------------
RE_PART_START = re.compile(r"\b1\s*/\s*\d+\b|\b1\s+of\s+\d+\b", re.IGNORECASE)
RE_PART_CONT  = re.compile(r"\b[2-9]\s*/\s*\d+\b|\b[2-9]\s+of\s+\d+\b", re.IGNORECASE)
RE_CONT_LABEL = re.compile(r"\(cont\.?\)", re.IGNORECASE)

# Twitter-style URL pattern
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENTION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#(\w+)")

# Common Twitter slang → normalised form
SLANG_MAP: dict[str, str] = {
    r"\bplz\b": "please",
    r"\bpls\b": "please",
    r"\bu\b": "you",
    r"\bur\b": "your",
    r"\br\b": "are",
    r"\bthanku\b": "thank you",
    r"\bthx\b": "thanks",
    r"\bty\b": "thanks",
    r"\bbtw\b": "by the way",
    r"\bimo\b": "in my opinion",
    r"\bafaik\b": "as far as I know",
    r"\bsmh\b": "shaking my head",
    r"\bwtf\b": "what the",
    r"\bfyi\b": "for your information",
    r"\basap\b": "as soon as possible",
    r"\bbc\b": "because",
    r"\bcant\b": "cannot",
    r"\bdont\b": "do not",
    r"\bwont\b": "will not",
    r"\bim\b": "i am",
    r"\bive\b": "i have",
}

# Entity extraction patterns
DEVICE_PATTERNS: dict[str, str] = {
    r"\bsurface pro\b": "Surface Pro",
    r"\bsurface book\b": "Surface Book",
    r"\bsurface laptop\b": "Surface Laptop",
    r"\bsurface go\b": "Surface Go",
    r"\bsurface studio\b": "Surface Studio",
    r"\bsurface\b": "Surface",
    r"\bxbox one\b": "Xbox One",
    r"\bxbox series x\b": "Xbox Series X",
    r"\bxbox series s\b": "Xbox Series S",
    r"\bxbox\b": "Xbox",
    r"\blaptop\b": "laptop",
    r"\bdesktop\b": "desktop",
    r"\bpc\b": "PC",
    r"\btablet\b": "tablet",
}

OS_PATTERNS: dict[str, str] = {
    r"\bwindows 11\b": "Windows 11",
    r"\bwindows 10\b": "Windows 10",
    r"\bwindows 8\.?1?\b": "Windows 8",
    r"\bwindows 7\b": "Windows 7",
    r"\bwin\s?11\b": "Windows 11",
    r"\bwin\s?10\b": "Windows 10",
    r"\bwindows\b": "Windows",
}

COMPONENT_PATTERNS: dict[str, str] = {
    r"\boutlook\b": "Outlook",
    r"\bonedrive\b": "OneDrive",
    r"\bexcel\b": "Excel",
    r"\bword\b": "Word",
    r"\bpowerpoint\b": "PowerPoint",
    r"\bteams\b": "Teams",
    r"\boffice\s?365\b": "Office 365",
    r"\bmicrosoft\s?365\b": "Microsoft 365",
    r"\bhotmail\b": "Hotmail",
    r"\bskype\b": "Skype",
    r"\bcourtana\b": "Cortana",
    r"\bdefender\b": "Windows Defender",
    r"\bbitlocker\b": "BitLocker",
    r"\bedge\b": "Edge",
    r"\bstore\b": "Microsoft Store",
    r"\bminecraft\b": "Minecraft",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_tweet(text: str, remove_urls: bool = True) -> str:
    """
    Normalize a raw tweet for downstream NLP tasks.

    Steps:
    1. Remove @mentions (preserve context without noise)
    2. Optionally remove URLs (bare URL tweets stay as-is for flagging)
    3. Expand hashtags (remove # symbol)
    4. Apply slang map
    5. Lowercase & strip extra whitespace
    """
    if not isinstance(text, str):
        return ""

    # Remove @mentions
    text = RE_MENTION.sub(" ", text)

    if remove_urls:
        text = RE_URL.sub(" <URL> ", text)

    # Expand hashtags: #Windows → Windows
    text = RE_HASHTAG.sub(r"\1", text)

    # Apply slang map
    for pattern, replacement in SLANG_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Lowercase and clean whitespace
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Extract Microsoft-domain entities from tweet text.

    Returns:
        {
            "devices":     ["Surface Pro", "Xbox"],
            "os":          ["Windows 10"],
            "components":  ["Outlook", "OneDrive"],
        }
    """
    lower = text.lower()
    entities: dict[str, list[str]] = {"devices": [], "os": [], "components": []}

    for pattern, label in DEVICE_PATTERNS.items():
        if re.search(pattern, lower):
            if label not in entities["devices"]:
                entities["devices"].append(label)

    for pattern, label in OS_PATTERNS.items():
        if re.search(pattern, lower):
            if label not in entities["os"]:
                entities["os"].append(label)

    for pattern, label in COMPONENT_PATTERNS.items():
        if re.search(pattern, lower):
            if label not in entities["components"]:
                entities["components"].append(label)

    return entities


def _is_multipart_start(text: str) -> bool:
    """Return True if tweet text starts or contains a '1/N' or '1 of N' marker."""
    return bool(RE_PART_START.search(text) or RE_CONT_LABEL.search(text))


def _is_multipart_continuation(text: str) -> bool:
    """Return True if tweet text contains a '2/N' … '9/N' or '(cont)' marker."""
    return bool(RE_PART_CONT.search(text) or RE_CONT_LABEL.search(text))


def stitch_multipart(
    turns: list[dict[str, Any]],
    author_key: str = "author",
    text_key: str = "text",
) -> list[dict[str, Any]]:
    """
    Merge consecutive same-author multi-part tweets (1/2, 2/2, (cont)) into a
    single logical turn before chunking/embedding.

    Args:
        turns: Ordered list of turn dicts, each with at least `author_key` and
               `text_key`. May also have `tweet_id`, `created_at`, etc.
        author_key: Key for the author field.
        text_key:   Key for the tweet text field.

    Returns:
        New list of turns with multipart tweets concatenated. The stitched turn
        inherits all metadata from the *first* part; an `is_stitched` flag is
        added when stitching occurs.
    """
    if not turns:
        return []

    stitched: list[dict[str, Any]] = []
    i = 0

    while i < len(turns):
        current = dict(turns[i])  # shallow copy to avoid mutating input
        text = current.get(text_key, "")

        if _is_multipart_start(text):
            # Accumulate all consecutive continuation turns from the same author
            parts = [text]
            j = i + 1
            while j < len(turns):
                nxt = turns[j]
                if nxt.get(author_key) == current.get(author_key) and (
                    _is_multipart_continuation(nxt.get(text_key, ""))
                    or _is_multipart_start(nxt.get(text_key, ""))
                ):
                    parts.append(nxt.get(text_key, ""))
                    j += 1
                else:
                    break

            if len(parts) > 1:
                # Strip part markers and concatenate
                cleaned_parts = []
                for p in parts:
                    p = RE_PART_START.sub("", p)
                    p = RE_PART_CONT.sub("", p)
                    p = RE_CONT_LABEL.sub("", p)
                    p = p.strip()
                    cleaned_parts.append(p)
                current[text_key] = " ".join(cleaned_parts)
                current["is_stitched"] = True
                current["num_parts_stitched"] = len(parts)
                i = j  # skip consumed continuation turns
            else:
                i += 1
        else:
            i += 1

        stitched.append(current)

    return stitched


def prepare_thread_for_embedding(
    thread_turns: list[dict[str, Any]],
    max_turns: int = 6,
    author_key: str = "author",
    text_key: str = "text",
) -> str:
    """
    Convert a list of thread turns into a single embedding-ready string.

    Pipeline:
        1. Stitch multipart tweets (Phase 1 stitcher)
        2. Cap at max_turns to stay within 512-token embedding window
        3. Format as "Author: Text" newline-joined string

    Args:
        thread_turns: Ordered list of turn dicts.
        max_turns:    Maximum number of turns to include (default 6, covers 81%+ corpus).
        author_key:   Key for author field.
        text_key:     Key for text field.

    Returns:
        Single string representation of the thread.
    """
    turns = stitch_multipart(thread_turns, author_key=author_key, text_key=text_key)
    turns = turns[:max_turns]
    return "\n".join(
        f"{t.get(author_key, 'Unknown')}: {t.get(text_key, '')}" for t in turns
    )


def is_bare_url_tweet(text: str) -> bool:
    """
    Return True if the tweet text consists *only* of URLs (a RAG dead-zone).
    These are flagged as type='reference_link' and excluded from generative retrieval.
    """
    if not isinstance(text, str):
        return False
    stripped = RE_MENTION.sub("", text).strip()
    stripped = RE_URL.sub("", stripped).strip()
    # After removing mentions and URLs, bare-URL tweets have less than 5 characters or no alphanumeric chars left
    alphanumeric = re.sub(r"\W+", "", stripped)
    return len(alphanumeric) == 0 or len(stripped) < 5


def safety_precheck(text: str) -> dict[str, bool]:
    """
    Rule-based pre-check before entering the main pipeline.

    Returns a dict of flags:
        {
            "is_spam":         bool,
            "is_promotional":  bool,
            "has_pii_risk":    bool,
        }
    """
    lower = text.lower()

    spam_signals = ["follow us", "click here", "win a prize", "limited time offer"]
    promo_signals = ["buy now", "discount", "sale", "coupon", "promo code"]
    pii_signals = [
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",   # phone numbers
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # credit card
    ]

    is_spam        = any(s in lower for s in spam_signals)
    is_promotional = any(s in lower for s in promo_signals)
    has_pii_risk   = any(re.search(p, text) for p in pii_signals)

    return {
        "is_spam": is_spam,
        "is_promotional": is_promotional,
        "has_pii_risk": has_pii_risk,
    }
