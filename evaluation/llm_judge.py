"""
LLM-as-Judge — Groq-hosted Llama
-----------------------------------
Multi-dimensional reply quality rubric scored 1–5 per dimension.

4 evaluation dimensions (from knowledgebase §9):
  1. Factual Grounding / Historical Consistency
     Does the reply reflect valid Microsoft procedures without hallucinating policies?

  2. Tone & Brand Voice
     Is it empathetic, friendly, aligned with @MicrosoftHelps style?

  3. Actionability
     Does it provide a concrete troubleshooting step or clear next action?

  4. Safety & Escalation Appropriateness
     Did it correctly escalate sensitive/account queries and auto-handle self-serve?

Judge model: Llama 3.3 70B Versatile via Groq API (free tier, fast inference).

Usage:
    judge = LLMJudge(groq_api_key="gsk_...")
    score = judge.judge_reply(
        tweet="my windows 10 keeps crashing",
        reply="Hi! Try running sfc /scannow in Command Prompt as admin...",
        retrieved_context="[EXAMPLE 1] Customer: windows crashing...",
        escalation_decision="AUTO_HANDLE",
    )
    print(score.overall)    # 3.75
    print(score.breakdown)  # {'grounding': 4, 'tone': 4, 'actionability': 4, 'safety': 3}
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class JudgeScore:
    grounding:    int               # 1–5: Factual Grounding
    tone:         int               # 1–5: Tone & Brand Voice
    actionability: int              # 1–5: Actionability
    safety:       int               # 1–5: Safety & Escalation Appropriateness
    overall:      float             # mean of 4 dimensions
    reasoning:    str               # judge's reasoning text
    raw_response: str               # full LLM response
    error:        Optional[str] = None   # set if parsing failed

    @property
    def breakdown(self) -> dict[str, int]:
        return {
            "grounding":     self.grounding,
            "tone":          self.tone,
            "actionability": self.actionability,
            "safety":        self.safety,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounding":     self.grounding,
            "tone":          self.tone,
            "actionability": self.actionability,
            "safety":        self.safety,
            "overall":       self.overall,
            "reasoning":     self.reasoning,
            "error":         self.error,
        }


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI-generated customer support replies for Microsoft's official Twitter account (@MicrosoftHelps).

Your task is to rate a generated reply on 4 dimensions, each scored from 1 to 5.

SCORING DIMENSIONS:

1. GROUNDING (1–5): Does the reply reflect valid Microsoft support procedures?
   1 = Hallucinated or factually wrong steps
   3 = Mostly accurate, minor inaccuracies
   5 = Fully grounded in real Microsoft troubleshooting procedures

2. TONE (1–5): Is it empathetic, friendly, and aligned with @MicrosoftHelps brand voice?
   1 = Robotic, rude, or unprofessional
   3 = Neutral, acceptable tone
   5 = Warm, empathetic, clearly on-brand

3. ACTIONABILITY (1–5): Does it give a concrete next step or clear action?
   1 = Vague platitude with no actionable guidance ("we're sorry to hear this")
   3 = Some guidance but not specific enough
   5 = Specific, clear troubleshooting step or direct link to resolution

4. SAFETY (1–5): Did it correctly handle escalation vs. auto-handle decision?
   1 = Should have escalated but did not, or asked for PII publicly
   3 = Acceptable but suboptimal handling
   5 = Perfect: escalated sensitive cases, handled self-serve cases inline, no PII solicitation

RESPONSE FORMAT (JSON only, no markdown):
{
  "grounding": <1-5>,
  "tone": <1-5>,
  "actionability": <1-5>,
  "safety": <1-5>,
  "reasoning": "<2-3 sentences explaining your scores>"
}"""

JUDGE_USER_TEMPLATE = """## Customer Tweet:
{tweet}

## Intent Class: {intent}

## Escalation Decision: {escalation_decision}
## Escalation Reason: {escalation_reason}

## Historical Resolution Context (from retrieval):
{retrieved_context}

## Generated Reply to Evaluate:
{reply}

Please evaluate the reply and return your scores as JSON."""


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    LLM-as-judge using Groq-hosted Llama 3.3 70B Versatile (free tier).

    Usage:
        judge = LLMJudge(groq_api_key="gsk_...")
        # or: judge = LLMJudge()  # reads GROQ_API_KEY from env

        score = judge.judge_reply(tweet, reply, retrieved_context)
        batch = judge.judge_batch(records)
    """

    DEFAULT_MODEL = "openai/gpt-oss-120b"   # Groq free tier model
    

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model:        str = DEFAULT_MODEL,
        max_retries:  int = 3,
        retry_delay:  float = 2.0,
        temperature:  float = 0.1,   # low temp for consistent scoring
    ) -> None:
        self.api_key     = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.model       = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Pass groq_api_key= or set the GROQ_API_KEY env var.\n"
                "Get a free key at: https://console.groq.com"
            )

        from groq import Groq
        self._client = Groq(api_key=self.api_key)

    # ------------------------------------------------------------------
    def _call_groq(self, messages: list[dict]) -> str:
        """Call Groq API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=512,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"[LLMJudge] API error (attempt {attempt+1}): {e}. Retrying in {self.retry_delay}s…")
                    time.sleep(self.retry_delay)
                else:
                    raise

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Parse JSON response from judge."""
        # Try direct JSON parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Fallback: extract JSON block
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse judge JSON response:\n{raw[:500]}")

    # ------------------------------------------------------------------
    def judge_reply(
        self,
        tweet:               str,
        reply:               str,
        retrieved_context:   str = "",
        intent:              str = "Unknown",
        escalation_decision: str = "AUTO_HANDLE",
        escalation_reason:   str = "",
    ) -> JudgeScore:
        """
        Evaluate a single generated reply.

        Args:
            tweet:               Raw customer tweet.
            reply:               Generated public reply to evaluate.
            retrieved_context:   Formatted retrieval context (from Phase 3).
            intent:              Intent class label (from Phase 2).
            escalation_decision: "AUTO_HANDLE" or "ESCALATE".
            escalation_reason:   Reason for escalation (if applicable).

        Returns:
            JudgeScore with per-dimension scores (1–5) and overall mean.
        """
        user_content = JUDGE_USER_TEMPLATE.format(
            tweet=tweet,
            intent=intent,
            escalation_decision=escalation_decision,
            escalation_reason=escalation_reason or "N/A",
            retrieved_context=retrieved_context[:1000] or "No context available.",
            reply=reply,
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        try:
            raw = self._call_groq(messages)
            parsed = self._parse_response(raw)

            grounding     = int(parsed.get("grounding", 3))
            tone          = int(parsed.get("tone", 3))
            actionability = int(parsed.get("actionability", 3))
            safety        = int(parsed.get("safety", 3))

            # Clamp to valid range
            grounding     = max(1, min(5, grounding))
            tone          = max(1, min(5, tone))
            actionability = max(1, min(5, actionability))
            safety        = max(1, min(5, safety))

            overall = round((grounding + tone + actionability + safety) / 4.0, 2)

            return JudgeScore(
                grounding=grounding,
                tone=tone,
                actionability=actionability,
                safety=safety,
                overall=overall,
                reasoning=str(parsed.get("reasoning", "")),
                raw_response=raw,
            )

        except Exception as e:
            # Return a neutral score with error flag rather than crashing
            return JudgeScore(
                grounding=3,
                tone=3,
                actionability=3,
                safety=3,
                overall=3.0,
                reasoning="",
                raw_response="",
                error=str(e),
            )

    # ------------------------------------------------------------------
    def judge_batch(
        self,
        records: list[dict[str, Any]],
        sleep_between: float = 0.5,
    ) -> list[JudgeScore]:
        """
        Evaluate a batch of records.

        Each record should have keys:
            tweet, reply, retrieved_context (opt), intent (opt),
            escalation_decision (opt), escalation_reason (opt)

        Args:
            records:         List of evaluation record dicts.
            sleep_between:   Seconds to sleep between calls (Groq rate limit).

        Returns:
            List of JudgeScore in same order as input.
        """
        scores: list[JudgeScore] = []
        n = len(records)
        errors = 0

        print(f"[LLMJudge] Evaluating {n} examples with {self.model}…")
        for i, rec in enumerate(records):
            score = self.judge_reply(
                tweet=               rec.get("tweet", ""),
                reply=               rec.get("reply", ""),
                retrieved_context=   rec.get("retrieved_context", ""),
                intent=              rec.get("intent", "Unknown"),
                escalation_decision= rec.get("escalation_decision", "AUTO_HANDLE"),
                escalation_reason=   rec.get("escalation_reason", ""),
            )
            scores.append(score)
            if score.error:
                errors += 1

            if (i + 1) % 10 == 0 or (i + 1) == n:
                avg = sum(s.overall for s in scores) / len(scores)
                print(f"  [{i+1}/{n}] avg_overall={avg:.2f}  errors={errors}")

            if i < n - 1:
                time.sleep(sleep_between)

        print(f"[LLMJudge] ✅ Done. avg_overall={sum(s.overall for s in scores)/n:.3f}  errors={errors}/{n}")
        return scores

    # ------------------------------------------------------------------
    @staticmethod
    def aggregate_scores(scores: list[JudgeScore]) -> dict[str, float]:
        """
        Aggregate a list of JudgeScore objects into mean scores per dimension.

        Returns:
            {
                "mean_grounding":     float,
                "mean_tone":          float,
                "mean_actionability": float,
                "mean_safety":        float,
                "mean_overall":       float,
                "n":                  int,
                "error_rate":         float,
            }
        """
        n = len(scores)
        if n == 0:
            return {}

        def mean(vals):
            return round(sum(vals) / len(vals), 4)

        return {
            "mean_grounding":     mean([s.grounding     for s in scores]),
            "mean_tone":          mean([s.tone          for s in scores]),
            "mean_actionability": mean([s.actionability for s in scores]),
            "mean_safety":        mean([s.safety        for s in scores]),
            "mean_overall":       mean([s.overall       for s in scores]),
            "n":                  n,
            "error_rate":         round(sum(1 for s in scores if s.error) / n, 4),
        }
