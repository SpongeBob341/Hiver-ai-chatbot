"""
Pipeline Orchestrator — SupportAgent
--------------------------------------
Wires all 6 phases into a single end-to-end agent.

  Phase 1: Intake, Normalization & Thread Stitching
  Phase 2: Intent Classification (BERTweet-base)
  Phase 3: Hybrid Retrieval (BM25 + FAISS)
  Phase 4: Escalation Gate (hard rules)
  Phase 5: Reply Drafter (Llama 3.2 3B LoRA)
  Phase 6: Safety Guardrails + Trace Logger

Output schema:
    AgentOutput {
        intent, intent_confidence, action, reason,
        public_reply, internal_handoff, trace_id
    }
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from .phase1_intake import normalize_tweet, extract_entities, is_bare_url_tweet, safety_precheck, RE_MENTION
    from .phase4_escalation import EscalationGate, Action
    from .phase6_guardrails import apply_guardrails, TraceLogger
except ImportError:
    from phase1_intake import normalize_tweet, extract_entities, is_bare_url_tweet, safety_precheck, RE_MENTION
    from phase4_escalation import EscalationGate, Action
    from phase6_guardrails import apply_guardrails, TraceLogger

# Lazy imports for heavy models (only loaded when SupportAgent is initialized)
# from .phase2_intent    import IntentClassifier
# from .phase3_retrieval import HybridRetriever
# from .phase5_drafter   import ReplyDrafter


# ---------------------------------------------------------------------------
# Static responses
# ---------------------------------------------------------------------------

CHANNEL_GUIDE_REPLY = (
    "Hi! 👋 You can reach us via:\n"
    "• 🌐 https://support.microsoft.com\n"
    "• 💬 Virtual Agent: https://aka.ms/MicrosoftVirtualAgent\n"
    "• 📞 Contact: https://support.microsoft.com/contactus\n"
    "We'll get you to the right channel!"
)

LOW_CONFIDENCE_FALLBACK = (
    "Hi! We're looking into this for you — a specialist will follow up shortly. "
    "For faster help, visit https://support.microsoft.com 🙏"
)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    # Core pipeline outputs
    intent:             str
    intent_confidence:  float
    action:             str           # "AUTO_HANDLE" or "ESCALATE"
    reason:             str           # escalation reason or "No escalation"
    public_reply:       str           # final tweet-ready reply
    internal_handoff:   Optional[str] # populated on ESCALATE

    # Tracing & debugging
    trace_id:           str
    latency_ms:         float
    entities:           dict          # extracted entities
    safety_flags:       dict          # pre-check flags
    guardrail_result:   dict          # post-generation safety results
    retrieved_count:    int
    retrieval_score:    float

    # Raw intermediate outputs (for evaluation harness)
    raw_reply:          str           # before guardrails
    metadata:           dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent":             self.intent,
            "intent_confidence":  self.intent_confidence,
            "action":             self.action,
            "reason":             self.reason,
            "public_reply":       self.public_reply,
            "internal_handoff":   self.internal_handoff,
            "trace_id":           self.trace_id,
            "latency_ms":         self.latency_ms,
            "entities":           self.entities,
            "safety_flags":       self.safety_flags,
            "retrieved_count":    self.retrieved_count,
            "retrieval_score":    self.retrieval_score,
        }


# ---------------------------------------------------------------------------
# SupportAgent
# ---------------------------------------------------------------------------

class SupportAgent:
    """
    End-to-end @MicrosoftHelps AI support agent.

    Usage (Colab):
        agent = SupportAgent.load(
            intent_model="your-hf-user/bertweet-mshelps-intent",
            index_dir="/content/drive/MyDrive/hiver_index/",
            drafter_base="meta-llama/Llama-3.2-3B-Instruct",
            drafter_lora="your-hf-user/llama32-mshelps-drafter",
        )

        output = agent.run("my windows 10 keeps crashing with blue screen every hour")
        print(output.public_reply)

    For baseline comparison (no LoRA adapter):
        agent = SupportAgent.load(..., drafter_lora=None)
    """

    def __init__(
        self,
        intent_classifier,
        retriever,
        drafter,
        escalation_gate: Optional[EscalationGate] = None,
        logger:          Optional[TraceLogger]     = None,
        verbose:         bool = False,
    ) -> None:
        self.classifier = intent_classifier
        self.retriever  = retriever
        self.drafter    = drafter
        self.gate       = escalation_gate or EscalationGate()
        self.logger     = logger or TraceLogger(echo=verbose)
        self.verbose    = verbose

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        intent_model:  str,
        index_dir:     str,
        drafter_base:  str  = "meta-llama/Llama-3.2-3B-Instruct",
        drafter_lora:  Optional[str] = None,
        log_dir:       Optional[str] = None,
        verbose:       bool = False,
    ) -> "SupportAgent":
        """
        Load all components from HuggingFace Hub / Drive index directory.

        Args:
            intent_model:  HF repo id for fine-tuned BERTweet intent classifier.
            index_dir:     Path to directory with bm25_index.pkl, faiss_index.bin,
                           thread_metadata.json.
            drafter_base:  HF repo id for Llama 3.2 3B Instruct base model.
            drafter_lora:  HF repo id for LoRA adapter. None = zero-shot Baseline 1.
            log_dir:       Directory to write pipeline_traces.jsonl.
            verbose:       Print step-by-step progress.
        """
        try:
            from .phase2_intent import IntentClassifier
            from .phase3_retrieval import HybridRetriever
            from .phase5_drafter import ReplyDrafter
        except ImportError:
            from phase2_intent import IntentClassifier
            from phase3_retrieval import HybridRetriever
            from phase5_drafter import ReplyDrafter

        print("=" * 60)
        print("🚀 Loading Hiver AI Support Agent")
        print("=" * 60)

        print("\n[1/3] Loading Intent Classifier...")
        if isinstance(intent_model, IntentClassifier):
            classifier = intent_model
            print("  Using provided IntentClassifier instance.")
        else:
            classifier = IntentClassifier.from_pretrained(intent_model)

        print("\n[2/3] Loading Hybrid Retrieval Index...")
        retriever = HybridRetriever.load(index_dir)

        print("\n[3/3] Loading Reply Drafter...")
        drafter = ReplyDrafter.from_pretrained(
            base_model=drafter_base,
            lora_adapter=drafter_lora,
        )

        logger = TraceLogger(log_dir=log_dir, echo=verbose)
        gate   = EscalationGate(verbose=verbose)

        print("\n✅ SupportAgent ready.\n")
        return cls(classifier, retriever, drafter, gate, logger, verbose)

    # ------------------------------------------------------------------
    def run(
        self,
        tweet_text: str,
        top_k:      int = 5,
    ) -> AgentOutput:
        """
        Run the full 6-phase pipeline on a single tweet.

        Args:
            tweet_text: Raw incoming customer tweet.
            top_k:      Number of threads to retrieve from index.

        Returns:
            AgentOutput with all pipeline results.
        """
        t_start = time.perf_counter()
        _log = self._log if self.verbose else lambda *a, **kw: None
        tweet_text = str(tweet_text or "")

        # ── Phase 1: Intake & Normalization ──────────────────────────────
        _log("Phase 1: Intake & Normalization")
        normalized = normalize_tweet(tweet_text)
        entities   = extract_entities(tweet_text)
        safety_flags = safety_precheck(tweet_text)
        is_bare_url  = is_bare_url_tweet(tweet_text)

        _log(f"  Normalized: {normalized[:80]}…")
        _log(f"  Entities:   {entities}")
        _log(f"  Safety:     {safety_flags}")

        # ── Phase 2: Intent Classification ───────────────────────────────
        # NOTE: classifier was trained on 'clean' column = lowercase + @mention removal.
        # Do NOT pass the slang-expanded `normalized` text — it shifts token distributions.
        # Apply only the same minimal cleaning used during training.
        _log("Phase 2: Intent Classification")
        classifier_input = RE_MENTION.sub("", tweet_text).lower().strip()
        intent_pred = self.classifier.predict(classifier_input)
        _log(f"  Intent: {intent_pred.label} (conf={intent_pred.confidence:.3f})")

        # Hard short-circuit: C3 Support Channel Navigation
        if intent_pred.is_routing:
            _log("  → C3 routing class detected. Short-circuiting to channel guide.")
            t_ms = (time.perf_counter() - t_start) * 1000
            trace_id = self.logger.log(
                input_tweet=tweet_text,
                intent=intent_pred.label,
                intent_confidence=intent_pred.confidence,
                escalation_action="AUTO_HANDLE",
                escalation_reason="Support Channel Navigation — static channel guide returned",
                final_reply=CHANNEL_GUIDE_REPLY,
            )
            return AgentOutput(
                intent=intent_pred.label,
                intent_confidence=intent_pred.confidence,
                action="AUTO_HANDLE",
                reason="Support Channel Navigation — channel guide returned",
                public_reply=CHANNEL_GUIDE_REPLY,
                internal_handoff=None,
                trace_id=trace_id,
                latency_ms=t_ms,
                entities=entities,
                safety_flags=safety_flags,
                guardrail_result={},
                retrieved_count=0,
                retrieval_score=0.0,
                raw_reply=CHANNEL_GUIDE_REPLY,
            )

        # ── Phase 3: Hybrid Retrieval ─────────────────────────────────────
        _log("Phase 3: Hybrid Retrieval")
        retrieved = self.retriever.retrieve(
            query=normalized,
            intent_label=intent_pred.label,
            top_k=top_k,
        )
        retrieval_confidence = self.retriever.score_confidence(retrieved)
        _log(f"  Retrieved {len(retrieved)} threads. Confidence: {retrieval_confidence:.3f}")

        # ── Phase 4: Escalation Gate ──────────────────────────────────────
        _log("Phase 4: Escalation Gate")
        escalation = self.gate.decide(
            tweet_text=tweet_text,
            intent_label=intent_pred.label,
            retrieval_confidence=retrieval_confidence,
        )
        action = escalation.action.value
        _log(f"  Decision: {action} | Reason: {escalation.reason}")

        # Early exit for low-confidence escalation (static reply)
        if (
            escalation.triggered_by == "Low retrieval confidence"
            and action == "ESCALATE"
        ):
            t_ms = (time.perf_counter() - t_start) * 1000
            trace_id = self.logger.log(
                input_tweet=tweet_text,
                intent=intent_pred.label,
                intent_confidence=intent_pred.confidence,
                retrieval_score=retrieval_confidence,
                escalation_action=action,
                escalation_reason=escalation.reason,
                final_reply=LOW_CONFIDENCE_FALLBACK,
            )
            return AgentOutput(
                intent=intent_pred.label,
                intent_confidence=intent_pred.confidence,
                action=action,
                reason=escalation.reason,
                public_reply=LOW_CONFIDENCE_FALLBACK,
                internal_handoff=escalation.internal_note,
                trace_id=trace_id,
                latency_ms=t_ms,
                entities=entities,
                safety_flags=safety_flags,
                guardrail_result={},
                retrieved_count=len(retrieved),
                retrieval_score=retrieval_confidence,
                raw_reply=LOW_CONFIDENCE_FALLBACK,
            )

        # ── Phase 5: Reply Synthesis ──────────────────────────────────────
        _log("Phase 5: Reply Synthesis")
        drafter_out = self.drafter.draft(
            tweet=normalized,
            retrieved_results=retrieved,
            action=action,
            intent=intent_pred.label,
            escalation_decision=escalation,
        )
        _log(f"  Draft: {drafter_out.public_reply[:80]}…")

        # ── Phase 6: Safety Guardrails ────────────────────────────────────
        _log("Phase 6: Safety Guardrails")
        guardrail_result = apply_guardrails(drafter_out.public_reply, action=action)
        final_reply = guardrail_result["final_reply"]
        _log(f"  All safe: {guardrail_result['all_safe']}")
        if guardrail_result["applied_fixes"]:
            _log(f"  Fixes: {guardrail_result['applied_fixes']}")

        # ── Trace Log ─────────────────────────────────────────────────────
        t_ms = (time.perf_counter() - t_start) * 1000
        trace_id = self.logger.log(
            input_tweet=tweet_text,
            intent=intent_pred.label,
            intent_confidence=intent_pred.confidence,
            retrieved_threads=[r.thread.thread_id for r in retrieved],
            retrieval_score=retrieval_confidence,
            escalation_action=action,
            escalation_reason=escalation.reason,
            draft_reply=drafter_out.public_reply,
            final_reply=final_reply,
            guardrail_result=guardrail_result,
            metadata={
                "latency_ms": t_ms,
                "entities": entities,
                "safety_flags": safety_flags,
            },
        )

        return AgentOutput(
            intent=intent_pred.label,
            intent_confidence=intent_pred.confidence,
            action=action,
            reason=escalation.reason,
            public_reply=final_reply,
            internal_handoff=drafter_out.internal_handoff,
            trace_id=trace_id,
            latency_ms=t_ms,
            entities=entities,
            safety_flags=safety_flags,
            guardrail_result=guardrail_result,
            retrieved_count=len(retrieved),
            retrieval_score=retrieval_confidence,
            raw_reply=drafter_out.public_reply,
        )

    # ------------------------------------------------------------------
    def run_batch(
        self,
        tweets: list[str],
        top_k:  int = 5,
    ) -> list[AgentOutput]:
        """
        Run the pipeline on a list of tweets (for evaluation harness).
        Note: intent classification is batched for speed; other phases are sequential.
        """
        results: list[AgentOutput] = []
        for i, tweet in enumerate(tweets):
            if self.verbose or (i % 10 == 0):
                print(f"  [{i+1}/{len(tweets)}] Processing tweet…")
            results.append(self.run(tweet, top_k=top_k))
        return results

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  {msg}")
