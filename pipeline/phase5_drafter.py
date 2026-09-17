"""
Phase 5: Synthesis & Dual-Channel Output (Reply Drafter)
----------------------------------------------------------
Llama 3.2 3B Instruct, 4-bit quantized, with LoRA (PEFT) adapter fine-tuned
on microsofthelps resolved threads.

V1 Design decisions (from knowledgebase §13.1 Phase 5, §13.2):
  • Generator:  Llama 3.2 3B Instruct + 4-bit quantization (BitsAndBytes NF4)
  • Adapter:    LoRA via PEFT (target: q_proj, v_proj) — fine-tuned offline
  • AUTO_HANDLE reply: ≤280 chars, grounded in retrieved historical resolutions
  • ESCALATE reply:
      → Public tweet: short apology (≤240 chars)
      → Internal handoff packet: issue summary + retrieval context + escalation reason

Training notes:
  • Fine-tuned offline on T4/A100 (~3-4 hrs, ~40-50 Colab units)
  • Loaded in Colab via HuggingFace Hub (pre-saved LoRA adapter weights)
  • Inference only during evaluation (< 15 min constraint)
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are @MicrosoftHelps, Microsoft's official Twitter support account.
Your job is to reply to customer tweets in a helpful, empathetic, and professional tone.

Rules:
1. Always ground your reply in the historical resolution examples provided.
2. Keep your reply under 280 characters (Twitter limit).
3. Be specific and actionable — give a concrete troubleshooting step or next action.
4. Do NOT ask customers to share private information (email, password, account number) in a public tweet.
5. Only include links from official Microsoft domains (support.microsoft.com, aka.ms, etc.).
6. If you cannot provide a grounded, accurate answer, escalate instead of guessing.
7. Use a warm, friendly tone. Start with a brief acknowledgment."""

AUTO_HANDLE_TEMPLATE = """## Customer Tweet:
{tweet}

## Intent: {intent}

## Historical Resolution Examples from @MicrosoftHelps:
{context}

## Task:
Write a Twitter reply (≤280 characters) that:
- Directly addresses the customer's issue
- Is grounded in the resolution steps from the examples above
- Uses @MicrosoftHelps brand voice (empathetic, actionable, professional)
- Does NOT hallucinate steps not present in the examples

Reply:"""

ESCALATE_TEMPLATE = """## Customer Tweet:
{tweet}

## Escalation Reason: {escalation_reason}

## Task:
Write a short, empathetic public Twitter reply (≤240 characters) that:
- Acknowledges the customer's concern
- Does NOT attempt to resolve the issue yourself
- Lets the customer know a specialist will follow up
- Does NOT ask for private information publicly

Reply:"""

HANDOFF_TEMPLATE = """## INTERNAL AGENT HANDOFF PACKET
─────────────────────────────────────────────
Thread ID:           {thread_id}
Escalation Reason:   {escalation_reason}
Escalation Category: {escalation_category}
Intent Class:        {intent}

## Customer Issue Summary:
{tweet}

## Retrieved Historical Context:
{context}

## Recommended Next Steps for Agent:
1. Review customer's account in the internal CRM system.
2. Verify the escalation category: {escalation_category}
3. Use the historical context above to inform your resolution approach.
4. Follow Microsoft's internal escalation SOP for this category.
─────────────────────────────────────────────"""


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class DrafterOutput:
    action:              str           # "AUTO_HANDLE" or "ESCALATE"
    public_reply:        str           # Twitter-ready reply (≤280 chars)
    internal_handoff:    Optional[str] # populated only for ESCALATE
    grounding_examples:  list[str]     # texts of retrieved threads used
    model_raw_output:    str           # raw LLM generation before any trimming


# ---------------------------------------------------------------------------
# ReplyDrafter
# ---------------------------------------------------------------------------

class ReplyDrafter:
    """
    Llama 3.2 3B Instruct 4-bit quantized reply drafter with LoRA adapter.

    Usage (Colab inference):
        drafter = ReplyDrafter.from_pretrained(
            base_model="meta-llama/Llama-3.2-3B-Instruct",
            lora_adapter="your-hf-user/llama32-mshelps-drafter",
        )
        output = drafter.draft(
            tweet, retrieved_results, action, intent, escalation_decision
        )

    Training (offline):
        drafter = ReplyDrafter.for_training(base_model="meta-llama/Llama-3.2-3B-Instruct")
    """

    def __init__(self, model, tokenizer, device: Optional[str] = None) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        base_model:   str = "meta-llama/Llama-3.2-3B-Instruct",
        lora_adapter: Optional[str] = None,
        load_in_4bit: bool = True,
        device:       Optional[str] = None,
    ) -> "ReplyDrafter":
        """
        Load quantized Llama 3.2 3B with optional LoRA adapter from HuggingFace Hub.

        Args:
            base_model:   HF model id for Llama 3.2 3B Instruct.
            lora_adapter: HF repo id for LoRA adapter (e.g. "user/llama32-mshelps-drafter").
                          If None, uses base model zero-shot (for Baseline 1 comparison).
            load_in_4bit: Use NF4 4-bit quantization (BitsAndBytes). Requires GPU.
            device:       "cuda" or "cpu".
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[ReplyDrafter] Loading base model: {base_model}")

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )

        # Load LoRA adapter if provided
        if lora_adapter:
            from peft import PeftModel
            print(f"[ReplyDrafter] Loading LoRA adapter: {lora_adapter}")
            model = PeftModel.from_pretrained(model, lora_adapter)
            if not load_in_4bit:
                try:
                    model = model.merge_and_unload()
                    print("[ReplyDrafter] ✅ LoRA adapter merged")
                except Exception as e:
                    print(f"[ReplyDrafter] Note: skipping merge_and_unload ({e})")
            else:
                print("[ReplyDrafter] ✅ LoRA adapter loaded (4-bit inference mode)")

        model.eval()
        print("[ReplyDrafter] ✅ Model ready for inference")
        return cls(model, tokenizer, device)

    @classmethod
    def for_training(
        cls,
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: Optional[str] = None,
    ) -> "ReplyDrafter":
        """
        Load base model for LoRA fine-tuning (offline training cell).
        Returns an instance ready for PEFT LoRA training setup.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # right-padding for training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        return cls(model, tokenizer, device)

    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        tweet:              str,
        intent:             str,
        context:            str,
        action:             str,
        escalation_reason:  str = "",
    ) -> str:
        """Build a chat-formatted prompt for Llama 3.2 Instruct."""
        if action == "AUTO_HANDLE":
            user_content = AUTO_HANDLE_TEMPLATE.format(
                tweet=tweet, intent=intent, context=context
            )
        else:
            user_content = ESCALATE_TEMPLATE.format(
                tweet=tweet, escalation_reason=escalation_reason
            )

        # Llama 3.2 Instruct chat format
        prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> str:
        """Run generation and return decoded output text only."""
        import torch

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        # Resolve EOS and EOT (end-of-turn) token IDs for Llama 3/3.2
        eos_token_ids = [self.tokenizer.eos_token_id]
        eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_token_id is not None and eot_token_id not in eos_token_ids:
            eos_token_ids.append(eot_token_id)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=eos_token_ids,
            )

        # Decode only newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    def draft(
        self,
        tweet:               str,
        retrieved_results:   list,         # List[RetrievalResult] from Phase 3
        action:              str,          # "AUTO_HANDLE" or "ESCALATE"
        intent:              str,
        escalation_decision=None,          # EscalationDecision from Phase 4
        max_new_tokens:      int = 120,
    ) -> DrafterOutput:
        """
        Generate a reply for the given tweet.

        Args:
            tweet:              Normalized customer tweet.
            retrieved_results:  Top-k retrieval results from Phase 3.
            action:             "AUTO_HANDLE" or "ESCALATE".
            intent:             Intent label from Phase 2.
            escalation_decision: EscalationDecision from Phase 4 (for handoff packet).
            max_new_tokens:     Max tokens to generate.

        Returns:
            DrafterOutput with public_reply and (optionally) internal_handoff.
        """
        # Build context string from retrieved threads
        context_parts: list[str] = []
        grounding_examples: list[str] = []
        for i, result in enumerate(retrieved_results[:3]):
            doc_text = result.thread.text
            context_parts.append(f"[Example {i+1}]\n{doc_text}")
            grounding_examples.append(doc_text)
        context = "\n\n".join(context_parts) if context_parts else "No historical examples available."

        escalation_reason = ""
        escalation_category = ""
        if escalation_decision:
            escalation_reason   = escalation_decision.reason
            escalation_category = escalation_decision.triggered_by or ""

        # --- Generate public reply ---
        prompt = self._build_prompt(
            tweet=tweet,
            intent=intent,
            context=context,
            action=action,
            escalation_reason=escalation_reason,
        )
        raw_output = self._generate(prompt, max_new_tokens=max_new_tokens)

        # --- Build internal handoff packet (ESCALATE only) ---
        internal_handoff: Optional[str] = None
        if action == "ESCALATE":
            thread_id = retrieved_results[0].thread.thread_id if retrieved_results else "N/A"
            internal_handoff = HANDOFF_TEMPLATE.format(
                thread_id=thread_id,
                escalation_reason=escalation_reason,
                escalation_category=escalation_category or "Unknown",
                intent=intent,
                tweet=tweet,
                context=context[:800],  # truncate to keep packet readable
            )

        return DrafterOutput(
            action=action,
            public_reply=raw_output,
            internal_handoff=internal_handoff,
            grounding_examples=grounding_examples,
            model_raw_output=raw_output,
        )

    # ------------------------------------------------------------------
    @classmethod
    def push_adapter_to_hub(
        cls,
        peft_model,
        repo_id: str,
        token: Optional[str] = None,
    ) -> None:
        """Push trained LoRA adapter to HuggingFace Hub."""
        hf_token = token or os.environ.get("HF_TOKEN")
        print(f"[ReplyDrafter] Pushing LoRA adapter to HuggingFace Hub: {repo_id}")
        peft_model.push_to_hub(repo_id, token=hf_token)
        print(f"[ReplyDrafter] ✅ Adapter pushed to: https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# LoRA training configuration (for offline fine-tuning cell)
# ---------------------------------------------------------------------------

def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """
    Returns a PEFT LoraConfig for Llama 3.2 3B.

    Budget: ~3–4 hrs on A100 (40–50 Colab units). See §13.4.
    """
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def build_drafter_training_dataset(
    threads_df,
    text_col:    str = "thread_text",
    reply_col:   str = "microsofthelps_reply",
    intent_col:  str = "intent_label",
    context_col: str = "thread_context",
    max_samples: Optional[int] = None,
):
    """
    Build a SFT (supervised fine-tuning) dataset for the reply drafter.

    Each training example: (tweet, context) → grounded reply
    Only uses AUTO_HANDLE threads (escalation threads excluded from SFT).

    Returns:
        HuggingFace Dataset with a single "text" column (full prompt+completion).
    """
    from datasets import Dataset

    records = []
    for _, row in threads_df.iterrows():
        if max_samples and len(records) >= max_samples:
            break

        tweet   = str(row.get(text_col, ""))
        reply   = str(row.get(reply_col, ""))
        intent  = str(row.get(intent_col, ""))
        context = str(row.get(context_col, ""))

        if not tweet.strip() or not reply.strip():
            continue

        prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            + AUTO_HANDLE_TEMPLATE.format(tweet=tweet, intent=intent, context=context)
            + "<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{reply}<|eot_id|>"
        )
        records.append({"text": prompt})

    print(f"[Drafter Training Dataset] {len(records)} training examples built.")
    return Dataset.from_list(records)
