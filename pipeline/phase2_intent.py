"""
Phase 2: Intent Classification
-------------------------------
BERTweet-base fine-tuned on the 8-class microsofthelps taxonomy:

  7 resolution classes (from knowledgebase §7.2):
    0 — Windows General        (C1_general_os + C8)
    1 — Windows Update & Install (C6 + C7)
    2 — Office & Productivity  (C1_office_365)
    3 — Account & Sign-In      (C5 + C1_email_account)
    4 — Surface Hardware       (C4)
    5 — BSOD & Crash / Distress (C2 + C0)
    6 — Xbox & Gaming          (C1_xbox_gaming)

  1 routing class (hard short-circuit):
    7 — Support Channel Navigation (C3) → skip RAG, return static guide

Decision: Encoder architecture (discriminative) is correct for
          classification. BERTweet pre-trained on 850M English tweets =
          domain-optimal. 44M params → fits < 15 min eval constraint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer


import torch


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

INTENT_LABELS: list[str] = [
    "Windows General",           # 0
    "Windows Update & Install",  # 1
    "Office & Productivity",     # 2
    "Account & Sign-In",         # 3
    "Surface Hardware",          # 4
    "BSOD & Crash / Distress",   # 5
    "Xbox & Gaming",             # 6
    "Support Channel Navigation",# 7 — routing class
]

ID2LABEL: dict[int, str] = {i: label for i, label in enumerate(INTENT_LABELS)}
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(INTENT_LABELS)}

# Intent 7 is a routing class — hard short-circuit, skip RAG
ROUTING_CLASS_ID: int = 7
ROUTING_CLASS_LABEL: str = "Support Channel Navigation"

# Static reply for channel navigation class
CHANNEL_GUIDE_REPLY: str = (
    "Hi! 👋 For direct support, you can reach us at:\n"
    "• 🌐 https://support.microsoft.com\n"
    "• 💬 Virtual Agent: https://aka.ms/MicrosoftVirtualAgent\n"
    "• 📞 Phone/Chat: https://support.microsoft.com/contactus\n"
    "• 🏢 Community: https://answers.microsoft.com\n"
    "We're happy to help find the right channel for you!"
)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class IntentPrediction:
    label:        str
    label_id:     int
    confidence:   float
    all_scores:   dict[str, float]
    is_routing:   bool   # True if this is C3 → hard short-circuit


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    BERTweet-base fine-tuned for 8-class microsofthelps intent classification.

    Usage (Colab):
        classifier = IntentClassifier.from_pretrained("your-hf-username/bertweet-mshelps-intent")
        prediction = classifier.predict("my windows 10 keeps crashing with blue screen")

    Training (offline, see §13.4):
        classifier = IntentClassifier.for_training()
        # Fine-tune with your training loop, then push to HuggingFace Hub
    """

    BASE_MODEL = "vinai/bertweet-base"

    def __init__(self, model, tokenizer, device: Optional[str] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "vinai/bertweet-base",
        num_labels: int = 8,
        device: Optional[str] = None,
    ) -> "IntentClassifier":
        """
        Load a fine-tuned intent classifier from HuggingFace Hub or local path.

        Args:
            model_name_or_path: HF repo id (e.g. "your-user/bertweet-mshelps-intent")
                                or local path to saved model directory.
            num_labels:         Number of output classes (8 for V1 taxonomy).
            device:             "cuda", "cpu", or None (auto-detect).
        """
        print(f"[IntentClassifier] Loading model weights from: {model_name_or_path}")
        print(f"[IntentClassifier] Loading tokenizer from: {cls.BASE_MODEL}")
        
        # ALWAYS load tokenizer from vinai/bertweet-base.
        # BertweetTokenizer requires bpe.codes, which HuggingFace save_pretrained fails
        # to serialize to custom repos. Tokenizers are frozen and never modified during fine-tuning.
        tokenizer = AutoTokenizer.from_pretrained(
            cls.BASE_MODEL,
            normalization=True,
            use_fast=False,
        )

        # Load the fine-tuned classification weights
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
        )
        return cls(model, tokenizer, device)

    @classmethod
    def for_training(
        cls,
        num_labels: int = 8,
        device: Optional[str] = None,
    ) -> "IntentClassifier":
        """
        Initialize from the base BERTweet checkpoint for fine-tuning.
        Use this in the offline training cell of the notebook.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"[IntentClassifier] Initializing base model for training: {cls.BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(
            cls.BASE_MODEL,
            normalization=True,
            use_fast=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            cls.BASE_MODEL,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,  # head is randomly init'd for fine-tuning
        )
        return cls(model, tokenizer, device)

    def predict(
        self,
        text: str,
        max_length: int = 128,
    ) -> IntentPrediction:
        """
        Classify a single tweet text.

        Args:
            text:       Incoming customer tweet (raw or normalized).
            max_length: BERTweet max sequence length (128 recommended).

        Returns:
            IntentPrediction with label, confidence, and all class scores.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        probs_np = probs.cpu().numpy()
        pred_id = int(probs_np.argmax())
        confidence = float(probs_np[pred_id])

        all_scores = {
            ID2LABEL[i]: float(probs_np[i]) for i in range(len(INTENT_LABELS))
        }

        return IntentPrediction(
            label=ID2LABEL[pred_id],
            label_id=pred_id,
            confidence=confidence,
            all_scores=all_scores,
            is_routing=(pred_id == ROUTING_CLASS_ID),
        )

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 128,
    ) -> list[IntentPrediction]:
        """
        Classify a list of tweet texts in mini-batches.

        Args:
            texts:      List of raw/normalized tweet strings.
            batch_size: Number of examples per forward pass.
            max_length: Max token length.

        Returns:
            List of IntentPrediction in same order as input.
        """
        all_predictions: list[IntentPrediction] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs_batch = torch.softmax(outputs.logits, dim=-1)

            for probs in probs_batch:
                probs_np = probs.cpu().numpy()
                pred_id = int(probs_np.argmax())
                confidence = float(probs_np[pred_id])
                all_scores = {
                    ID2LABEL[j]: float(probs_np[j]) for j in range(len(INTENT_LABELS))
                }
                all_predictions.append(
                    IntentPrediction(
                        label=ID2LABEL[pred_id],
                        label_id=pred_id,
                        confidence=confidence,
                        all_scores=all_scores,
                        is_routing=(pred_id == ROUTING_CLASS_ID),
                    )
                )

        return all_predictions

    def push_to_hub(self, repo_id: str, token: Optional[str] = None) -> None:
        """
        Push fine-tuned model and tokenizer to HuggingFace Hub.

        Args:
            repo_id: e.g. "your-username/bertweet-mshelps-intent"
            token:   HuggingFace API token (or set HF_TOKEN env var).
        """
        hf_token = token or os.environ.get("HF_TOKEN")
        print(f"[IntentClassifier] Pushing to HuggingFace Hub: {repo_id}")
        self.model.push_to_hub(repo_id, token=hf_token)
        self.tokenizer.push_to_hub(repo_id, token=hf_token)
        print(f"[IntentClassifier] ✅ Model pushed to: https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Training helper: build HuggingFace Dataset from microsofthelps data
# ---------------------------------------------------------------------------

def build_training_dataset(
    df,
    text_col: str = "first_inbound_text",
    label_col: str = "intent_label_id",
    test_size: float = 0.15,
    val_size: float = 0.10,
    seed: int = 42,
):
    """
    Build a HuggingFace Dataset split from a pandas DataFrame.

    Expected columns:
        text_col:   Raw first-inbound tweet text.
        label_col:  Integer class ID (0–7).

    Returns:
        DatasetDict with "train", "validation", "test" splits.
    """
    from datasets import Dataset, DatasetDict
    from sklearn.model_selection import train_test_split

    df = df[[text_col, label_col]].dropna().copy()
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].astype(int)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size / (1 - test_size),
        random_state=seed,
        stratify=train_df["label"],
    )

    return DatasetDict(
        {
            "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
            "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
        }
    )


def get_training_args(
    output_dir: str = "./bertweet_intent_checkpoints",
    num_epochs: int = 4,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
):
    """
    Returns HuggingFace TrainingArguments optimised for BERTweet fine-tuning on T4.

    Budget estimate: ~15–20 min on T4 (≈0.5 Colab units).
    """
    import inspect
    import dataclasses
    from transformers import TrainingArguments

    dc_fields = {f.name for f in dataclasses.fields(TrainingArguments)} if dataclasses.is_dataclass(TrainingArguments) else set()
    sig_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    valid_params = (dc_fields | sig_params) - {"args", "kwargs"}

    training_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": 64,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "logging_steps": 50,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",   # set to "wandb" if tracking enabled
        "push_to_hub": False,  # push manually after training via push_to_hub()
    }

    # transformers >= 4.41 deprecated evaluation_strategy in favor of eval_strategy
    if "eval_strategy" in valid_params:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in valid_params:
        training_kwargs["evaluation_strategy"] = "epoch"

    # Prefer warmup_steps (universally supported across transformers versions)
    if "warmup_steps" in valid_params:
        training_kwargs["warmup_steps"] = 50
    elif "warmup_ratio" in valid_params:
        training_kwargs["warmup_ratio"] = 0.1

    # Only pass parameters accepted by this environment's TrainingArguments
    filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in valid_params}

    return TrainingArguments(**filtered_kwargs)
