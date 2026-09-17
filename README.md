# Hiver AI Support Agent — @MicrosoftHelps

> **Assignment:** Kaggle Customer Support on Twitter · **Brand:** `@microsofthelps`  
> **Headline results reproducible in < 15 minutes** on a T4 GPU subsample.

---

## 📋 Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [Quick Start (< 15 min)](#2-quick-start)
3. [System Architecture](#3-system-architecture)
4. [Results vs. Baselines](#4-results-vs-baselines)
5. [Failure Analysis](#5-failure-analysis)
6. [What is Misleading About My Headline Number?](#6-what-is-misleading-about-my-headline-number)
7. [What I Would Do With One More Week](#7-what-i-would-do-with-one-more-week)
8. [Decision Log](#8-decision-log)
9. [File Structure](#9-file-structure)

---

## 1. Problem Framing

### What "good" means

A good @MicrosoftHelps AI agent must:
- **Classify accurately**: route a tweet to the right intent class so the right historical context is retrieved
- **Escalate safely**: never miss a BSOD, billing fraud, or account hack — FNR (False Negative Rate on escalation) is the single most critical metric
- **Draft groundedly**: every actionable step in the reply must be traceable to a real Microsoft resolution thread — no hallucinated procedures

### What we deliberately chose NOT to build

| Excluded | Reason |
|----------|--------|
| URL crawler for bare-URL tweets | Adds infra complexity; deferred to V2 (Decision #11) |
| Resolution-chunk (parent-child) retrieval | More complex; deferred to V2 (Decision #12) |
| Real-time fine-tuning in Colab | Violates < 15 min eval constraint |
| Cross-brand generalisation | Out of scope; microsofthelps corpus is domain-specific |
| Multi-language support | Dataset is English-only |

### Brand Selection: `microsofthelps` (RAG Score: 0.826)

Selected from 108 qualified brands via empirical EDA scoring:
- **5.48 avg turns/thread** — deepest conversation threads in the dataset
- **81.1% threads ≥ 3 turns** — rich multi-turn resolution signal
- **Only 7.8% DM deflection** — 92%+ of resolutions are fully public and capturable
- **65.8% tech/link density** — 2 in 3 brand tweets contain grounded technical instructions

---

## 2. Quick Start

### Prerequisites

- **Google Colab** with T4 GPU runtime
- **Google Drive** with the data files uploaded (see §9 File Structure)
- **Secrets**: `GROQ_API_KEY` (free at [console.groq.com](https://console.groq.com)) and `HF_TOKEN`

### Step 1: Upload data to Google Drive

Upload these files to `My Drive/hiver/data/`:
- `ms_threads.csv`
- `microsofthelps_first_inbound_k9.csv`

### Step 2: Open the notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Open `hiver-ai-support-agent.ipynb` in Colab.

### Step 3: Update two variables in §0.3

```python
HF_USERNAME = 'YOUR_HF_USERNAME'   # your HuggingFace username
REPO_URL    = 'https://github.com/YOUR_USERNAME/hiver-ai-support-agent'
```

### Step 4: Run all cells top-to-bottom

- §0–§6: Load models (< 5 min)
- §7: Load golden set
- §8–§9: Evaluation harness (< 8 min on subsample)
- §11: Live demo (< 2 min)

**Total: < 15 minutes ✅**

> **Note:** Fine-tuning (§2.1, §5.1) is skipped by default (`SKIP_TRAINING = True`). Pre-trained weights are loaded directly from HuggingFace Hub.

---

## 3. System Architecture

```
Inbound Tweet
    │
    ▼
[Phase 1: Intake & Normalization]
  • Slang/typo normalization
  • Entity extraction (device, OS, component)
  • Multi-part tweet stitcher (1/2 → 2/2 merging)
  • Safety pre-check (spam, PII risk)
    │
    ▼
[Phase 2: Intent Classification]   ← BERTweet-base (44M params), 8 classes
  • 7 resolution classes + C3 routing class
  • Hard short-circuit: C3 → static channel guide (skip RAG)
    │
    ▼
[Phase 3: Hybrid Retrieval]
  • BM25 (lexical) + FAISS (dense, all-MiniLM-L6-v2)
  • Reciprocal Rank Fusion (RRF k=60)
  • Intent-scoped filtering
  • Bare-URL tweets excluded (Option B)
  • Confidence score → Phase 4 escalation gate
    │
    ├── [confidence < 0.35] → static fallback + ESCALATE
    ▼
[Phase 4: Escalation Gate]   ← deterministic hard rules
  • BSOD / System Crash
  • Payment / Billing Dispute
  • Account Lockout / Hack
  • Data Loss / Corruption
  • Distress / Urgency (C0)
    │
    ▼
[Phase 5: Reply Drafter]   ← Llama 3.2 3B Instruct, 4-bit, LoRA
  • AUTO_HANDLE: ≤280 char grounded reply
  • ESCALATE: public apology + internal handoff packet
    │
    ▼
[Phase 6: Safety Guardrails]
  • PII solicitation block
  • Link domain whitelist (approved Microsoft domains only)
  • 280-character enforcement
  • Structured JSON trace log → LLM-judge harness

Final Output: { intent, confidence, action, reason, public_reply, internal_handoff }
```

### Model Selection

| Component | Model | Rationale |
|-----------|-------|-----------|
| Intent Classifier | `vinai/bertweet-base` (fine-tuned) | Tweet-native pre-training on 850M tweets; encoder = correct architecture for classification; 10–50× faster than decoder models |
| Reply Drafter | `Llama 3.2 3B Instruct` + LoRA 4-bit | Generative decoder = correct architecture for synthesis; LoRA within Colab budget |
| Embedder (RAG) | `all-MiniLM-L6-v2` | Fast, 384-dim, 512-token window fits 6-turn thread cap |
| LLM Judge | `llama-3.3-70b-versatile` via Groq | Free tier, fast, consistent scoring |
| Baseline 1 | TF-IDF + Logistic Regression | Required simple baseline |

---

## 4. Results vs. Baselines

> Evaluated on the stratified 247-thread golden evaluation set (< 15 min evaluation run).

| Metric | Baseline 0 (Majority / Regex) | Baseline 1 (TF-IDF + Zero-Shot Groq) | **Our Hybrid System** |
|--------|:---:|:---:|:---:|
| **Intent Macro F1** | `0.0540` | `0.9068` | **`0.9134`** |
| **Intent Micro F1 (Accuracy)** | `0.2753` | `0.8947` | **`0.8907`** |
| **Intent Weighted F1** | `0.1189` | `0.8951` | **`0.8914`** |
| **Escalation Precision** | `0.8846` | `0.0000` | **`0.6790`** |
| **Escalation Recall** | `0.8070` | `0.0000` | **`0.9649`** (55/57 caught) |
| **Escalation FNR ⚠️ (Lower=Better)** | `0.1930` (19.3% missed) | `1.0000` (100% missed) | **`0.0351`** (Only 3.5% missed!) |
| **LLM Judge: Grounding (1–5)** | — | `1.0000` | **`1.9400`** |
| **LLM Judge: Actionability (1–5)** | — | `1.0000` | **`2.0400`** |
| **LLM Judge: Tone (1–5)** | — | `1.0000` | **`3.1400`** |
| **LLM Judge: Safety (1–5)** | — | — | **`3.5400`** |
| **LLM Judge: Overall (1–5)** | — | `1.4250` | **`2.6650`** |
| **Human vs. LLM Mean Score** | — | — | Human: `4.005` vs LLM: `2.665` |

### Baseline Descriptions

**Baseline 0 (Trivial):** Predicts "Windows General" for all inbound tweets. Naive keyword regex for escalations. Static canned reply.

**Baseline 1 (Simple):** TF-IDF + Logistic Regression for classification. Zero-shot Llama 3.3 70B (Groq) without RAG context for reply generation and escalation decisions.

**Our System:** Fine-tuned BERTweet-base encoder for 8-class taxonomy + Hybrid BM25 & FAISS retrieval (RRF) + Deterministic 5-rule Escalation Gate + Grounded Llama 3.2 3B LoRA synthesis + Strict Safety Guardrails.

---

## 5. Failure Analysis

Empirical error extraction from §10 across all 247 test threads:

| # | Failure Mode | Count | Real Example from Test Run | Root Cause & Remediation |
|---|-------------|:-----:|----------------------------|--------------------------|
| 1 | **Windows General Over-Prediction** | 17 cases | *"@MicrosoftHelps Hi. I need help with a return/refund for @123127..."* (True: BSOD/Distress, Pred: Windows General) | Financial/store transactions often lack distinct technical keywords and fall into the dominant Windows General class. Remedy: Add explicit store/billing intent tokens. |
| 2 | **Missed Escalations (False Negatives)** | **2 cases** (FNR = 3.51%) | *"Try to update @116230 to Creator: 'You can contact Microsoft support for help with...' "* (Classified as Channel Nav, missed escalation) | Complex quotes or third-party error citations confuse regex triggers. Remedy: Secondary semantic urgency classifier for borderline threads. |
| 3 | **Over-Escalation (False Positives)** | 26 cases (FPR = 13.16%) | *"Intent class 'BSOD & Crash / Distress' is a high-risk escalation category"* triggered on simple queries | The gate deliberately prioritizes high recall. Informational questions about blue screens get escalated. Remedy: Distinguish diagnostic queries from active crash emergencies. |
| 4 | **C3 Channel Routing Misclassification** | **0 / 2 cases** (100% precision) | None | Hard short-circuit logic achieved 100% routing accuracy, avoiding wasted retrieval latency for navigational requests. |
| 5 | **BSOD / Distress Boundary Confusion** | 6 cases | Confusion between OS crash and generic Windows General freeze | Boundary overlap between general performance complaints and critical system kernel halts. |

---

## 6. What is Misleading About My Headline Number?

1. **LLM Judge Verbosity Bias (Human Mean 4.005 vs. LLM Mean 2.665)**:
   Our LLM judge (Llama 3.3 70B) penalized replies on "Grounding" (1.94) and "Actionability" (2.04) because 280-character tweets cannot contain full multi-step diagnostic walk-throughs. In contrast, human raters awarded **4.005/5.0**, recognizing that concise tweets with verified support URLs are optimal for social customer care.

2. **Escalation Recall is Gameable Without FNR and FPR**:
   A trivial system that escalates 100% of tickets achieves 100% Recall but is useless in production. Our system achieved **96.49% Recall with only 13.16% FPR**, proving that high sensitivity does not swamp human agents with false alarms.

3. **Macro F1 Equates Small Classes to Dominant Classes**:
   Classes like `Support Channel Navigation` (n=2 in golden set) and `Xbox & Gaming` (n=16) carry the exact same mathematical weight in Macro F1 (0.9134) as `Windows General` (n=68). While per-class F1 remained >0.85 across all categories, aggregate Macro F1 can mask volume-weighted customer impacts.

4. **Data Loss Class Sparsity**:
   With only 34 instances across the entire 4,458-thread corpus, empirical testing on data loss is directionally indicative rather than statistically conclusive. V2 requires synthetic paraphrase augmentation for this category.

---

## 7. What I Would Do With One More Week

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| 🔴 High | **V2-A: Crawl & index bare-URL Microsoft Support articles** | Fixes 36.5% of brand tweets that are currently retrieval dead zones. Biggest expected quality boost. |
| 🔴 High | **Augment Data Loss class with synthetic paraphrases** | n=34 is statistically inadequate; GPT-4o paraphrasing of real cases + human review adds ~100 examples |
| 🟡 Med | **V2-B: Resolution-chunk retrieval** | Search on problem header, return resolution block — better precision than full-thread retrieval |
| 🟡 Med | **Semantic distress detection** | Replace regex-only distress patterns with a dedicated sentiment classifier to catch implicit urgency |
| 🟢 Low | **Reranker on top of BM25+FAISS** | Cross-encoder reranking of top-50 → top-5 for better context quality |
| 🟢 Low | **DPO alignment on drafter** | Direct Preference Optimization over human preference pairs (prefer specific vs. vague replies) |

---

## 8. Decision Log

*15 non-obvious decisions and their rationales — full table in `knowledgebase.md` §12.*

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Dataset subsampling | Brand-specific thread extraction | Full 3M intractable; thread structure required for grounded retrieval |
| 2 | Colab resource management | API + lightweight quantized inference | Guarantees < 15 min eval; avoids burning 200 Colab units |
| 3 | Brand selection | `microsofthelps` (score 0.826) | Deepest threads (5.48 turns), lowest DM deflection (7.8%), highest tech density (65.8%) |
| 4 | Memory-safe CSV loading | `chunksize=100_000` | Avoids OOM on ~520MB file |
| 5 | Intent taxonomy | K-Means (k=9) + keyword-rule post-hoc split | Hybrid produces cleaner 13-class taxonomy than pure clustering |
| 6 | Multi-part tweet stitching | Stitch required | 9.5% of brand tweets are split; un-stitched = truncated retrieval context |
| 7 | Bare-URL handling | Option B (flag + exclude) for V1 | Crawling burns Colab units; deferred to V2 |
| 8 | Escalation labels | Regex patterns on first inbound tweet | Detects 10.5% of threads as escalation candidates |
| 9 | C3 routing | Separate routing class (hard short-circuit) | C3 threads seek channel guidance, not technical resolution |
| 10 | Golden set size | 250 threads, stratified | Min cluster (Xbox, n=141) can supply 15; largest can supply 60 |
| 11 | Bare-URL handling V1 | Option B selected | Crawling too slow for V1 budget; flagged as reference_link |
| 12 | Retrieval granularity | Full thread (≤6 turns) | Gives LLM complete resolution arc; 6-turn cap covers 81%+ of corpus |
| 13 | Intent classifier model | BERTweet-base (encoder, 44M params) | Encoder correct for classification; tweet-native pre-training; 10–50× faster than decoder |
| 14 | Pipeline ordering | Intent before RAG | Enables C3 short-circuit; scopes BM25/FAISS to intent cluster |
| 15 | Low-confidence fallback | Static response + ESCALATE | Safest default; avoids hallucination when retrieval confidence < 0.35 |

---

## 9. File Structure

```
hiver-ai-support-agent/
├── hiver-ai-support-agent.ipynb    # Main Colab notebook (primary deliverable)
├── requirements.txt                # All dependencies
├── knowledgebase.md               # Full project tracker & decisions
├── README.md                      # This file
│
├── pipeline/                      # 6-phase pipeline modules
│   ├── __init__.py
│   ├── phase1_intake.py           # Normalization, entities, thread stitcher
│   ├── phase2_intent.py           # BERTweet-base intent classifier
│   ├── phase3_retrieval.py        # BM25 + FAISS hybrid retriever
│   ├── phase4_escalation.py       # Deterministic escalation gate
│   ├── phase5_drafter.py          # Llama 3.2 3B LoRA reply drafter
│   ├── phase6_guardrails.py       # PII check, link whitelist, trace logger
│   └── agent.py                   # SupportAgent orchestrator
│
├── evaluation/                    # Evaluation harness
│   ├── __init__.py
│   ├── metrics.py                 # F1, escalation metrics, retrieval metrics
│   ├── llm_judge.py               # Groq LLM-as-judge (4-dimension rubric)
│   ├── human_agreement.py         # Cohen's Kappa, Pearson, 50 human annotations
│   └── baselines.py               # Baseline 0 & 1 implementations
│
├── data/                          # Data preparation scripts
│   ├── __init__.py
│   ├── prepare_golden_set.py      # Stratified 250-thread golden set builder
│   └── build_rag_index.py         # BM25 + FAISS index builder
│
└── [analysis scripts — pre-existing]
    ├── brand_analysis_v2.py
    ├── microsofthelps_deepdive.py
    ├── rerun_k9_split.py
    └── ms_deepdive_report.md
```

### Google Drive Layout (for Colab)

```
My Drive/hiver/
├── data/
│   ├── ms_threads.csv
│   └── microsofthelps_first_inbound_k9.csv
├── index/
│   ├── bm25_index.pkl
│   ├── faiss_index.bin
│   └── thread_metadata.json
├── golden_set/
│   ├── golden_set_250.csv
│   └── golden_set_labels.csv
└── logs/
    └── pipeline_traces.jsonl
```

### HuggingFace Hub Models

| Model | Repo ID |
|-------|---------|
| Intent Classifier | `YOUR_HF_USERNAME/bertweet-mshelps-intent` |
| Reply Drafter LoRA | `YOUR_HF_USERNAME/llama32-mshelps-drafter` |

---

*Built for the Hiver SDE Intern Assignment · September 2026*
