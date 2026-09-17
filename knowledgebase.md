# Knowledge Base & Project Tracker: Hiver AI Support Agent

> **Status:** Live & Continuously Updated  
> **Environment & Constraints:** Google Colab (200 Compute Units available)  
> **Evaluation Constraint:** Headline results must be reproducible in under 15 minutes. Subsample expected and encouraged.  
> **Key Restriction:** Do NOT open or inspect `hiver-ai-support-agent.ipynb` until explicitly directed.

---

## 1. Assignment Overview & Core Philosophy
* **Core Philosophy:** *"Turn a messy real-world dataset into a working AI system and prove it works. The proof is worth more than the system."*
* **Primary Objective:** Select **one brand** from the Kaggle Customer Support on Twitter dataset and build an AI support agent that performs three core tasks:
  1. **Intent Classification:** Classify incoming customer tweets into a compact, meaningful set of intents defined directly from data.
  2. **Grounded Reply Drafting:** Draft a reply firmly grounded in how that brand has historically resolved similar issues.
  3. **Escalation Routing:** Decide whether to auto-handle or escalate to a human agent, providing an explicit, stated reason.
* **Trust & Evaluation:** Prove the system's reliability with robust evaluation, LLM-as-judge calibration against human ratings, and honest failure analysis.

---

## 2. Hard Deliverables Checklist

| # | Deliverable | Key Requirements | Status |
|---|-------------|------------------|--------|
| 1 | **Runnable Pipeline & Repo** | Self-contained, modular codebase; clear README; reproducible headline results in **< 15 minutes** on a subsample. | Pending |
| 2 | **Golden Evaluation Set** | **150–250 hand-labelled examples**, stratified/curated; documentation on sampling strategy & labeling guidelines. | Pending |
| 3 | **Evaluation Harness** | Automated metrics (classification accuracy/F1, retrieval relevance, escalation precision/recall) + LLM-as-judge rubric for reply quality + **human-judge agreement analysis** (Cohen's Kappa / Pearson correlation / confusion matrix). | Pending |
| 4 | **Comprehensive Report / README** | Max 6 pages (or README section) covering:<br>• Problem framing (what "good" means, what we deliberately chose NOT to build)<br>• Results vs. 2 baselines (Trivial baseline & Simple baseline)<br>• Failure analysis: Top 5 failure modes with real examples & hypotheses<br>• Mandatory section: *"What is misleading about my headline number?"*<br>• What to do next with one more week | Pending |
| 5 | **Decision Log** | Plain list of **10–15 non-obvious decisions** and their rationales. | In Progress |

---

## 3. Hardware & Runtime Constraints (Google Colab Budget)

* **Compute Units:** 200 Colab Compute Units.
  * T4 GPU runs at ~1.5 - 2 units/hour.
  * A100 GPU runs at ~12 - 13 units/hour (use sparingly if needed for fine-tuning or heavy inference).
  * High-RAM CPU runs at ~0.5 - 1 unit/hour.
* **API vs. Local Open-Source LLMs:**
  * Leveraging lightweight API models (e.g. Gemini 1.5 Flash / GPT-4o-mini) or fast open-weights (e.g. Qwen-2.5-7B-Instruct / Llama-3.1-8B-Instruct via vLLM/Groq/Ollama/HuggingFace).
  * API / small quantized models preserve Colab units and ensure the reviewer can reproduce headline results in < 15 minutes without expensive GPU timeouts.
* **Data Volume:** Subsampling the 3M tweet dataset to a high-quality brand-specific subset (e.g. 5,000–20,000 cleaned conversation pairs/threads) keeps embedding generation and indexing fast and economical.

---

## 4. Key Architectural Pillars

```
                     Incoming Customer Tweet
                               │
                               ▼
               ┌───────────────────────────────┐
               │ 1. Intent Classification Gate │
               │   (Domain-specific ontology)  │
               └───────────────┬───────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
   ┌───────────────────────────┐ ┌───────────────────────────┐
   │ 2. Grounded Historical RAG│ │ 3. Policy & Escalation    │
   │  - Clean historical turns │ │    Decision Engine        │
   │  - Dense + Lexical search │ │  - PII / Auth / Account   │
   │  - Resolution extractors  │ │  - High sentiment/urgency │
   └────────────┬──────────────┘ └─────────────┬─────────────┘
                │                             │
                └──────────────┬──────────────┘
                               ▼
               ┌───────────────────────────────┐
               │ 4. Constrained Reply Drafter  │
               │   - Grounded in retrieved ex. │
               │   - Brand tone & guidelines   │
               │   - Auto-handle vs Escalate   │
               └───────────────┬───────────────┘
                               │
                               ▼
                      Final Structured Output
                 { intent, reply, action, reason }
```

---

## 5. Brand Selection — Empirical EDA Findings (`twcs.csv`)

### 5.1 Full Dataset Statistics
- **Total rows:** 2,811,774 tweets streamed via `pd.read_csv(chunksize=100_000)`
- **Unique brand handles:** 108 brands with ≥ 50 brand tweets qualified
- **Analysis scripts:** `brand_analysis_v2.py` → `brand_metrics_scored.csv`

### 5.2 RAG Suitability Scoring Methodology
Each brand was scored on a **weighted composite metric (sum = 1.0)**:

| Metric | Weight | Direction |
|--------|--------|-----------|
| Tech / Link Density | 0.25 | ↑ Higher is better |
| % Threads ≥ 3 Turns | 0.20 | ↑ Higher is better |
| Customer Confirmation Rate | 0.20 | ↑ Higher is better |
| DM Deflection Rate | 0.15 | ↓ Lower is better (penalised) |
| Avg Turns / Thread | 0.10 | ↑ Higher is better |
| Total Brand Tweets (volume) | 0.10 | ↑ Higher is better |
| PII Exposure Rate | 0.10 | ↓ Lower is better (penalised) |

All metrics min-max normalised to [0, 1] before weighting.

### 5.3 Top Brand Comparison Table

| Brand | RAG Score | Brand Tweets | Avg Turns | % ≥3 Turns | DM Deflect % | Tech Density % | Confirm Rate % |
|-------|-----------|-------------|-----------|------------|-------------|----------------|----------------|
| **microsofthelps** ✅ | **0.826** | 11,304 | **5.48** | **81.1%** | 7.8% | 65.8% | 25.4% |
| amazonhelp | 0.720 | 169,840 | 4.52 | 62.1% | 0.8% | 48.8% | 13.1% |
| nikesupport | 0.698 | 3,468 | 4.98 | 63.0% | 11.0% | 58.0% | 21.0% |
| spotifycares | 0.536 | 43,265 | 3.25 | 37.1% | 30.8% | 58.5% | 17.7% |
| applesupport | — | 106,860 | — | — | **52.5%** | 83.0% | — |
| tmobilehelp | — | 34,317 | — | — | **82.1%** | 53.1% | — |
| comcastcares | — | 33,031 | — | — | **71.8%** | 43.6% | — |
| sprintcare | — | 22,381 | — | — | **48.2%** | 25.4% | — |

### 5.4 ✅ Selected Brand: `microsofthelps`

**Rationale:**
- **Deepest conversation threads** in the entire dataset (5.48 avg turns, 81.1% threads ≥ 3 turns) — richest multi-turn training signal.
- **65.8% tech/link density** — nearly 2 in 3 brand tweets contain grounded technical instructions (restart, settings, update, install steps).
- **Only 7.8% DM deflection** — 92%+ of resolutions stay fully public and capturable.
- **25.4% customer confirmation rate** — strong positive resolution signal for training labels.
- **Low PII risk (7.6%)** — minimal data sanitisation burden.

**Brands explicitly rejected:**
- `tmobilehelp`, `comcastcares`, `sprintcare`: DM deflection >48–82% hides most resolution context.
- `spotifycares`: Shallow threads (3.25 avg turns, only 37% ≥ 3 turns), 30.8% DM deflection.
- `applesupport`: 52.5% DM deflection; responses are scripted templates, not substantive troubleshooting.

---

## 6. microsofthelps — Deep-Dive Corpus Statistics

### 6.1 Thread Corpus
| Metric | Value |
|--------|-------|
| Total reconstructed threads | **4,458** |
| Total tweet-turns (all participants) | **24,514** |
| Threads with ≥ 3 turns | **3,618 (81.2%)** |
| Brand tweets (microsofthelps) | **11,304** |
| Threads with any escalation trigger | **470 (10.5%)** |

**Source files:**
- `microsofthelps_dataset.csv` / `.parquet` — full thread data with `cluster_k9`, `cluster_final`, `intent_name` columns
- `microsofthelps_brand_tweets.csv` — brand-only response tweets (11,304 rows)
- `microsofthelps_first_inbound_k9.csv` — first customer tweet per thread with cluster labels

### 6.2 Brand Response Quality Audit

| Signal | Count | % | RAG Design Implication |
|--------|-------|---|------------------------|
| Self-contained instruction tweets | 3,002 | **26.6%** | Core grounded training examples |
| **Bare-URL-only tweets** | 4,125 | **36.5%** | ⚠️ Dead zone — URL content not in CSV; must crawl separately |
| Any URL present | 5,786 | 51.2% | Half of all brand tweets include a link |
| **Multi-part tweets (1/2, 2/2, cont)** | 1,073 | **9.5%** | ✅ Thread stitcher required — see §12, Decision #6 |

### 6.3 Escalation Trigger Audit (first inbound tweet per thread)

| Escalation Type | Threads | % of Corpus |
|----------------|---------|-------------|
| BSOD / System Crash | 244 | 5.47% |
| Payment / Billing Dispute | 123 | 2.76% |
| Account Lockout / Hack | 78 | 1.75% |
| Data Loss / Corruption | 34 | **0.76% ⚠️ sparse** |
| **Any escalation trigger** | **470** | **10.54%** |

> **Note:** Data loss threads (n=34) are critically underrepresented. Supplement with synthetic paraphrases or external forum data during golden set curation.

---

## 7. Intent Taxonomy (k=9 K-Means + C1 Keyword Split)

**Method:** TF-IDF (8k features, 1–2 grams, min_df=5) → L2-normalise → K-Means (k=9, n_init=20) on first inbound tweet per thread. C1 (catch-all, 49.8%) further split by priority-ordered keyword rules.

**Source file:** `ms_intent_clusters_k9_split.csv`

### 7.1 Full 13-Class Taxonomy

| Cluster ID | Intent Name | Size | % | Top Keywords |
|-----------|------------|------|---|-------------|
| `C1_general_os` | 🪟 General Windows / OS Issues | 1,661 | 37.3% | update, win, laptop, pc, computer, fix |
| `C6` | ⬆️ Windows Update & Upgrade Failures | 813 | 18.2% | windows update, install, updates, laptop, time |
| `C8` | 🐛 Generic Error & Troubleshooting | 357 | 8.0% | error, trying, problem, having, hello |
| `C1_office_365` | 📄 Office 365 & Productivity Apps | 316 | 7.1% | outlook, word, onedrive, excel, access |
| `C5` | 🔐 Account Access & Sign-In Issues | 225 | 5.0% | microsoft account, password, access, xbox |
| `C4` | 🖥️ Surface Hardware & Accessories | 201 | 4.5% | surface pro, surface book, pen, screen |
| `C7` | 📦 Feature / Major Release Updates | 193 | 4.3% | fall creators, windows fall, install |
| `C3` | 📞 Support Channel Navigation | 186 | 4.2% | customer service, chat, tech support |
| `C2` | 💻 BSOD / Screen & Display Failures | 181 | 4.1% | blue screen, black screen, fix, laptop |
| `C1_xbox_gaming` | 🎮 Xbox & Gaming Issues | 141 | 3.2% | xbox, minecraft, controller, live |
| `C0` | 🆘 Urgent / Distress Escalation | 83 | 1.9% | need help, asap, help asap |
| `C1_email_account` | 📧 Email, Hotmail & Account Sync | 74 | 1.7% | hotmail, sync, emails, app |
| `C1_store_app` | 🏪 Microsoft Store & App Downloads | 27 | 0.6% | store, download, bought, apps |

### 7.2 Recommended Production Taxonomy (7 Collapsed Classes)

| # | Intent Class | Merged From | Size |
|---|-------------|------------|------|
| 1 | Windows General | C1_general_os + C8 | 2,018 |
| 2 | Windows Update & Install | C6 + C7 | 1,006 |
| 3 | Office & Productivity | C1_office_365 | 316 |
| 4 | Account & Sign-In | C5 + C1_email_account | 299 |
| 5 | Surface Hardware | C4 | 201 |
| 6 | BSOD & Crash / Distress | C2 + C0 | 264 |
| 7 | Xbox & Gaming | C1_xbox_gaming | 141 |

> **Note:** `C3` (Support Channel Navigation, n=186) should be flagged as a **routing/deflection** class, not a resolution class. Treat separately in the escalation gate — these threads indicate the customer cannot find the right support channel and should be auto-routed, not resolved inline.
>
> **Note:** `C1_store_app` (n=27) is too small for standalone evaluation. Merge into Windows General.

---

## 8. Baselines Strategy

1. **Baseline 0 (Trivial Baseline):**
   * *Classification:* Majority class classifier → always predicts "Windows General" (~37%).
   * *Reply:* Static canned template ("Hi! We're sorry to hear that. Please visit support.microsoft.com or DM us for further assistance.").
   * *Escalation:* Rule-based keyword trigger (BSOD/crash/billing/locked keywords → escalate; else auto-handle).

2. **Baseline 1 (Simple / Standard Baseline):**
   * *Classification:* Zero-shot prompt with small model OR TF-IDF + Logistic Regression on 7-class taxonomy.
   * *Reply:* Unconstrained zero-shot LLM generation without RAG grounding ("You are a Microsoft support agent. Reply to this tweet: …").
   * *Escalation:* Naive LLM prompt ("Should this be escalated? Yes or No.") without policy guidance.

3. **Proposed System (Grounded Agentic Pipeline):**
   * Intent classifier with 7-class microsofthelps ontology (from §7.2).
   * Hybrid retrieval (BM25 + dense embeddings) over stitched historical resolved threads.
   * Policy-aware escalation gate (escalates on: BSOD/data loss, billing disputes, account lockout, repeated troubleshooting failure, explicit urgency/distress signals).
   * Hallucination-controlled response drafter with brand voice and grounded step extraction.

---

## 9. Evaluation Rubric & Human-in-the-Loop Calibration

* **Automated Metrics:**
  * Intent Classification: Macro/Micro F1, Precision, Recall, Confusion Matrix on Golden Set.
  * Escalation Decision: Escalation Precision, Recall, False Negative Rate (critical — missing a real escalation is high-cost).
  * Retrieval Relevance: Hit@k, MRR.
* **LLM-as-a-Judge Rubric (Multi-dimensional 1–5 scale):**
  1. *Factual Grounding / Historical Consistency:* Does the reply reflect valid Microsoft procedures without hallucinating policies?
  2. *Tone & Brand Voice:* Is it empathetic, friendly, aligned with @MicrosoftHelps style?
  3. *Actionability:* Does it provide a concrete troubleshooting step or clear next action?
  4. *Safety & Escalation Appropriateness:* Did it correctly escalate sensitive/account queries and auto-handle self-serve queries?
* **Human-Judge Agreement:**
  * Hand-grade a subset (e.g. 50 examples) alongside the LLM Judge.
  * Metrics: Quadratic Weighted Cohen's Kappa, Pearson/Spearman correlation, Exact Match %, Off-by-One tolerance %.

---

## 10. Golden Evaluation Set — Sampling Strategy

**Target:** 150–250 threads. **Available pool:** 4,458 threads (3,618 with ≥ 3 turns).

| Intent Class | Sample Size | Notes |
|-------------|------------|-------|
| Windows General (C1_general_os + C8) | 60 | Stratify by escalation flag |
| Windows Update & Install (C6 + C7) | 40 | Prioritise 3+ turn threads |
| Office & Productivity (C1_office_365) | 25 | |
| Account & Sign-In (C5 + C1_email_account) | 25 | |
| Surface Hardware (C4) | 20 | |
| BSOD & Crash / Distress (C2 + C0) | 30 | Overlap with escalation sample |
| Xbox & Gaming (C1_xbox_gaming) | 15 | |
| **Escalation overlay** (stratified) | **+35** | BSOD×15, Billing×10, Lockout×7, Data Loss×3 |
| **Total** | **~250** | |

---

## 11. RAG Design Decisions & Constraints

### 11.1 Thread Stitcher (Multi-Part Tweets)
- **9.5% of brand tweets** (1,073 / 11,304) contain `1/2`, `2/2`, `(cont)` markers.
- **Decision: Must stitch.** Un-stitched multi-part tweets produce truncated context windows.
- Implementation: consecutive same-author turn merge before chunking, detected via regex `\b\d\s*/\s*\d+\b`.

### 11.2 Bare-URL Tweet Handling
- **36.5% of brand tweets** are bare-URL-only (link to support.microsoft.com docs with no inline text).
- **Decision:** Build a separate Microsoft Support URL knowledge base.
  - Option A *(Recommended)*: Crawl all unique `t.co` / `support.microsoft.com` links in `microsofthelps_dataset.csv`, chunk article content, embed, add to RAG retrieval index.
  - Option B *(Simpler)*: Flag bare-URL tweets as `type=reference_link`; exclude from generative training; keep as retrieval metadata.

### 11.3 C3 Support Channel Navigation
- 186 threads (4.2%) where customers ask *how* to reach support rather than seeking technical resolution.
- **Decision:** Do not include in RAG resolution training. Route directly via intent gate as `channel_routing` → return static channel guide.

---

## 12. Running Decision Log (10–15 Non-Obvious Decisions)

| # | Decision | Options Considered | Selected Choice & Rationale |
|---|----------|--------------------|------------------------------|
| 1 | *Dataset subsampling strategy* | Full 3M tweets vs Random sample vs Brand-specific thread extraction | Extract complete multi-turn conversation threads for a single brand. Full 3M is intractable; thread structure is required for grounded historical resolution. |
| 2 | *Colab resource management* | Heavy local model fine-tuning vs API / lightweight quantized inference | Modular lightweight pipeline (API / quantized local) to guarantee <15 min evaluation runtime and avoid burning 200 Colab units. |
| 3 | *Brand Selection* | SpotifyCares vs AmazonHelp vs AppleSupport vs microsofthelps (and 104 others) | **microsofthelps** (RAG score 0.826). Highest avg turn depth (5.48), best % threads ≥3 turns (81.1%), only 7.8% DM deflection, strong tech density (65.8%). SpotifyCares rejected: shallow threads (3.25 turns), 30.8% DM deflection. |
| 4 | *Memory-safe CSV processing* | `pd.read_csv()` full load vs `chunksize` streaming vs DuckDB/Polars | `pd.read_csv(chunksize=100_000)` — avoids OOM on ~520MB file, maintains pandas compatibility for downstream operations. |
| 5 | *Intent taxonomy method* | Manual labelling vs pure K-Means vs LDA vs hybrid K-Means + keyword rules | K-Means (k=9) on TF-IDF of first inbound tweets, then keyword-rule post-hoc split of catch-all C1 cluster → 13 raw classes → collapsed to 7 production classes. |
| 6 | *Multi-part tweet stitching* | Ignore splits vs stitch adjacent same-author turns | **Stitch required.** 9.5% of brand tweets (1,073) are split across `1/2`/`2/2` pairs; un-stitched = truncated context in retrieval. |
| 7 | *Bare-URL tweet treatment* | Train on them as-is vs filter vs crawl linked content | Crawl & index linked Microsoft Support articles as a separate knowledge base. 36.5% of brand tweets are bare-URL-only — including them without content is a retrieval dead zone. |
| 8 | *Escalation label source* | Manual annotation vs rule-based regex patterns | Regex patterns on first inbound tweet for 4 trigger categories (BSOD, data loss, billing, account lockout). Detects 10.5% of threads as escalation candidates — sufficient for golden set coverage. |
| 9 | *C3 (Support Channel Navigation) treatment* | Include as intent class vs separate routing class | Separate routing class — these threads seek channel guidance, not technical resolution. Including in RAG training contaminates the resolution corpus with non-resolution examples. |
| 10 | *Golden set size & sampling* | 150 vs 250 vs stratified vs random | 250 threads, stratified by 7-class taxonomy + escalation overlay. Min cluster (Xbox, n=141) can supply 15 samples; largest (Windows General, n=2,018) supplies 60. Data loss escalation (n=34) supplemented with synthetic paraphrases. |
| 11 | *Bare-URL tweet handling (V1)* | Option A (crawl) vs Option B (flag + exclude) | Option B selected for V1 — crawling burns Colab units and adds infra complexity; bare-URL tweets flagged as `type="reference_link"` and excluded from generative retrieval. Option A deferred to V2. |
| 12 | *Retrieval granularity* | Turn-level vs Full-thread vs Resolution-chunk | Full-thread selected for V1. Gives LLM the full resolution arc (issue → clarification → fix). Threads capped at 6 turns before embedding to stay within 512-token encoder limit. Resolution-chunk (parent-child) deferred to V2. |
| 13 | *Intent classifier model* | Llama 3.2 3B (decoder) vs BERTweet-base (encoder) | BERTweet-base selected. Encoder architecture is correct for classification (discriminative task). Pre-trained on 850M English tweets = domain-optimal. 44M params vs 3B → 10–50× faster inference, trivially fits <15 min eval constraint. |
| 14 | *Pipeline ordering* | Intent after RAG vs Intent before RAG | Intent classification moved before RAG. Enables hard short-circuit for C3 (channel navigation); scopes BM25/FAISS search to relevant intent cluster; sets escalation prior before retrieval runs. |
| 15 | *Low-confidence retrieval fallback* | Zero-shot LLM call vs Escalate vs Static response | Static response + ESCALATE. Safest default — avoids hallucinated replies when retrieval confidence is below threshold. Static text: "We're looking into this — a specialist will follow up shortly." |

---

## 13. Finalized System Architecture (6-Phase Pipeline)

> **#1 Constraint: Headline results must be fully reproducible in under 15 minutes on a subsample.**
> Every design decision is weighed against this. Complexity that can't be reproduced fast is deferred to the Decision Log.

### 13.1 Pipeline Diagram

```
Inbound Tweet
    │
    ▼
[Phase 1: Intake, Normalization & Thread Stitching]
  • Typo/slang normalization  ("airpods dead on left" → "AirPods left earbud battery failure")
  • Entity extraction (Device, OS, Component)
  • Rule-based safety pre-check
  • Thread stitcher — merge consecutive same-author 1/2, 2/2, (cont) turns before indexing
    RE_PART_START = re.compile(r"\b1\s*/\s*\d+\b|\b1\s+of\s+\d+\b", re.IGNORECASE)
    RE_PART_CONT  = re.compile(r"\b[2-9]\s*/\s*\d+\b|\b[2-9]\s+of\s+\d+\b", re.IGNORECASE)
    │
    ▼
[Phase 2: Intent Classification]  ← runs BEFORE retrieval
  • Model: BERTweet-base (vinai/bertweet-base), fine-tuned for sequence classification
  • Pre-trained on 850M English tweets — domain-optimal for this corpus
  • 7-class microsofthelps taxonomy (§7.2) + C3 as an 8th routing class
  • Hard branch: if C3 (Support Channel Navigation) → return static channel guide, SKIP RAG
  • Sets retrieval scope + escalation prior for downstream phases
  • Baseline 1 comparison: DistilBERT or TF-IDF + Logistic Regression (required by §8)
    │
    ▼
[Phase 3: Hybrid Trajectory Retrieval]
  • BM25 (lexical) + FAISS (dense) over stitched full-thread corpus
  • Bare-URL strategy: Option B — bare-URL tweets marked type="reference_link",
    excluded from generative retrieval, kept as metadata only (Option A deferred to V2)
  • Retrieval granularity: Full Thread (V1)
    → entire stitched thread concatenated → one embedding per thread
    → capped at 6 turns before embedding to prevent silent truncation
    → Option C (resolution-chunk / parent-child) deferred to V2
  • Retrieve top-k full (Customer Issue → Clarification → Resolution) thread flows
  • Relevance confidence score computed per retrieved thread
    │
    ├── [If confidence < threshold] ──→ Static response + flag for ESCALATION
    │                                   ("We're looking into this — a specialist will follow up shortly.")
    ▼
[Phase 4: Escalation Gate]
  • Hard rules (deterministic — no LLM judgment needed):
      BSOD / System Crash      → ESCALATE  (244 threads, 5.47%)
      Payment / Billing        → ESCALATE  (123 threads, 2.76%)
      Account Lockout / Hack   → ESCALATE  (78 threads, 1.75%)
      Data Loss / Corruption   → ESCALATE  (34 threads, 0.76% — sparse, augment golden set)
      Distress / Urgency (C0)  → ESCALATE  (83 threads, 1.9%)
      Low retrieval confidence → ESCALATE  (Phase 3 branch)
  • Everything else            → AUTO_HANDLE
    │
    ▼
[Phase 5: Synthesis & Dual-Channel Output]
  • AUTO_HANDLE:
      → Grounded public tweet reply (≤280 chars)
      → Grounded in retrieved historical resolution steps
      → Brand voice: empathetic, actionable, @MicrosoftHelps tone
  • ESCALATE:
      → Public tweet: short apology (≤280 chars)
      → Internal Handoff Packet: issue summary + retrieval context + escalation reason
  • Generator: Llama 3.2 3B Instruct, 4-bit quantized, LoRA (PEFT) adapter for reply synthesis
    │
    ▼
[Phase 6: Deterministic Safety Guardrails]
  • No PII solicitation check (block "please DM your email/account number" patterns)
  • Link domain whitelist (support.microsoft.com, aka.ms, etc.)
  • Full trace log → feeds LLM-as-judge evaluation harness (§9)
```

---

### 13.2 Model Selection & Rationale

| Component | Model | Architecture | Rationale |
|-----------|-------|-------------|-----------|
| **Intent Classifier** | `vinai/bertweet-base` fine-tuned | Encoder (44M params) | Tweet-native pre-training (850M tweets); encoder = correct architecture for discriminative classification; 10–50× faster inference than decoder models |
| **Reply Drafter** | `Llama 3.2 3B Instruct` + LoRA 4-bit | Decoder (3B params) | Generative decoder = correct architecture for reply synthesis; LoRA keeps fine-tune cost within Colab budget |
| **Baseline 0 classifier** | Majority-class (always "Windows General") | Rule | Required trivial baseline per §8 |
| **Baseline 1 classifier** | `DistilBERT` or TF-IDF + Logistic Regression | Encoder / Classical | Required simple baseline per §8 |
| **Embedder (RAG index)** | `all-MiniLM-L6-v2` | Encoder | Fast, lightweight; 512-token window aligns with 6-turn thread cap |

---

### 13.3 Retrieval Granularity (V1 Decision)

When building the FAISS/BM25 index, each document must be a fixed-size chunk. Three options were considered:

| Option | Unit indexed | Pros | Cons | Status |
|--------|-------------|------|------|--------|
| Turn-level | Single tweet | Precise, fast | No context — fragment is meaningless without surrounding turns | ❌ Rejected |
| **Full thread** | Entire stitched conversation (≤6 turns) | Full resolution arc; simple to implement | Long threads risk encoder truncation — mitigated by 6-turn cap | ✅ **V1 Locked** |
| Resolution chunk | Problem header (search key) + resolution block (returned) | Best precision/context balance | More complex; two FAISS shards | 🔜 Deferred V2 |

**V1 implementation rule:**
```python
def prepare_thread_for_embedding(thread_turns, max_turns=6):
    turns = stitch_multipart(thread_turns)  # Phase 1 stitcher
    turns = turns[:max_turns]               # cap to avoid truncation
    return "\n".join(f"{t['author']}: {t['text']}" for t in turns)
```
A 6-turn cap covers 81%+ of the corpus (avg = 5.48 turns) and keeps every document within the 512-token embedding window.

---

### 13.4 Model & Runtime Budget

| Component | Model | Hardware | Est. Train Time | Est. Colab Units |
|-----------|-------|----------|-----------------|------------------|
| Embedding (index build) | `all-MiniLM-L6-v2` | T4 | ~10 min (4,458 threads) | ~0.3 |
| Intent classifier (fine-tune) | `BERTweet-base` (44M params) | T4 | ~15–20 min | ~0.5 |
| Reply drafter (LoRA fine-tune) | `Llama 3.2 3B Instruct` 4-bit | A100 | ~3–4 hrs | ~40–50 |
| Baseline 1 classifier | `DistilBERT` / TF-IDF+LR | T4/CPU | ~5–10 min | ~0.1 |
| **Headline eval run (subsample)** | All above (inference only) | T4 | **< 15 min** ✅ | ~0.5 |
| **Total (training + eval)** | — | — | — | **~45–55 / 200** |

> **Key rule:** Fine-tuning happens once offline. The <15 min constraint applies to **inference + evaluation on the golden set subsample only**. README must load pre-saved LoRA adapter weights — not retrain from scratch.

---

### 13.5 Deferred to V2 (Future Work & Empirical Findings)

Based on empirical testing, golden set evaluation, and live testing with edge cases, the following architectural upgrades are documented for V2:

| ID | Topic | Priority | Problem Discovered in V1 Testing | V2 Technical Architecture |
|----|-------|:--------:|----------------------------------|---------------------------|
| **V2-A** | **Bare-URL Option A (KB Crawler)** | 🔴 High | 36.5% of historical @MicrosoftHelps tweets only contain a URL with zero explanatory text steps. This creates retrieval "dead zones" for grounded generation. | Crawl & index `support.microsoft.com` / `aka.ms` articles into a secondary FAISS shard. Brand tweet acts as pointer; actual article markdown grounds the generator. |
| **V2-B** | **Turn-Scoped Retrieval (Resolution Pairs)** | 🔴 High | **Multi-turn context leakage:** In live testing on Surface Pen, the agent replied *"Hey! Have you had time to test our suggested steps yet?"* because 6-turn retrieval pulled a mid-ticket follow-up turn into the prompt. | Split index into two tiers: **Tier 1 (Ticket Opening)** indexes `(Turn 0 customer problem → Turn 1 brand solution)` pairs for new inquiries. **Tier 2 (Multi-turn conversation)** is only invoked when customer replies to a thread. |
| **V2-C** | **Anonymized Handle Sanitizer / Dynamic Injector** | 🟡 Med | Generated replies frequently output Kaggle TWCS artifacts (e.g. `@290481`, `@800096 @128530`) because LoRA learned the anonymized numerical handle formatting. | Add a regex cleaning filter in Phase 6 guardrails: strip all leading `@\d+` tokens and dynamically prepend the live inbound user's verified `@handle`. |
| **V2-D** | **Temporal Drift & Dynamic Documentation RAG** | 🔴 High | Queries referencing hardware/software released after late 2017 (e.g. Surface Pro 8 released in 2021, Windows 11 in 2021) have zero corpus coverage, forcing semantic search onto closest historical partial matches. | Add an API fallback (Bing Search API / Microsoft Learn REST API) for entities detected with release dates post-2018. |
| **V2-E** | **Semantic Urgency & Distress Classifier** | 🟡 Med | Failure Mode 2 revealed 2 false negative escalations (FNR = 3.51%) where users quoted complex error messages without using explicit trigger words like "BSOD" or "hack". | Train a compact binary cross-encoder (`distilroberta-base`) to score implicit customer distress/frustration (0–1), triggering escalation when distress > 0.80 regardless of regex matches. |
| **V2-F** | **Escalation False Positive Tuning (Active vs Informational)** | 🟡 Med | 26 over-escalations (FPR = 13.16%) occurred because the keyword gate escalated informational queries mentioning "crash" or "blue screen". | Add a two-stage filter: Stage 1 triggers candidate escalation; Stage 2 checks whether the customer is experiencing an active incident vs. asking an informational/retrospective question. |
| **V2-G** | **Data Loss Augmentation** | 🟢 Low | Only 34 data-loss threads exist across the entire 4,458 corpus (0.76%), making statistical evaluation on this class noisy (n=3 in golden set). | Synthetic paraphrase generation via frontier LLM with human-in-the-loop validation to add ~100 verified data loss threads to the training and evaluation splits. |

