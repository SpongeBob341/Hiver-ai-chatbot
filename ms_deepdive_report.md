# `microsofthelps` Deep-Dive Diagnostic Report
**Threads reconstructed:** 4,458 · **Total tweet-turns:** 24,514 · **Threads with 3+ turns:** 3,618 (81.2%)

---

## 1 · Intent Taxonomy — 6 Discovered Clusters

> [!IMPORTANT]
> Cluster 1 is a **catch-all bucket (53.8%)** — too broad for clean intent separation. Recommend re-running with **k=8 or k=9** before finalising taxonomy, or manually splitting C1 using keyword heuristics post-hoc.

| Cluster | Size | % | Top-10 Keywords | Recommended Intent Name |
|---------|------|---|-----------------|------------------------|
| **C0** | 235 | 5.3% | surface, pro, surface pro, surface book, screen, book, pen, battery, new, help | 🖥️ **Surface Hardware & Device Issues** |
| **C1** | 2,400 | 53.8% | update, microsoft, account, email, win, laptop, xbox, support, hi, pc | ⚠️ **Mixed / General — needs k-split** |
| **C2** | 306 | 6.9% | just, windows, update, updated, just updated, computer, got, new, pc, just got | 🔄 **Post-Update Regression (broke after update)** |
| **C3** | 193 | 4.3% | creators, creators update, fall creators, fall, windows fall, windows creators, install | 📦 **Feature Update Installation (Creators / major releases)** |
| **C4** | 856 | 19.2% | windows, update, windows update, updates, laptop, pc, update windows, install, time | ⬆️ **Windows Update & Upgrade Failures** |
| **C5** | 468 | 10.5% | help, need, need help, account, hi, trying, error, having, microsoft, windows | 🔐 **Account Access, Sign-In & Error States** |

### Recommended Final Taxonomy (7 classes)

Once C1 is split, the natural taxonomy that emerges:

| # | Intent Class | Description |
|---|-------------|-------------|
| 1 | **Windows Update & Upgrade** | Stuck updates, failed installs, rollback |
| 2 | **Post-Update Regression** | Something broke after updating |
| 3 | **Feature Update (Major Release)** | Creators, Anniversary, 22H2 installs |
| 4 | **Surface Hardware & Peripherals** | Surface Pro/Book/Pen, screen, battery |
| 5 | **Account & Sign-In Issues** | Login failures, password, 2FA, lockout |
| 6 | **General Error & Troubleshooting** | Error codes, crashes, generic help requests |
| 7 | **Subscription & Xbox / Office** | Xbox Live, Microsoft 365, billing (split from C1) |

---

## 2 · Brand Response Quality Audit

**Total `microsofthelps` brand tweets:** 11,304

| Metric | Count | % | Interpretation |
|--------|-------|---|----------------|
| Self-contained instruction tweets | 3,002 | **26.6%** | ⚠️ LOW — only 1 in 4 tweets has step-by-step guidance |
| Bare-URL-only tweets | 4,125 | **36.5%** | 🔴 HIGH — over a third are just a link, no retrievable text |
| Any URL present | 5,786 | **51.2%** | Half of all brand tweets include a URL |
| Multi-part tweet markers (`1/2`, `2/2`, `(cont)`) | 1,073 | **9.5%** | ✅ Significant — must stitch |

> [!WARNING]
> **36.5% bare-URL tweets are a RAG retrieval dead zone.** The linked content (support articles, docs pages) is not captured in the CSV. Your RAG pipeline must either:
> (a) **Crawl and index** the linked Microsoft Support URLs as a separate knowledge base, or
> (b) **Filter out** bare-URL tweets during training and treat linked articles as separate retrieval chunks.

### Sample Multi-Part Tweets (confirms stitching requirement)
```
[1/2] Hi there, Bill! We're here to help. Just to clarify, where are you 
      trying to download an app? Is it from the Store app or via Microsoft Store online?
[2/2] Store app or via Microsoft Store online?

[1/2] We don't have direct email. You can post your query via Community 
      Forum for assistance: https://t.co/jsa5yePQ4
[2/2] Click "Ask a Question" on the upper right corner. Thank you.
```

---

## 3 · Escalation Trigger Audit

**Scan target:** First inbound tweet of each of the 4,458 threads

| Escalation Type | Thread Count | % of Corpus | Edge Case Coverage |
|----------------|-------------|-------------|-------------------|
| 🔴 BSOD / System Crash | **244** | **5.47%** | ✅ Sufficient (>200) |
| 💳 Payment / Billing Dispute | **123** | **2.76%** | ✅ Sufficient (>100) |
| 🔒 Account Lockout / Hack | **78** | **1.75%** | ⚠️ Borderline (aim for >100) |
| 💾 Data Loss / Corruption | **34** | **0.76%** | 🔴 Sparse — augment manually |
| **ANY escalation trigger** | **470** | **10.54%** | ✅ Rich escalation pool |

> [!NOTE]
> **Data loss threads (34) are critically underrepresented.** These are high-stakes scenarios for a support agent. Consider supplementing with synthetic paraphrases or pulling from external forums during golden set curation.

---

## 4 · RAG Design Recommendations

### 4a · Thread Stitcher — Multi-Part Tweets

**✅ YES — your thread builder MUST stitch multi-part tweets.**

- **9.5% of brand tweets** (1,073 of 11,304) contain `1/2`, `2/2`, `(cont)`, or similar markers.
- Failing to stitch these produces truncated, semantically incomplete context windows — exactly the wrong training signal for a RAG generator.

**Implementation rule:**
```python
# Stitch logic: if tweet_n contains "1/2" or "1 of 2",
# and tweet_{n+1} (same author, consecutive turn) contains "2/2",
# concatenate text before chunking.
RE_PART_START = re.compile(r"\b1\s*/\s*\d+\b|\b1\s+of\s+\d+\b", re.IGNORECASE)
RE_PART_CONT  = re.compile(r"\b[2-9]\s*/\s*\d+\b|\b[2-9]\s+of\s+\d+\b", re.IGNORECASE)
```

### 4b · Bare-URL Handling Strategy

```
Option A (Recommended): Build a Microsoft Support URL knowledge base
  → Crawl all unique t.co / support.microsoft.com links in ms_threads.csv
  → Chunk article content → embed → add to RAG retrieval index
  → Brand tweet becomes a "pointer" retrieved doc; article is the grounded answer

Option B (Simpler): Filter + flag
  → Mark bare-URL tweets as type="reference_link"
  → Exclude from generative training; keep in retrieval index as metadata
```

---

## 5 · Golden Evaluation Set — Feasibility ✅ CONFIRMED

| Criterion | Value | Status |
|-----------|-------|--------|
| Total threads available | 4,458 | ✅ |
| Multi-turn threads (≥3 turns) | **3,618** (81.2%) | ✅ Excellent depth |
| Escalation threads (edge cases) | **470** | ✅ |
| Smallest cluster (C3) | 193 threads | ✅ Enough to sample from |
| Target golden set | 150–250 threads | ✅ |

### Recommended Sampling Strategy (250 threads)

| Intent Class | Sample Size | Source |
|-------------|------------|--------|
| Windows Update & Upgrade (C4) | 45 | Random stratified from C4 |
| Post-Update Regression (C2) | 30 | Random from C2 |
| Feature Update / Major Release (C3) | 25 | All 193 → sample 25 |
| Surface Hardware (C0) | 25 | Random from C0 |
| Account & Sign-In (C5) | 35 | Weighted toward 3+ turn threads |
| General Error / Mixed (C1 subset) | 40 | Random from C1 |
| **Escalation edge cases** (overlay) | **50** | Stratified from 470 escalation threads |
| **Total** | **250** | |

> [!TIP]
> For the 50 escalation samples, stratify as: BSOD×20, Billing×15, Account Lockout×10, Data Loss×5. This ensures all four escalation types are represented even where the corpus is thin.

---

## 6 · Output Files

| File | Description |
|------|-------------|
| [`ms_threads.csv`](file:///c:/Users/divya/Desktop/Hiver/ms_threads.csv) | All 24,514 tweet-rows with `thread_id`, `turn_pos`, `cluster` |
| [`ms_threads.parquet`](file:///c:/Users/divya/Desktop/Hiver/ms_threads.parquet) | Same, Parquet |
| [`ms_intent_clusters.csv`](file:///c:/Users/divya/Desktop/Hiver/ms_intent_clusters.csv) | 6-cluster summary with top-10 keywords |
| [`ms_brand_quality.csv`](file:///c:/Users/divya/Desktop/Hiver/ms_brand_quality.csv) | Brand response quality metrics |
| [`ms_escalation_summary.csv`](file:///c:/Users/divya/Desktop/Hiver/ms_escalation_summary.csv) | Escalation trigger counts |
| [`microsofthelps_deepdive.py`](file:///c:/Users/divya/Desktop/Hiver/microsofthelps_deepdive.py) | Reproducible analysis script |
