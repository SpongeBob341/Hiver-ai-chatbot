"""
brand_analysis_v2.py  -- FIXED root identification + optimised vectorised counters
"""
import re, collections
import pandas as pd
import numpy as np
from pathlib import Path

CSV_PATH   = Path("twcs/twcs.csv")
OUT_CSV    = Path("brand_metrics.csv")
OUT_PQ     = Path("brand_metrics.parquet")
CHUNK_SIZE = 100_000

RE_DM = re.compile(r"\b(dm|direct\s*message|private\s*message|pm)\b", re.IGNORECASE)
RE_TECH = re.compile(
    r"\b(settings?|restart|reboot|update|upgrade|install|uninstall|steps?|"
    r"troubleshoot|diagnos|reset|refresh|config|configure|enable|disable|"
    r"password|login|log\s*in|app|browser|cache|clear|connect|disconnect|"
    r"network|signal|data|plan|account|bill|charge)\b", re.IGNORECASE)
RE_URL  = re.compile(r"https?://\S+|t\.co/\S+", re.IGNORECASE)
RE_CONFIRM = re.compile(
    r"\b(thank|thanks|thank\s*you|ty|worked|works|fixed|resolved|resolve|"
    r"appreciate|appreciated|perfect|great|awesome|solved|got\s*it|done|"
    r"sorted|all\s*good|no\s*problem|cheers)\b", re.IGNORECASE)
RE_PII = re.compile(
    r"(\d{10,}|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|"
    r"\b(?:email|phone|number|mobile|address|zip|postal|ssn|dob|birth)\b)",
    re.IGNORECASE)

brand_tweet_count = collections.Counter()
dm_deflect_count  = collections.Counter()
tech_link_count   = collections.Counter()
brand_handles     = set()

# tweet_store: tweet_id -> (author_idx, parent, is_inbound, text)
# Use arrays for memory efficiency
tweet_ids_list    = []
tweet_author_list = []
tweet_parent_list = []
tweet_inbound_list= []
tweet_text_list   = []

print("=" * 70)
print("Phase 1 -- Streaming CSV ...")
print("=" * 70)
total_rows = 0

dtype_map = {"tweet_id":"str","author_id":"str","inbound":"str",
             "text":"str","response_tweet_id":"str","in_response_to_tweet_id":"str"}

for chunk_idx, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE,
        dtype=dtype_map, encoding="utf-8", on_bad_lines="skip", low_memory=False)):
    total_rows += len(chunk)
    chunk["text"]      = chunk["text"].fillna("")
    chunk["author_id"] = chunk["author_id"].fillna("").str.strip().str.lower()
    chunk["inbound"]   = chunk["inbound"].map({"True":True,"False":False,True:True,False:False}).fillna(False)
    chunk["tweet_id"]  = pd.to_numeric(chunk["tweet_id"], errors="coerce").fillna(0).astype("int64")
    chunk["parent"]    = pd.to_numeric(chunk["in_response_to_tweet_id"], errors="coerce").fillna(0).astype("int64")

    brand_mask = ~chunk["inbound"].astype(bool)
    brand_handles.update(chunk.loc[brand_mask, "author_id"].unique())

    # Vectorised counters for brand tweets
    brand_chunk = chunk[brand_mask].copy()
    for _, row in brand_chunk.iterrows():
        a = row["author_id"]
        t = row["text"]
        brand_tweet_count[a] += 1
        if RE_DM.search(t):    dm_deflect_count[a]  += 1
        if RE_TECH.search(t) or RE_URL.search(t): tech_link_count[a] += 1

    # Collect tweet store lists
    tweet_ids_list.extend(chunk["tweet_id"].tolist())
    tweet_author_list.extend(chunk["author_id"].tolist())
    tweet_parent_list.extend(chunk["parent"].tolist())
    tweet_inbound_list.extend(chunk["inbound"].tolist())
    tweet_text_list.extend(chunk["text"].tolist())

    if (chunk_idx + 1) % 10 == 0:
        print(f"  Rows: {total_rows:,}  Brands: {len(brand_handles):,}")

print(f"\nTotal rows: {total_rows:,}  Brand handles: {len(brand_handles):,}")

print("\n" + "=" * 70)
print("Phase 2 -- Building tweet store + child index ...")
print("=" * 70)

# Build dict and child index
tweet_store = {}
for i in range(len(tweet_ids_list)):
    tweet_store[tweet_ids_list[i]] = {
        "author":  tweet_author_list[i],
        "inbound": tweet_inbound_list[i],
        "text":    tweet_text_list[i],
        "parent":  tweet_parent_list[i],
    }

# Free raw lists
del tweet_ids_list, tweet_author_list, tweet_parent_list, tweet_inbound_list, tweet_text_list

all_tids = set(tweet_store.keys())

children = collections.defaultdict(list)
for tid, info in tweet_store.items():
    p = info["parent"]
    if p != 0:
        children[p].append(tid)

# FIXED: roots = tweets whose parent is 0 OR parent not in dataset
root_tids = [tid for tid, info in tweet_store.items()
             if info["parent"] == 0 or info["parent"] not in all_tids]
print(f"Roots identified: {len(root_tids):,}  (parent==0 or parent not in dataset)")

print("\n" + "=" * 70)
print("Phase 3 -- Walking threads (BFS) ...")
print("=" * 70)

thread_data = collections.defaultdict(lambda: {"turns":[], "thread_count":0, "confirm_hits":0})

def walk(root):
    q, seq = [root], []
    while q:
        t = q.pop(0)
        if t in tweet_store:
            seq.append(t)
            q.extend(children.get(t, []))
    return seq

n_done = 0
for root_tid in root_tids:
    seq = walk(root_tid)
    if len(seq) < 2:
        continue
    tb = None
    for tid in seq:
        info = tweet_store[tid]
        if not info["inbound"] and info["author"] in brand_handles:
            tb = info["author"]
            break
    if tb is None:
        continue
    thread_data[tb]["turns"].append(len(seq))
    thread_data[tb]["thread_count"] += 1
    last_ib = None
    for tid in reversed(seq):
        if tweet_store[tid]["inbound"]:
            last_ib = tweet_store[tid]["text"]
            break
    if last_ib and RE_CONFIRM.search(last_ib):
        thread_data[tb]["confirm_hits"] += 1
    n_done += 1
    if n_done % 200_000 == 0:
        print(f"  Threads walked: {n_done:,}")

print(f"  Threads walked total: {n_done:,}")

print("\n" + "=" * 70)
print("Phase 4 -- Inbound / PII attribution ...")
print("=" * 70)

brand_inbound_count = collections.Counter()
brand_pii_count     = collections.Counter()

for tid, info in tweet_store.items():
    if not info["inbound"]:
        continue
    p = info["parent"]
    brand_for_this = None
    if p != 0 and p in tweet_store and not tweet_store[p]["inbound"]:
        pa = tweet_store[p]["author"]
        if pa in brand_handles:
            brand_for_this = pa
    if brand_for_this is None:
        for c in children.get(tid, []):
            ca = tweet_store[c]["author"]
            if not tweet_store[c]["inbound"] and ca in brand_handles:
                brand_for_this = ca
                break
    if brand_for_this:
        brand_inbound_count[brand_for_this] += 1
        if RE_PII.search(info["text"]):
            brand_pii_count[brand_for_this] += 1

print("\n" + "=" * 70)
print("Phase 5 -- Assembling metrics ...")
print("=" * 70)

records = []
for brand in sorted(brand_handles):
    bt = brand_tweet_count.get(brand, 0)
    if bt < 50:
        continue
    ib  = brand_inbound_count.get(brand, 0)
    td  = thread_data.get(brand, {})
    tl  = td.get("turns", [])
    tc  = td.get("thread_count", 0)
    ch  = td.get("confirm_hits", 0)
    avg_t = float(np.mean(tl))   if tl else 0.0
    med_t = float(np.median(tl)) if tl else 0.0
    pct3  = (100.0 * sum(1 for t in tl if t >= 3) / len(tl)) if tl else 0.0
    dm_r  = (100.0 * dm_deflect_count.get(brand, 0) / bt) if bt else 0.0
    tec_r = (100.0 * tech_link_count.get(brand, 0)  / bt) if bt else 0.0
    conf_r= (100.0 * ch / tc)                             if tc else 0.0
    pii_r = (100.0 * brand_pii_count.get(brand, 0)  / ib) if ib else 0.0
    records.append({
        "brand":                    brand,
        "total_brand_tweets":       bt,
        "total_inbound_tweets":     ib,
        "thread_count":             tc,
        "avg_turns_per_thread":     round(avg_t,  2),
        "median_turns_per_thread":  round(med_t,  2),
        "pct_threads_3plus_turns":  round(pct3,   2),
        "dm_deflection_rate_pct":   round(dm_r,   2),
        "tech_link_density_pct":    round(tec_r,  2),
        "customer_confirm_rate_pct":round(conf_r, 2),
        "pii_exposure_rate_pct":    round(pii_r,  2),
    })

df_out = pd.DataFrame(records).sort_values("total_brand_tweets", ascending=False)
print(f"\nBrands (>=50 brand tweets): {len(df_out)}")
print(df_out.head(25).to_string(index=False))
df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
df_out.to_parquet(OUT_PQ, index=False)
print(f"\nSaved {OUT_CSV}  |  {OUT_PQ}")

print("\n" + "=" * 70)
print("Phase 6 -- Composite RAG suitability score ...")
print("=" * 70)

def norm(series, higher_is_better=True):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series([0.5]*len(series), index=series.index)
    s = (series - lo) / (hi - lo)
    return s if higher_is_better else (1 - s)

df_s = df_out.copy()
df_s["rag_score"] = (
      0.10 * norm(df_s["total_brand_tweets"])
    + 0.20 * norm(df_s["pct_threads_3plus_turns"])
    + 0.10 * norm(df_s["avg_turns_per_thread"])
    + 0.25 * norm(df_s["tech_link_density_pct"])
    + 0.20 * norm(df_s["customer_confirm_rate_pct"])
    + 0.15 * norm(df_s["dm_deflection_rate_pct"], higher_is_better=False)
    + 0.10 * norm(df_s["pii_exposure_rate_pct"],  higher_is_better=False)
)
df_s = df_s.sort_values("rag_score", ascending=False)

cols = ["brand","total_brand_tweets","thread_count","avg_turns_per_thread",
        "pct_threads_3plus_turns","dm_deflection_rate_pct",
        "tech_link_density_pct","customer_confirm_rate_pct",
        "pii_exposure_rate_pct","rag_score"]
print("\nTop 20 by RAG suitability:")
print(df_s[cols].head(20).to_string(index=False))

df_s.to_csv("brand_metrics_scored.csv", index=False, encoding="utf-8")
df_s.to_parquet("brand_metrics_scored.parquet", index=False)
print("\nSaved brand_metrics_scored.csv  |  brand_metrics_scored.parquet")

top = df_s.iloc[0]
print("\n" + "=" * 70)
print(f"RECOMMENDED BRAND FOR RAG AGENT : {top['brand'].upper()}")
print(f"  RAG Suitability Score          : {top['rag_score']:.4f}")
print(f"  Total Brand Tweets             : {int(top['total_brand_tweets']):,}")
print(f"  Thread Count                   : {int(top['thread_count']):,}")
print(f"  Avg Turns / Thread             : {top['avg_turns_per_thread']:.2f}")
print(f"  % Threads >= 3 Turns           : {top['pct_threads_3plus_turns']:.1f}%")
print(f"  Tech / Link Density            : {top['tech_link_density_pct']:.1f}%")
print(f"  Customer Confirmation Rate     : {top['customer_confirm_rate_pct']:.1f}%")
print(f"  DM Deflection Rate             : {top['dm_deflection_rate_pct']:.1f}%")
print(f"  PII Exposure Rate              : {top['pii_exposure_rate_pct']:.1f}%")
print("=" * 70)
