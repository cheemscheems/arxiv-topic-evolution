#!/usr/bin/env python3
"""Multi-scale topic + change-point analysis for 28-year arXiv data (1998-2026).

Four tiers:
  T1 fine (2023-06 ~ 2026-05): monthly × category, 400/group → 48×400 = 19,200
  T2 medium (2020-06 ~ 2023-05): monthly × category, 400/group → 48×400 = 19,200
  T3 coarse (2016-06 ~ 2020-05): yearly × category, 1500/group → 4y×4c×1500 = 24,000
  T4 historic (1998-06 ~ 2016-05): yearly × category, all papers → ~30,000
  Total: ~92,400

Plus change-point detection on monthly paper counts to find inflection points.
"""

import sys, os, time, signal
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from umap import UMAP
from hdbscan import HDBSCAN
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, ensure_dirs, setup_logging, safe_save_csv, resolve_path

logger = setup_logging("multiscale")
# Force immediate flush on all handlers
for h in logger.handlers:
    if hasattr(h, 'stream'):
        h.stream.reconfigure(line_buffering=True) if hasattr(h.stream, 'reconfigure') else None

# ── Tier Config ──────────────────────────────────────────────────────────
TIERS = {
    "T1_fine": {
        "start": "2023-06", "end": "2026-05",
        "group_by": "month", "sample_per_group": 400,
        "label": "近3年 (2023-2026)",
    },
    "T2_medium": {
        "start": "2020-06", "end": "2023-05",
        "group_by": "month", "sample_per_group": 400,
        "label": "中3年 (2020-2023)",
    },
    "T3_coarse": {
        "start": "2016-06", "end": "2020-05",
        "group_by": "year", "sample_per_group": 1500,
        "label": "早4年 (2016-2020)",
    },
    "T4_historic": {
        "start": "1998-06", "end": "2016-05",
        "group_by": "year", "sample_per_group": 5000,  # effectively all
        "label": "历史基线 (1998-2016)",
    },
}

N_JOBS = config.get("parallel_workers", 1)  # physical cores

TOPIC_PARAMS = dict(
    vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2),
    umap_model=UMAP(n_neighbors=8, n_components=5, min_dist=0.0, metric="cosine",
                    random_state=42, n_jobs=N_JOBS, low_memory=True),
    hdbscan_model=HDBSCAN(min_cluster_size=30, min_samples=10, metric="euclidean",
                           cluster_selection_method="eom", prediction_data=True,
                           core_dist_n_jobs=N_JOBS),
    nr_topics=None, calculate_probabilities=False, verbose=True,
)

TARGET_CATS = ["cs.AI", "cs.CL", "cs.LG", "cs.CV"]


def load_and_filter(csv_path, tier_cfg):
    """Load data, filter time range, clean, dedup."""
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df["published_dt"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    df["month"] = df["published_dt"].dt.strftime("%Y-%m")
    df["year"] = df["published_dt"].dt.strftime("%Y")

    start, end = tier_cfg["start"], tier_cfg["end"]
    df = df[(df["month"] >= start) & (df["month"] <= end)].copy()
    df = df[df["primary_category"].isin(TARGET_CATS)].copy()

    withdrawn = df["title"].fillna("").str.lower().str.contains("withdrawn", na=False)
    withdrawn |= df["abstract"].fillna("").str.lower().str.contains("withdrawn", na=False)
    df = df[~withdrawn]

    df["text"] = df["title"].fillna("") + ". " + df["abstract"].fillna("")

    df["_tk"] = df["title"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip().str[:100]
    df = df.sort_values("published_dt", ascending=False)
    df = df.drop_duplicates(subset="_tk", keep="first")
    df = df.drop(columns=["_tk"])

    logger.info(f"  [{tier_cfg['label']}] {len(df)} papers after cleaning")
    return df


def _sample_one_group(group, target, random_seed):
    """Sample one group with TF-IDF+KMeans diversity. Top-level for joblib."""
    pool = len(group)
    if target >= pool:
        return group
    if pool < 100:
        return group.sample(n=target, random_state=random_seed)

    n_clusters = min(20, max(4, target // 15))
    try:
        vec = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
        X = vec.fit_transform(group["text"].fillna("").tolist())
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_seed, n_init=3)
        labels = km.fit_predict(X)
        parts = []
        for c in range(n_clusters):
            cluster = group[labels == c]
            n_c = max(1, int(target * len(cluster) / pool))
            n_c = min(n_c, len(cluster))
            parts.append(cluster.sample(n=n_c, random_state=random_seed))
        result = pd.concat(parts, ignore_index=True)
        if len(result) > target:
            result = result.sample(n=target, random_state=random_seed)
        return result
    except Exception:
        return group.sample(n=target, random_state=random_seed)


def semantic_sample(df, group_col, n_per_group, random_seed=42):
    """Per-group TF-IDF + MiniBatchKMeans diversity sampling — PARALLEL."""
    groups_list = [(key, group) for key, group in df.groupby(group_col)]
    n_groups = len(groups_list)
    print(f"  Sampling {n_groups} groups in parallel ({N_JOBS} cores)...", flush=True)

    results = Parallel(n_jobs=N_JOBS, verbose=10, batch_size=max(1, n_groups // N_JOBS))(
        delayed(_sample_one_group)(group, min(len(group), n_per_group), random_seed)
        for _, group in groups_list
    )
    return pd.concat(results, ignore_index=True)


def train_topic_model(docs, tier_name):
    """Train BERTopic model."""
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info(f"  [{tier_name}] embeddings ({len(docs)} docs, batch=256, {N_JOBS} cores)...")
    emb = emb_model.encode(docs, batch_size=256, show_progress_bar=True, convert_to_numpy=True)

    logger.info(f"  [{tier_name}] training BERTopic...")
    model = BERTopic(**TOPIC_PARAMS)
    topics, _ = model.fit_transform(docs, emb)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_noise = sum(1 for t in topics if t == -1)
    logger.info(f"  [{tier_name}] topics={n_topics}, noise={n_noise}/{len(topics)} ({n_noise/len(topics)*100:.1f}%)")
    return model, topics, emb


def compute_trend_stats(tier_df, tier_name):
    """Mann-Kendall + Theil-Sen for monthly-tier data."""
    months = sorted(tier_df["month"].unique())
    if len(months) < 6:
        return pd.DataFrame()

    topic_totals = tier_df.groupby("topic_id")["topic_id"].count().sort_values(ascending=False)
    top_topics = topic_totals[topic_totals.index != -1].head(15).index.tolist()

    rows = []
    for tid in top_topics:
        mc = tier_df[tier_df["topic_id"] == tid].groupby("month").size()
        mt = tier_df.groupby("month").size()
        ratios = (mc / mt).reindex(months, fill_value=0.0)

        if ratios.std() < 1e-8:
            continue
        tau, p = stats.kendalltau(range(len(ratios)), ratios.values)
        ts = stats.theilslopes(ratios.values, range(len(ratios)), alpha=0.95)
        rows.append({
            "tier": tier_name, "topic_id": tid,
            "mk_tau": round(tau, 4), "mk_pvalue": round(p, 4),
            "ts_slope": round(ts.slope, 6),
            "ts_slope_lo": round(ts.low_slope, 6), "ts_slope_hi": round(ts.high_slope, 6),
            "significant": p < 0.05,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Change-Point Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_change_points(df: pd.DataFrame):
    """Detect publication volume inflection points using PELT + CUSUM.

    Returns:
      - change_points.csv: per-category inflection dates
      - monthly_growth.csv: monthly paper counts for plotting
    """
    logger.info("\n=== Step 1: Change-Point Detection ===")
    t0 = time.time()

    df = df.copy()
    df["published_dt"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    df["month"] = df["published_dt"].dt.strftime("%Y-%m")
    df = df[df["primary_category"].isin(TARGET_CATS)]
    logger.info(f"  Data prepared: {len(df):,} papers")

    all_points = []
    monthly_records = []

    for cat in tqdm(TARGET_CATS, desc="  Change-point detection"):
        cdf = df[df["query_category"] == cat].copy()
        monthly = cdf.groupby("month").size().sort_index()
        monthly.index = pd.to_datetime(monthly.index)
        monthly = monthly.resample("ME").sum().fillna(0)
        monthly_records.append(pd.DataFrame({"category": cat, "month": monthly.index, "count": monthly.values}))

        y = monthly.values.astype(float)
        if len(y) < 12:
            continue

        n = len(y)
        t = np.arange(n)

        # ── Method 1: PELT (Pruned Exact Linear Time) ──
        # Find changepoints in mean using ruptures library
        cps = []
        try:
            from ruptures import Pelt
            model = Pelt(model="rbf", jump=3, min_size=6).fit(y.reshape(-1, 1))
            bkps = model.predict(pen=3 * np.log(n))
            cps = [monthly.index[i - 1].strftime("%Y-%m") for i in bkps[:-1] if i > 0 and i < n]
        except ImportError:
            logger.warning(f"  ruptures not installed; falling back to CUSUM only")

        # ── Method 2: CUSUM (Cumulative Sum) ──
        mean_y = np.mean(y)
        cusum = np.cumsum(y - mean_y)
        cusum_max = np.argmax(np.abs(cusum))
        # Find additional CUSUM peaks by recursive partitioning
        cusum_peaks = []
        _find_cusum_peaks(y, 0, n - 1, cusum_peaks, min_seg=12, max_depth=3)

        cusum_dates = [monthly.index[p].strftime("%Y-%m") for p in sorted(cusum_peaks) if 1 < p < n - 1]

        # Combine methods
        all_dates = set(cps) | set(cusum_dates)
        for d in sorted(all_dates):
            all_points.append({
                "category": cat, "date": d, "method": "+".join(
                    ["PELT" if d in cps else "", "CUSUM" if d in cusum_dates else ""]
                ).strip("+"),
            })

        logger.info(f"  {cat}: PELT={len(cps)} CUSUM={len(cusum_dates)} → {len(all_dates)} unique inflection points")
        for d in sorted(all_dates)[:8]:
            month_count = monthly.loc[d,].sum() if isinstance(monthly.loc[d,], pd.Series) else monthly.loc[d,]
            logger.info(f"    {d}: {int(monthly.get(d, 0))} papers")

    if all_points:
        cp_df = pd.DataFrame(all_points)
        safe_save_csv(cp_df, str(resolve_path("data/results/change_points.csv")))

    if monthly_records:
        mg_df = pd.concat(monthly_records, ignore_index=True)
        safe_save_csv(mg_df, str(resolve_path("data/results/monthly_growth.csv")))

    logger.info(f"  Change-point detection done in {time.time()-t0:.0f}s. {len(all_points)} points found.")
    return pd.DataFrame(all_points) if all_points else pd.DataFrame()


def _find_cusum_peaks(y, left, right, peaks, min_seg=12, max_depth=3):
    """Recursive CUSUM peak detection."""
    if max_depth <= 0 or right - left < min_seg * 2:
        return
    seg = y[left:right + 1]
    mean_seg = np.mean(seg)
    cusum = np.cumsum(seg - mean_seg)
    idx = np.argmax(np.abs(cusum))
    if min_seg <= idx <= len(seg) - min_seg:
        peaks.append(left + idx)
        _find_cusum_peaks(y, left, left + idx, peaks, min_seg, max_depth - 1)
        _find_cusum_peaks(y, left + idx + 1, right, peaks, min_seg, max_depth - 1)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    config = load_config()
    ensure_dirs()

    csv_path = resolve_path("data/raw/arxiv_raw_full.csv")
    if not csv_path.exists():
        csv_path = resolve_path("data/raw/arxiv_raw_3year.csv")
    if not csv_path.exists():
        csv_path = resolve_path("data/raw/arxiv_raw.csv")

    import os
    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    logger.info(f"Data source: {csv_path} ({size_mb:.0f} MB)")

    # Run analysis
    logger.info("Loading full CSV for change-point detection...")
    t_load = time.time()
    full_df = pd.read_csv(csv_path, on_bad_lines="skip")
    logger.info(f"  CSV loaded: {len(full_df):,} papers [{time.time()-t_load:.0f}s]")

    # ── Step 1: Change-point detection ──
    cp_df = detect_change_points(full_df)

    # ── Step 2: Multi-scale topic analysis ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Multi-Scale Topic Analysis")
    logger.info("=" * 60)

    all_models = {}
    all_trends = []

    for tier_name, tier_cfg in TIERS.items():
        t_tier = time.time()
        logger.info(f"\n{'─'*50}")
        logger.info(f"Tier: {tier_cfg['label']} ({tier_cfg['group_by']}×category, {tier_cfg['sample_per_group']}/group)")
        logger.info(f"{'─'*50}")

        df = load_and_filter(csv_path, tier_cfg)
        if len(df) < 100:
            logger.warning(f"  Too few papers ({len(df)}), skipping")
            continue

        group_col = ["primary_category"]
        group_col.insert(0, tier_cfg["group_by"])

        t_samp = time.time()
        sampled = semantic_sample(df, group_col, tier_cfg["sample_per_group"])
        logger.info(f"  Sampled: {len(sampled)} papers ({len(sampled.groupby(group_col))} groups) [{time.time()-t_samp:.0f}s]")

        sampled["tier"] = tier_name
        safe_save_csv(sampled, str(resolve_path(f"data/processed/sampled_{tier_name}.csv")))

        # Topic modeling
        docs = sampled["text"].fillna("").tolist()
        t_mod = time.time()
        model, topics, _ = train_topic_model(docs, tier_name)
        logger.info(f"  Topic model trained [{time.time()-t_mod:.0f}s]")

        sampled["topic_id"] = topics
        safe_save_csv(sampled, str(resolve_path(f"data/results/topic_assignments_{tier_name}.csv")))

        # Keywords
        topic_info = model.get_topic_info()
        kw_rows = []
        for _, r in topic_info.iterrows():
            tid = r["Topic"]
            if tid == -1: continue
            for rank, (word, weight) in enumerate(model.get_topic(tid), 1):
                kw_rows.append({"topic_id": tid, "rank": rank, "keyword": word, "weight": round(weight, 6)})
        safe_save_csv(pd.DataFrame(kw_rows), str(resolve_path(f"data/results/topic_keywords_{tier_name}.csv")))
        safe_save_csv(topic_info, str(resolve_path(f"data/results/topic_info_{tier_name}.csv")))

        # Trend stats (monthly tiers only)
        if tier_cfg["group_by"] == "month":
            trend_df = compute_trend_stats(sampled, tier_name)
            if not trend_df.empty:
                safe_save_csv(trend_df, str(resolve_path(f"data/results/topic_trend_{tier_name}.csv")))
                all_trends.append(trend_df)

        all_models[tier_name] = {"model": model, "n_topics": len(topic_info[topic_info["Topic"] != -1])}
        logger.info(f"  Tier complete [{time.time()-t_tier:.0f}s total]")

    # ── Cross-tier summary ──
    logger.info(f"\n{'='*60}")
    logger.info("Cross-Tier Summary:")
    for name, info in all_models.items():
        logger.info(f"  {TIERS[name]['label']}: {info['n_topics']} topics")

    if all_trends:
        combined = pd.concat(all_trends, ignore_index=True)
        safe_save_csv(combined, str(resolve_path("data/results/cross_tier_trends.csv")))

    if not cp_df.empty:
        logger.info(f"\nChange points detected: {len(cp_df)}")
        for cat in TARGET_CATS:
            cat_cps = cp_df[cp_df["category"] == cat]["date"].tolist()
            if cat_cps:
                logger.info(f"  {cat}: {', '.join(sorted(cat_cps)[:12])}")

    logger.info(f"\n{'='*60}")
    logger.info(f"All analysis complete. Total time: {time.time()-t_total:.0f}s ({((time.time()-t_total)/60):.1f}min)")
    logger.info(f"Output: data/results/topic_*_{'{tier}'}.csv, data/results/change_points.csv, data/results/monthly_growth.csv")


if __name__ == "__main__":
    main()
