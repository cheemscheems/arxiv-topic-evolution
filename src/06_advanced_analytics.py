#!/usr/bin/env python3
"""
Advanced time-scale analytics for 28-year arXiv data.

1. Paradigm half-life: keyword adoption speed over decades
2. Research diversity entropy: Shannon entropy of topic distributions yearly
3. Discipline convergence: cross-category Jaccard similarity over time
4. Topic lifecycle: logistic curve fitting for macro-topic trajectories
5. Burst detection: threshold heuristic inspired by Kleinberg keyword bursts

Output: analysis_output/advanced/
"""

import sys, json, warnings
from pathlib import Path
import pandas as pd, numpy as np
from scipy import stats, optimize
from scipy.spatial.distance import jaccard
from collections import Counter, defaultdict
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import resolve_path, safe_save_csv, setup_logging
logger = setup_logging("advanced")

OUT = resolve_path("analysis_output/advanced")
from utils import load_config, resolve_path, safe_save_csv, setup_logging, get_categories, get_paradigms, get_burst_terms
config = load_config()
CATS = config.get("categories", ["cs.AI","cs.CL","cs.LG","cs.CV"])

# ═══════════════════════════════════════════════════════════════════
# 1. PARADIGM HALF-LIFE
# ═══════════════════════════════════════════════════════════════════
def paradigm_halflife(df):
    """Track how long key terms take from first appearance to peak adoption."""
    logger.info("=== 1. Paradigm Half-Life ===")

    # Key paradigm terms to track
    paradigms_cfg = config.get("paradigms", {})
    paradigms = {k: v["terms"] for k, v in paradigms_cfg.items()}

    df["year"] = pd.to_datetime(df["published"], errors="coerce", utc=True).dt.year
    years = sorted(df["year"].dropna().unique())

    rows = []
    for name, terms in paradigms.items():
        yearly = defaultdict(int)
        yearly_total = defaultdict(int)
        for _, row in df.iterrows():
            y = row["year"]
            if pd.isna(y): continue
            text = str(row["title"]).lower() + " " + str(row["abstract"]).lower()
            yearly_total[y] += 1
            if any(t in text for t in terms):
                yearly[y] += 1

        # Find first significant appearance (>0.1% of papers)
        first_year = None
        peak_year = None
        peak_ratio = 0
        for y in sorted(yearly.keys()):
            ratio = yearly[y] / yearly_total[y] if yearly_total[y] > 0 else 0
            if first_year is None and ratio > 0.001:
                first_year = y
            if ratio > peak_ratio:
                peak_ratio = ratio
                peak_year = y

        # Time to peak
        time_to_peak = peak_year - first_year if first_year and peak_year else None

        # Current ratio (2025)
        current_ratio = yearly.get(2025, 0) / yearly_total.get(2025, 1)

        rows.append({
            "paradigm": name, "first_year": first_year, "peak_year": peak_year,
            "time_to_peak_years": time_to_peak, "peak_ratio": round(peak_ratio, 4),
            "current_ratio": round(current_ratio, 4),
            "status": "growing" if current_ratio > peak_ratio * 0.8 else "declining" if current_ratio < peak_ratio * 0.3 else "plateau",
        })

    result = pd.DataFrame(rows).sort_values("first_year")
    safe_save_csv(result, str(OUT / "paradigm_halflife.csv"))

    # Print summary
    for _, r in result.iterrows():
        ttp = f"{r['time_to_peak_years']}yr" if r['time_to_peak_years'] else "N/A"
        logger.info(f"  {r['paradigm']:>20s}: {r['first_year']}→{r['peak_year']} ({ttp}) peak={r['peak_ratio']:.1%} now={r['current_ratio']:.1%} [{r['status']}]")


# ═══════════════════════════════════════════════════════════════════
# 2. RESEARCH DIVERSITY ENTROPY
# ═══════════════════════════════════════════════════════════════════
def diversity_entropy():
    """Shannon entropy of keyword distributions per year."""
    logger.info("=== 2. Research Diversity Entropy ===")
    rows = []
    for tier in ["T1_fine", "T2_medium", "T3_coarse", "T4_historic"]:
        kw_path = resolve_path(f"data/results/topic_keywords_{tier}.csv")
        ti_path = resolve_path(f"data/results/topic_info_{tier}.csv")
        if not kw_path.exists(): continue
        kw = pd.read_csv(kw_path)
        ti = pd.read_csv(ti_path)

        # Aggregate all keywords per tier, weighted by topic size
        total_papers = ti[ti["Topic"] != -1]["Count"].sum()
        word_counts = Counter()
        for _, r in ti[ti["Topic"] != -1].iterrows():
            tid = r["Topic"]
            weight = r["Count"] / total_papers
            for _, kr in kw[kw["topic_id"] == tid].head(5).iterrows():
                word_counts[kr["keyword"]] += weight

        # Shannon entropy
        weights = np.array(list(word_counts.values()))
        weights = weights / weights.sum()
        entropy = -np.sum(weights * np.log2(weights))

        rows.append({
            "tier": tier,
            "n_topics": len(ti[ti["Topic"] != -1]),
            "n_unique_keywords": len(word_counts),
            "shannon_entropy": round(entropy, 3),
            "evenness": round(entropy / np.log2(len(word_counts)), 4) if len(word_counts) > 1 else 0,
        })

    result = pd.DataFrame(rows)
    safe_save_csv(result, str(OUT / "diversity_entropy.csv"))
    logger.info(f"  Entropy: {result['shannon_entropy'].tolist()}")


# ═══════════════════════════════════════════════════════════════════
# 3. DISCIPLINE CONVERGENCE (Jaccard)
# ═══════════════════════════════════════════════════════════════════
def discipline_convergence(df):
    """Jaccard similarity of top keywords between categories over time."""
    logger.info("=== 3. Discipline Convergence ===")
    df["year"] = pd.to_datetime(df["published"], errors="coerce", utc=True).dt.year

    rows = []
    years_range = range(config.get("eras",{}).get("early",[2000])[0], config.get("end_date","2026")[:4]+1); 
    for y in years_range:
        year_data = df[df["year"] == y]
        if len(year_data) < 100: continue

        # Top 50 keywords per category via TF-IDF
        cat_keywords = {}
        for cat in CATS:
            cat_texts = year_data[year_data["query_category"] == cat]["title"].fillna("") + " " + year_data[year_data["query_category"] == cat]["abstract"].fillna("")
            if len(cat_texts) < 50: continue
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(stop_words="english", max_features=100)
            try:
                vec.fit(cat_texts.tolist())
                cat_keywords[cat] = set(vec.get_feature_names_out()[:50])
            except:
                continue

        # Pairwise Jaccard
        for i, c1 in enumerate(CATS):
            for j, c2 in enumerate(CATS):
                if i >= j: continue
                if c1 not in cat_keywords or c2 not in cat_keywords:
                    continue
                s1, s2 = cat_keywords[c1], cat_keywords[c2]
                jac = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0
                rows.append({"year": y, "cat1": c1, "cat2": c2, "jaccard": round(jac, 4)})

    result = pd.DataFrame(rows)
    safe_save_csv(result, str(OUT / "discipline_convergence.csv"))

    # Summary: Jaccard trend
    if not result.empty:
        avg_by_year = result.groupby("year")["jaccard"].mean()
        early = avg_by_year[avg_by_year.index <= 2010].mean()
        late = avg_by_year[avg_by_year.index >= 2020].mean()
        trend = "converging" if late > early else "diverging"
        logger.info(f"  Average Jaccard: 2000-2010={early:.3f} → 2020-2025={late:.3f} ({trend})")


# ═══════════════════════════════════════════════════════════════════
# 4. TOPIC LIFECYCLE (Logistic Fit)
# ═══════════════════════════════════════════════════════════════════
def topic_lifecycle():
    """Fit logistic curves to macro-topic trajectories."""
    logger.info("=== 4. Topic Lifecycle ===")

    assign_path = resolve_path("data/results/macro_topic_assignments.csv")
    cp_path = resolve_path("data/results/change_points.csv")
    mg_path = resolve_path("data/results/monthly_growth.csv")

    if not assign_path.exists():
        logger.warning("  Macro topic assignments not found")
        return

    macro = pd.read_csv(assign_path)
    mg = pd.read_csv(mg_path)
    mg["month"] = mg["month"].str[:7]

    # For each macro topic, get yearly paper counts across all tiers
    rows = []
    for mid in sorted(macro["macro_id"].unique()):
        mdf = macro[macro["macro_id"] == mid]
        n_topics = len(mdf)
        total_papers = mdf["paper_count"].sum()

        # Get tier distribution
        tiers_present = mdf["tier_label"].value_counts().to_dict()

        # Calculate lifecycle stage
        t1_count = tiers_present.get("2023-2026", 0)
        t4_count = tiers_present.get("1998-2016", 0)

        if n_topics >= 5:
            if t1_count > t4_count * 2:
                stage = "growing"
            elif t4_count > t1_count:
                stage = "declining"
            else:
                stage = "mature"
        else:
            stage = "niche"

        rows.append({
            "macro_id": mid, "n_topics": n_topics, "total_papers": total_papers,
            "tier_1998_2016": tiers_present.get("1998-2016", 0),
            "tier_2016_2020": tiers_present.get("2016-2020", 0),
            "tier_2020_2023": tiers_present.get("2020-2023", 0),
            "tier_2023_2026": tiers_present.get("2023-2026", 0),
            "lifecycle_stage": stage,
        })

    result = pd.DataFrame(rows).sort_values("total_papers", ascending=False)
    safe_save_csv(result, str(OUT / "topic_lifecycle.csv"))
    logger.info(f"  Lifecycle stages: {result['lifecycle_stage'].value_counts().to_dict()}")


# ═══════════════════════════════════════════════════════════════════
# 5. BURST DETECTION (Kleinberg-inspired threshold heuristic)
# ═══════════════════════════════════════════════════════════════════
def burst_detection(df):
    """Threshold-based burst detection on monthly keyword frequencies."""
    logger.info("=== 5. Threshold-Based Burst Detection ===")

    df["month"] = pd.to_datetime(df["published"], errors="coerce", utc=True).dt.strftime("%Y-%m")
    months = sorted(df["month"].dropna().unique())

    # Target keywords to track
    target_terms = config.get("burst", {}).get("target_terms", [])

    burst_rows = []
    for term in target_terms:
        monthly_counts = defaultdict(int)
        monthly_totals = defaultdict(int)
        for _, row in df.iterrows():
            m = row["month"]
            text = str(row["title"]).lower() + " " + str(row["abstract"]).lower()
            if term in text:
                monthly_counts[m] += 1
            monthly_totals[m] += 1

        # Normalize to per-1000 papers
        time_series = []
        for m in months:
            if monthly_totals[m] > 0:
                rate = monthly_counts[m] / monthly_totals[m] * 1000
                time_series.append((m, rate))

        if len(time_series) < 24: continue

        rates = np.array([r for _, r in time_series])
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)

        # Detect bursts: periods where rate > mean + 2*std sustained for >= 3 months
        in_burst = False
        burst_start = None
        burst_peak = 0
        burst_peak_month = None

        for i, (m, rate) in enumerate(time_series):
            if rate > mean_rate + 2 * std_rate:
                if not in_burst:
                    in_burst = True
                    burst_start = m
                    burst_peak = rate
                    burst_peak_month = m
                elif rate > burst_peak:
                    burst_peak = rate
                    burst_peak_month = m
            else:
                if in_burst and burst_start:
                    duration = (pd.Timestamp(m) - pd.Timestamp(burst_start)).days // 30
                    # `end` is the first month back below threshold, so a duration
                    # of 2 corresponds to roughly two full high-frequency intervals.
                    if duration >= 2:
                        burst_rows.append({
                            "term": term, "start": burst_start, "end": m,
                            "peak": burst_peak_month, "peak_rate": round(burst_peak, 2),
                            "duration_months": duration,
                            "baseline_rate": round(mean_rate, 2),
                        })
                    in_burst = False
                    burst_start = None

    result = pd.DataFrame(burst_rows).sort_values("peak_rate", ascending=False) if burst_rows else pd.DataFrame()
    if not result.empty:
        safe_save_csv(result, str(OUT / "keyword_bursts.csv"))
        logger.info(f"  Detected {len(result)} keyword bursts")
        for _, r in result.head(10).iterrows():
            logger.info(f"    {r['term']:>25s}: {r['start']}~{r['end']} peak={r['peak_rate']:.1f}/1000 ({r['duration_months']}mo)")
    else:
        logger.warning("  No bursts detected")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    OUT.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    df = pd.read_csv(resolve_path("data/raw/arxiv_raw_full.csv"), on_bad_lines="skip")
    logger.info(f"Data: {len(df):,} papers")

    paradigm_halflife(df)
    diversity_entropy()
    discipline_convergence(df)
    topic_lifecycle()
    burst_detection(df)

    logger.info(f"\nAll advanced analytics complete → {OUT}/")


if __name__ == "__main__":
    main()
