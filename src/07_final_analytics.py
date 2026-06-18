#!/usr/bin/env python3
"""
Final batch of advanced analyses — 5 follow-ups + 4 Codex recommendations.

A1. Generic-term-controlled Jaccard (decompose convergence)
A2. Entropy decomposition by macro-topic
A3. Primary-category vs all-category robustness
A4. Burst-to-lifecycle mapping
A5. Generalized Synthetic Control (ChatGPT/Transformer counterfactual)
A6. Panel VAR Granger-causality (cross-category lead/lag)
A7. Age-Period-Cohort decomposition (what drives growth?)
A8. Multilayer modularity (bridge topics between categories)

Output: analysis_output/final/
"""

import sys, json, warnings
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd, numpy as np
from scipy import stats, linalg
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import resolve_path, safe_save_csv, setup_logging
logger = setup_logging("final")

OUT = resolve_path("analysis_output/final")
OUT.mkdir(parents=True, exist_ok=True)
from utils import load_config, resolve_path, safe_save_csv, setup_logging, get_categories
config = load_config()
CATS = config.get("categories", ["cs.AI","cs.CL","cs.LG","cs.CV"])

# Load data once
logger.info("Loading data...")
full_df = pd.read_csv(resolve_path("data/raw/arxiv_raw_full.csv"), on_bad_lines="skip")
full_df["year"] = pd.to_datetime(full_df["published"], errors="coerce", utc=True).dt.year
full_df["month_key"] = pd.to_datetime(full_df["published"], errors="coerce", utc=True).dt.strftime("%Y-%m")
logger.info(f"Data: {len(full_df):,} papers")

# ──────────────────────────────────────────────────────────────────
# A1: GENERIC-TERM-CONTROLLED JACCARD
# ──────────────────────────────────────────────────────────────────
def generic_term_jaccard():
    """Recompute Jaccard after removing LLM-era umbrella terms.
    Tests whether convergence is real method sharing or vocabulary diffusion."""
    logger.info("=== A1: Generic-Term-Controlled Jaccard ===")

    # Load previous Jaccard results for comparison
    prev = pd.read_csv(resolve_path("analysis_output/advanced/discipline_convergence.csv"))
    prev["pair"] = prev["cat1"] + "-" + prev["cat2"]

    # Define umbrella terms that may inflate Jaccard
    umbrella_terms = {
        "large language model", "llm", "language model", "transformer",
        "attention", "foundation model", "deep learning", "neural network",
        "pretrained", "fine-tuning", "prompt", "agent", "rag",
        "diffusion model", "self-supervised", "contrastive",
        "multimodal", "vision language", "zero-shot", "few-shot",
        "reinforcement learning", "generative ai", "chatgpt",
    }

    df = full_df.copy()
    rows = []

    for y in range(config.get("eras",{}).get("early",[2000])[0], int(config.get("end_date","2026")[:4])+1):
        year_data = df[df["year"] == y]
        if len(year_data) < 100: continue

        cat_keywords_raw = {}
        cat_keywords_filtered = {}

        for cat in CATS:
            cat_texts = year_data[year_data["query_category"] == cat]["title"].fillna("") + " " + year_data[year_data["query_category"] == cat]["abstract"].fillna("")
            if len(cat_texts) < 50: continue

            vec = TfidfVectorizer(stop_words="english", max_features=100)
            try:
                vec.fit(cat_texts.tolist())
                all_terms = set(vec.get_feature_names_out()[:50])
                cat_keywords_raw[cat] = all_terms
                # Filter out umbrella terms
                cat_keywords_filtered[cat] = {t for t in all_terms if t.lower() not in umbrella_terms and not any(u in t.lower() for u in umbrella_terms) if len(u) > 5}
            except:
                continue

        for i, c1 in enumerate(CATS):
            for j, c2 in enumerate(CATS):
                if i >= j: continue
                if c1 not in cat_keywords_raw or c2 not in cat_keywords_raw: continue

                s1_raw, s2_raw = cat_keywords_raw[c1], cat_keywords_raw[c2]
                jac_raw = len(s1_raw & s2_raw) / len(s1_raw | s2_raw) if len(s1_raw | s2_raw) > 0 else 0

                s1_f, s2_f = cat_keywords_filtered[c1], cat_keywords_filtered[c2]
                jac_filtered = len(s1_f & s2_f) / len(s1_f | s2_f) if len(s1_f | s2_f) > 0 else 0

                rows.append({
                    "year": y, "cat1": c1, "cat2": c2,
                    "jaccard_raw": round(jac_raw, 4),
                    "jaccard_filtered": round(jac_filtered, 4),
                    "umbrella_effect": round(jac_raw - jac_filtered, 4),
                })

    result = pd.DataFrame(rows)
    safe_save_csv(result, str(OUT / "A1_generic_term_jaccard.csv"))

    # Summary
    early = result[result["year"] <= 2010]
    late = result[result["year"] >= 2020]
    logger.info(f"  Raw Jaccard: {early['jaccard_raw'].mean():.3f} → {late['jaccard_raw'].mean():.3f}")
    logger.info(f"  Filtered Jaccard: {early['jaccard_filtered'].mean():.3f} → {late['jaccard_filtered'].mean():.3f}")
    logger.info(f"  Umbrella contribution: {early['umbrella_effect'].mean():.3f} → {late['umbrella_effect'].mean():.3f}")


# ──────────────────────────────────────────────────────────────────
# A2: ENTROPY DECOMPOSITION BY MACRO-TOPIC
# ──────────────────────────────────────────────────────────────────
def entropy_decomposition():
    """Identify which macro-topics caused the macro-distribution entropy change."""
    logger.info("=== A2: Entropy Decomposition ===")

    macro = pd.read_csv(resolve_path("data/results/macro_topic_assignments.csv"))

    # Per-tier paper proportions after aggregating sub-topics into macro-topics.
    rows = []
    for tier in ["T4_historic", "T3_coarse", "T2_medium", "T1_fine"]:
        tier_data = macro[macro["tier"] == tier]
        total_papers = tier_data["paper_count"].sum()
        for mid in sorted(tier_data["macro_id"].unique()):
            mdata = tier_data[tier_data["macro_id"] == mid]
            papers = mdata["paper_count"].sum()
            p = papers / total_papers if total_papers > 0 else 0
            rows.append({"tier": tier, "macro_id": mid, "papers": papers, "proportion": round(p, 6)})

    prop_df = pd.DataFrame(rows)

    # Calculate entropy contribution per macro-topic.
    # Shannon entropy H = -sum(p_m * ln(p_m)); units are nats.
    entropy_rows = []
    for tier in ["T4_historic", "T3_coarse", "T2_medium", "T1_fine"]:
        tdata = prop_df[prop_df["tier"] == tier].copy()
        tdata["entropy_contrib"] = -tdata["proportion"] * np.log(tdata["proportion"] + 1e-10)
        total_h = tdata["entropy_contrib"].sum()
        tdata["entropy_pct"] = (tdata["entropy_contrib"] / total_h * 100).round(1)
        entropy_rows.append(tdata)

    result = pd.concat(entropy_rows, ignore_index=True)
    safe_save_csv(result, str(OUT / "A2_entropy_decomposition.csv"))

    # Show top contributors to entropy change T2→T1
    t2 = result[result["tier"] == "T2_medium"].set_index("macro_id")
    t1 = result[result["tier"] == "T1_fine"].set_index("macro_id")
    macro_ids = sorted(set(t2.index) | set(t1.index))
    changes = []
    for mid in macro_ids:
        delta = t1["entropy_contrib"].get(mid, 0.0) - t2["entropy_contrib"].get(mid, 0.0)
        changes.append((mid, delta, t2["proportion"].get(mid, 0.0), t1["proportion"].get(mid, 0.0)))
    changes.sort(key=lambda x: x[1])
    logger.info("  Top entropy-decreasing macro-topics (T2→T1):")
    for mid, delta, p2, p1 in changes[:3]:
        logger.info(f"    M{mid}: contrib {delta:+.4f} (proportion {p2:.1%}→{p1:.1%})")
    logger.info("  Top entropy-increasing macro-topics (T2→T1):")
    for mid, delta, p2, p1 in changes[-3:]:
        logger.info(f"    M{mid}: contrib {delta:+.4f} (proportion {p2:.1%}→{p1:.1%})")


# ──────────────────────────────────────────────────────────────────
# A3: PRIMARY-CATEGORY ROBUSTNESS
# ──────────────────────────────────────────────────────────────────
def primary_category_robustness():
    """Check how much multi-category tagging inflates convergence."""
    logger.info("=== A3: Primary-Category Robustness ===")

    df = full_df.copy()

    # Count multi-category papers per year
    df["n_categories"] = df["categories"].fillna("").str.count(r"\|") + 1
    yearly_multi = df.groupby("year")["n_categories"].agg(["mean", "std"]).reset_index()
    yearly_multi.columns = ["year", "avg_categories", "std_categories"]
    safe_save_csv(yearly_multi, str(OUT / "A3_multi_category_trend.csv"))

    # Jaccard using ONLY primary_category
    rows = []
    for y in range(config.get("eras",{}).get("early",[2000])[0], int(config.get("end_date","2026")[:4])+1):
        year_data = df[df["year"] == y]
        if len(year_data) < 100: continue

        cat_keywords = {}
        for cat in CATS:
            # Use ONLY papers where this is the PRIMARY category
            primary_only = year_data[(year_data["primary_category"] == cat) & (year_data["query_category"] == cat)]
            texts = primary_only["title"].fillna("") + " " + primary_only["abstract"].fillna("")
            if len(texts) < 50: continue
            vec = TfidfVectorizer(stop_words="english", max_features=100)
            try:
                vec.fit(texts.tolist())
                cat_keywords[cat] = set(vec.get_feature_names_out()[:50])
            except: continue

        for i, c1 in enumerate(CATS):
            for j, c2 in enumerate(CATS):
                if i >= j: continue
                if c1 not in cat_keywords or c2 not in cat_keywords: continue
                s1, s2 = cat_keywords[c1], cat_keywords[c2]
                jac = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0
                rows.append({"year": y, "cat1": c1, "cat2": c2, "jaccard_primary_only": round(jac, 4)})

    result = pd.DataFrame(rows)
    safe_save_csv(result, str(OUT / "A3_primary_category_jaccard.csv"))

    early = result[result["year"] <= 2010]["jaccard_primary_only"].mean()
    late = result[result["year"] >= 2020]["jaccard_primary_only"].mean()
    logger.info(f"  Primary-only Jaccard: {early:.3f} → {late:.3f} (cf. raw 0.214→0.490)")


# ──────────────────────────────────────────────────────────────────
# A4: BURST-TO-LIFECYCLE MAPPING
# ──────────────────────────────────────────────────────────────────
def burst_lifecycle_mapping():
    """Map Kleinberg bursts to macro-topic lifecycle stages."""
    logger.info("=== A4: Burst-Lifecycle Mapping ===")

    bursts = pd.read_csv(resolve_path("analysis_output/advanced/keyword_bursts.csv"))
    lifecycle = pd.read_csv(resolve_path("analysis_output/advanced/topic_lifecycle.csv"))

    # For each burst, determine which macro-topic it likely belongs to
    # by checking which macro-topic's top keywords match the burst term
    macro_kw = pd.read_csv(resolve_path("analysis_output/topics/keyword_frequency_by_tier.csv"))

    rows = []
    for _, b in bursts.iterrows():
        term = b["term"]
        # Find macro-topics where this term appears as a top keyword
        matching = macro_kw[macro_kw["keyword"] == term]
        for tier in matching["tier"].unique():
            # Find macro_id assignments for this tier
            macro_t = pd.read_csv(resolve_path("data/results/macro_topic_assignments.csv"))
            tier_topics = macro_t[macro_t["tier"] == tier]
            for _, tt in tier_topics.iterrows():
                if term in str(tt.get("top5_kw", "")):
                    mid = tt["macro_id"]
                    lf = lifecycle[lifecycle["macro_id"] == mid]
                    if len(lf) > 0:
                        rows.append({
                            "term": term, "tier": tier, "macro_id": mid,
                            "lifecycle": lf.iloc[0]["lifecycle_stage"],
                            "burst_start": b["start"], "burst_peak": b["peak"],
                            "burst_duration": b["duration_months"],
                        })
                    break

    result = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not result.empty:
        safe_save_csv(result, str(OUT / "A4_burst_lifecycle_map.csv"))
        # Cross-tabulation
        ct = pd.crosstab(result["lifecycle"], result["term"]).sum(axis=0)
        logger.info(f"  Burst terms by lifecycle: growing={result[result['lifecycle']=='growing']['term'].nunique()}, mature={result[result['lifecycle']=='mature']['term'].nunique()}, declining={result[result['lifecycle']=='declining']['term'].nunique()}")
    else:
        logger.warning("  No burst-lifecycle mappings found")


# ──────────────────────────────────────────────────────────────────
# A5: GENERALIZED SYNTHETIC CONTROL
# ──────────────────────────────────────────────────────────────────
def synthetic_control():
    """Synthetic control: what would AI research look like without ChatGPT?
    Treats cs.CL (most LLM-exposed) as treated, uses other categories as donors."""
    logger.info("=== A5: Synthetic Control (ChatGPT impact) ===")

    # Monthly paper counts per category
    monthly = full_df.groupby(["query_category", "month_key"]).size().unstack(fill_value=0).T
    monthly.index = pd.to_datetime(monthly.index)
    monthly = monthly.sort_index()
    for c in CATS:
        if c not in monthly.columns: monthly[c] = 0

    # Intervention: ChatGPT release = 2022-11
    pre_period = slice("2018-01", "2022-10")
    sc_cfg = config.get("synthetic_control", {})
    post_period = slice(sc_cfg.get("post_period",["2022-11-01","2026-05-31"])[0][:7], sc_cfg.get("post_period",["2022-11-01","2026-05-31"])[1][:7])

    # Treated: cs.CL (most exposed to LLM)
    treated = monthly["cs.CL"]
    donors = monthly[["cs.AI", "cs.LG", "cs.CV"]]

    # Pre-intervention data for fitting
    Y_pre_treated = treated[pre_period].values
    X_pre_donors = donors[pre_period].values

    # Find optimal donor weights via constrained optimization
    def objective(w):
        synth = X_pre_donors @ w
        return np.sum((Y_pre_treated - synth) ** 2)

    # Constraint: weights sum to 1, all non-negative
    from scipy.optimize import minimize
    n_donors = 3
    w0 = np.ones(n_donors) / n_donors
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_donors)]

    result = minimize(objective, w0, bounds=bounds, constraints=constraints, method="SLSQP")
    weights = result.x

    # Synthetic control
    synthetic = donors.values @ weights
    synthetic_series = pd.Series(synthetic, index=monthly.index)

    # Calculate treatment effect
    pre_mse = np.mean((Y_pre_treated - synthetic_series[pre_period].values) ** 2)
    post_gap = treated[post_period].values - synthetic_series[post_period].values
    avg_effect = np.mean(post_gap)
    cumulative = np.sum(post_gap)

    # Save
    output = pd.DataFrame({
        "month": monthly.index.strftime("%Y-%m"),
        "actual": treated.values,
        "synthetic": synthetic_series.values,
        "gap": treated.values - synthetic_series.values,
    })
    safe_save_csv(output, str(OUT / "A5_synthetic_control.csv"))

    logger.info(f"  Donor weights: cs.AI={weights[0]:.2f}, cs.LG={weights[1]:.2f}, cs.CV={weights[2]:.2f}")
    logger.info(f"  Pre-intervention MSE: {pre_mse:.1f}")
    logger.info(f"  Average post-ChatGPT effect on cs.CL: {avg_effect:+.0f} papers/month")
    logger.info(f"  Cumulative effect (2022-12 to 2026-05): {cumulative:+.0f} total papers")


# ──────────────────────────────────────────────────────────────────
# A6: PANEL VAR GRANGER-CAUSALITY
# ──────────────────────────────────────────────────────────────────
def panel_var_granger():
    """Quarterly Panel VAR + Granger causality between category growth rates."""
    logger.info("=== A6: Panel VAR Granger-Causality ===")

    # Aggregate to quarterly
    df = full_df.copy()
    df["quarter"] = pd.to_datetime(df["published"], errors="coerce", utc=True).dt.to_period("Q").astype(str)
    quarterly = df.groupby(["query_category", "quarter"]).size().unstack(fill_value=0).T
    for c in CATS:
        if c not in quarterly.columns: quarterly[c] = 0
    quarterly = quarterly[CATS].sort_index()

    # Log-difference for stationarity
    growth = np.log(quarterly + 1).diff().dropna()

    # Granger causality tests: does category X (lagged) predict category Y?
    max_lag = 4  # quarters
    granger_results = []

    for y_cat in CATS:
        for x_cat in CATS:
            if x_cat == y_cat: continue
            y = growth[y_cat].values
            x = growth[x_cat].values
            n = len(y)

            # Restricted model: Y ~ lag(Y, 1..4)
            X_restricted = np.column_stack([np.roll(y, i)[max_lag:] for i in range(1, max_lag+1)])
            y_target = y[max_lag:]
            X_restricted = np.column_stack([np.ones(len(y_target)), X_restricted])
            beta_r = np.linalg.lstsq(X_restricted, y_target, rcond=None)[0]
            ssr_r = np.sum((y_target - X_restricted @ beta_r) ** 2)

            # Unrestricted model: add lagged X
            X_unrestricted = np.column_stack([
                np.ones(len(y_target)),
                *[np.roll(y, i)[max_lag:] for i in range(1, max_lag+1)],
                *[np.roll(x, i)[max_lag:] for i in range(1, max_lag+1)],
            ])
            beta_u = np.linalg.lstsq(X_unrestricted, y_target, rcond=None)[0]
            ssr_u = np.sum((y_target - X_unrestricted @ beta_u) ** 2)

            # F-test
            df1 = max_lag
            df2 = n - max_lag - 2 * max_lag - 1
            f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2) if ssr_u > 0 and df2 > 0 else 0
            p_value = 1 - stats.f.cdf(f_stat, df1, df2) if f_stat > 0 else 1.0

            granger_results.append({
                "cause": x_cat, "effect": y_cat, "f_statistic": round(f_stat, 3),
                "p_value": round(p_value, 4), "significant": p_value < 0.05,
            })

    result = pd.DataFrame(granger_results).sort_values("p_value")
    safe_save_csv(result, str(OUT / "A6_granger_causality.csv"))

    sig = result[result["significant"]]
    logger.info(f"  Significant Granger-causal pairs: {len(sig)}/12")
    for _, r in sig.iterrows():
        logger.info(f"    {r['cause']} → {r['effect']}: F={r['f_statistic']:.1f} p={r['p_value']:.4f}")


# ──────────────────────────────────────────────────────────────────
# A7: AGE-PERIOD-COHORT DECOMPOSITION
# ──────────────────────────────────────────────────────────────────
def apc_decomposition():
    """Age-Period-Cohort decomposition of topic growth."""
    logger.info("=== A7: Age-Period-Cohort Decomposition ===")

    # Use macro-topic assignments with tier-based age
    macro = pd.read_csv(resolve_path("data/results/macro_topic_assignments.csv"))

    # Define periods and cohorts
    # Age: how many periods has this topic been active
    # Period: which time period (tier)
    # Cohort: when did the topic first appear (tier of first appearance)
    topic_cohort = macro.groupby("topic_id")["tier"].min().reset_index()
    topic_cohort.columns = ["topic_id", "cohort"]

    apc = macro.merge(topic_cohort, on="topic_id")
    apc["age"] = apc["tier"].map({"T4_historic": 1, "T3_coarse": 2, "T2_medium": 3, "T1_fine": 4})
    apc["period"] = apc["tier"].map({"T4_historic": 1, "T3_coarse": 2, "T2_medium": 3, "T1_fine": 4})
    apc["cohort_num"] = apc["cohort"].map({"T4_historic": 1, "T3_coarse": 2, "T2_medium": 3, "T1_fine": 4})

    # Simple decomposition: regress log(papers) on age + period dummies + cohort dummies
    apc["log_papers"] = np.log(apc["paper_count"] + 1)
    X = pd.get_dummies(apc[["age", "period", "cohort_num"]].astype(int), drop_first=True)
    y = apc["log_papers"]

    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)

    # Extract effects
    coefs = pd.DataFrame({"variable": X.columns, "coefficient": model.coef_}).sort_values("coefficient", ascending=False)

    safe_save_csv(coefs, str(OUT / "A7_apc_coefficients.csv"))
    safe_save_csv(apc, str(OUT / "A7_apc_data.csv"))

    logger.info(f"  APC model R² = {r2:.3f}")
    logger.info(f"  Top positive effects: {coefs.head(3)['variable'].tolist()}")
    logger.info(f"  Top negative effects: {coefs.tail(3)['variable'].tolist()}")


# ──────────────────────────────────────────────────────────────────
# A8: MULTILAYER MODULARITY (simplified)
# ──────────────────────────────────────────────────────────────────
def multilayer_bridge():
    """Identify bridge topics that connect categories, simplified community detection."""
    logger.info("=== A8: Multilayer Bridge Topics ===")

    # For each macro-topic, count how many categories it appears in per tier
    macro = pd.read_csv(resolve_path("data/results/macro_topic_assignments.csv"))

    bridge_rows = []
    for mid in sorted(macro["macro_id"].unique()):
        mdata = macro[macro["macro_id"] == mid]
        for tier in ["T4_historic", "T3_coarse", "T2_medium", "T1_fine"]:
            tdata = mdata[mdata["tier"] == tier]
            cats_present = tdata["tier_label"].nunique() if "tier_label" in tdata.columns else 0
            papers = tdata["paper_count"].sum()

            # Calculate category diversity: how many different categories does this macro-topic span?
            # We use the tier_label as a proxy for category spread (approximate)
            bridge_rows.append({
                "macro_id": mid, "tier": tier, "n_topics": len(tdata),
                "total_papers": papers,
                "bridge_score": len(tdata) * papers,  # simple bridge metric
            })

    result = pd.DataFrame(bridge_rows)
    safe_save_csv(result, str(OUT / "A8_bridge_topics.csv"))

    # Top bridge topics per tier
    for tier in ["T4_historic", "T3_coarse", "T2_medium", "T1_fine"]:
        top = result[result["tier"] == tier].nlargest(3, "bridge_score")
        logger.info(f"  {tier} top bridges: M{top.iloc[0]['macro_id']}, M{top.iloc[1]['macro_id']}, M{top.iloc[2]['macro_id']}")


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────
def main():
    analyses = [
        ("A1 Generic-Term Jaccard", generic_term_jaccard),
        ("A2 Entropy Decomposition", entropy_decomposition),
        ("A3 Primary-Category Robustness", primary_category_robustness),
        ("A4 Burst-Lifecycle Map", burst_lifecycle_mapping),
        ("A5 Synthetic Control", synthetic_control),
        ("A6 Panel VAR Granger", panel_var_granger),
        ("A7 Age-Period-Cohort", apc_decomposition),
        ("A8 Multilayer Bridge", multilayer_bridge),
    ]

    for name, func in analyses:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {name}...")
        try:
            func()
        except Exception as e:
            logger.error(f"  {name} FAILED: {e}")

    logger.info(f"\nAll analyses complete → {OUT}/")
    import os
    for f in sorted(os.listdir(str(OUT))):
        logger.info(f"  {f}")


if __name__ == "__main__":
    main()
