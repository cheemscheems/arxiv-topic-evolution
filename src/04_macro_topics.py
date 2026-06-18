#!/usr/bin/env python3
"""Aggregate 929 topics across 4 tiers into ~15 macro-topics via cross-tier keyword clustering.

Strategy: For each topic (any tier), create a "topic signature" from its top-10 keywords.
Cluster these signatures using SBERT embeddings into macro categories.
Each macro category gets a descriptive label and spans multiple tiers.
"""

import sys, json
from pathlib import Path
import pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist, squareform
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import setup_logging, resolve_path, safe_save_csv

logger = setup_logging("macro_topics")

# Load tiers from config.yaml
def _load_tier_info():
    from utils import load_config
    cfg = load_config()
    tiers = cfg.get("tiers", {})
    return list(tiers.keys()), {k: v["label"] for k, v in tiers.items()}

TIERS, TIER_LABELS = _load_tier_info()
N_MACRO = 16  # target number of macro topics


def load_all_topics():
    """Load all topic keywords + info from all tiers."""
    all_topics = []
    for tier in TIERS:
        kw = pd.read_csv(resolve_path(f"data/results/topic_keywords_{tier}.csv"))
        ti = pd.read_csv(resolve_path(f"data/results/topic_info_{tier}.csv"))

        for _, r in ti[ti["Topic"] != -1].iterrows():
            tid = r["Topic"]
            top_kw = kw[kw["topic_id"] == tid].sort_values("rank").head(10)
            kw_str = " ".join(top_kw["keyword"].tolist())
            all_topics.append({
                "tier": tier,
                "topic_id": int(tid),
                "paper_count": int(r["Count"]),
                "keywords_str": kw_str,
                "top5_kw": ", ".join(top_kw.head(5)["keyword"].tolist()),
            })

    df = pd.DataFrame(all_topics)
    logger.info(f"Loaded {len(df)} topics across {len(TIERS)} tiers")
    return df


def cluster_macro(df, n_clusters=N_MACRO):
    """Cluster topic signatures into macro categories."""
    logger.info("Generating SBERT embeddings for topic signatures...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    signatures = df["keywords_str"].tolist()
    embeddings = model.encode(signatures, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    # Agglomerative clustering with cosine distance
    logger.info(f"Clustering into {n_clusters} macro topics...")
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    )
    df["macro_id"] = clustering.fit_predict(embeddings)
    return df


def label_macro(df):
    """Generate descriptive labels for each macro topic."""
    labels = {}
    for mid in sorted(df["macro_id"].unique()):
        mdf = df[df["macro_id"] == mid]

        # Get most distinctive keywords across all topics in this macro
        all_kw = " ".join(mdf["keywords_str"].tolist())
        word_counts = Counter(all_kw.split())

        # Remove generic words, keep distinctive ones
        stop = {"model", "models", "learning", "data", "using", "based", "method",
                "approach", "new", "via", "training", "paper", "performance",
                "results", "show", "propose", "methods", "task", "tasks", "large"}
        distinctive = [(w, c) for w, c in word_counts.most_common(30) if w not in stop][:8]

        # Count per-tier
        tier_counts = mdf["tier"].value_counts().to_dict()

        # Top topics
        top_topics = mdf.nlargest(3, "paper_count")

        label_words = [w for w, _ in distinctive[:5]]
        labels[mid] = {
            "macro_id": mid,
            "label": " + ".join(label_words[:3]).title(),
            "keywords": ", ".join(label_words),
            "n_topics": len(mdf),
            "total_papers": int(mdf["paper_count"].sum()),
            "tier_distribution": {TIER_LABELS.get(k, k): tier_counts.get(k, 0) for k in TIERS},
            "top_topics": [
                {"tier": TIER_LABELS.get(r["tier"], r["tier"]),
                 "topic_id": int(r["topic_id"]),
                 "papers": int(r["paper_count"]),
                 "keywords": r["top5_kw"]}
                for _, r in top_topics.iterrows()
            ],
        }
    return labels


def main():
    df = load_all_topics()
    df = cluster_macro(df)
    labels = label_macro(df)

    # Save
    df_out = df[["tier", "topic_id", "paper_count", "macro_id", "top5_kw"]].copy()
    df_out["tier_label"] = df_out["tier"].map(TIER_LABELS)
    safe_save_csv(df_out, str(resolve_path("data/results/macro_topic_assignments.csv")))

    # Print summary
    print("\n" + "=" * 70)
    print("MACRO TOPICS — Cross-Era Research Themes")
    print("=" * 70)

    for mid in sorted(labels.keys()):
        l = labels[mid]
        n = l["n_topics"]
        p = l["total_papers"]
        tier_dist = l["tier_distribution"]
        tier_str = " | ".join(f"{tier_dist.get(t, 0):>5}" for t in
                              ["1998-2016", "2016-2020", "2020-2023", "2023-2026"])

        print(f"\n{'─'*60}")
        print(f"M{mid:02d}  {l['label']}")
        print(f"    {n} topics, {p:,} total papers")
        print(f"    Keywords: {l['keywords']}")
        print(f"    Era distribution: {tier_str}")
        print(f"    Representative topics:")
        for tt in l["top_topics"][:3]:
            print(f"      [{tt['tier']}] Topic {tt['topic_id']} ({tt['papers']}p): {tt['keywords']}")

    print(f"\n{'='*60}")
    print("Macro topic assignments saved to data/results/macro_topic_assignments.csv")

    # Save labels as JSON for report use
    with open(resolve_path("data/results/macro_topic_labels.json"), "w") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
