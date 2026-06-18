#!/usr/bin/env python3
"""MC4: Full-corpus BERTopic transform — 547K papers with GPU + multi-core.
Compare sampling-based vs full-corpus topic proportions, entropy, macro-topic sizes.
"""
import sys, os, time, json
from pathlib import Path
import numpy as np, pandas as pd
from collections import Counter
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import resolve_path, setup_logging
logger = setup_logging("full_transform")

# ═══ Load trained T1 model ═══
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

OUT = resolve_path("analysis_output/full_corpus")
OUT.mkdir(parents=True, exist_ok=True)

# Load pre-trained T1 model
logger.info("Loading trained T1 BERTopic model...")
model_path = resolve_path("data/results/bertopic_T1_fine")
if not Path(model_path).exists():
    # Try alternate paths
    for p in Path(resolve_path("data")).rglob("bertopic*T1*"):
        model_path = str(p)
        break

try:
    topic_model = BERTopic.load(model_path)
    logger.info(f"  Model loaded from {model_path}")
except Exception as e:
    logger.error(f"  Failed to load model: {e}")
    # Fallback: train a quick model
    logger.info("  Training new T1 model with saved parameters...")
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic.vectorizers import ClassTfidfTransformer

    umap_model = UMAP(n_neighbors=8, n_components=5, min_dist=0.0, metric="cosine",
                      random_state=42, n_jobs=56)
    hdbscan_model = HDBSCAN(min_cluster_size=30, min_samples=10, core_dist_n_jobs=56,
                           prediction_data=True)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,2), min_df=2)
    ctfidf = ClassTfidfTransformer()

    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                          vectorizer_model=vectorizer, ctfidf_model=ctfidf,
                          nr_topics=None, calculate_probabilities=False, verbose=True)

    # Load sampled T1 data and train
    sampled = pd.read_csv(resolve_path("data/processed/sampled_T1_fine.csv"))
    docs = sampled["text"].tolist()
    logger.info(f"  Training on {len(docs):,} docs...")
    topic_model.fit_transform(docs)
    logger.info("  Model trained. Consider saving for future use.")
    model_path = str(OUT / "bertopic_T1_trained")
    topic_model.save(model_path)
    logger.info(f"  Saved to {model_path}")

# ═══ Load full corpus ═══
logger.info("Loading full corpus (547K)...")
raw = pd.read_csv(resolve_path("data/raw/arxiv_raw_full.csv"), on_bad_lines="skip")
raw["published_dt"] = pd.to_datetime(raw["published"], errors="coerce", utc=True)
raw["text"] = raw["title"].fillna("") + ". " + raw["abstract"].fillna("")
logger.info(f"  Full corpus: {len(raw):,} papers")

# Filter to T1 period for transform
t1_raw = raw[(raw["published_dt"] >= "2023-06-01") & (raw["published_dt"] <= "2026-05-31")]
t1_raw = t1_raw[t1_raw["query_category"].isin(["cs.AI","cs.CL","cs.LG","cs.CV"])]
logger.info(f"  T1 period: {len(t1_raw):,} papers")
docs_full = t1_raw["text"].tolist()

# ═══ GPU-accelerated embeddings ═══
logger.info("Computing embeddings for full T1 corpus (GPU)...")
t0 = time.time()

embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
embeddings_full = embedder.encode(
    docs_full, batch_size=512, show_progress_bar=True,
    convert_to_numpy=True, normalize_embeddings=False
)
elapsed = time.time() - t0
logger.info(f"  {embeddings_full.shape} embeddings in {elapsed/60:.1f} min ({len(embeddings_full)/elapsed:.0f} docs/s)")

# ═══ Transform (approximate_predict for HDBSCAN) ═══
logger.info("Transforming full corpus through BERTopic...")
t0 = time.time()
topics_full, probs_full = topic_model.transform(docs_full, embeddings_full)
elapsed = time.time() - t0
logger.info(f"  Transform done in {elapsed/60:.1f} min")

# ═══ Compare: Sampling-based vs Full-corpus ═══
# Load sampling-based results
t1_info = pd.read_csv(resolve_path("data/results/topic_info_T1_fine.csv"))
t1_assign = pd.read_csv(resolve_path("data/results/topic_assignments_T1_fine.csv"))

# Full-corpus topic distribution
topic_counts_full = Counter(topics_full)
total_full = len(topics_full)
noise_full = topic_counts_full.get(-1, 0)
noise_rate_full = noise_full / total_full

# Sampling-based topic distribution
topic_counts_sampled = Counter(t1_assign["topic_id"])
total_sampled = len(t1_assign)

# Compute entropy for both
def compute_entropy(topic_counts, total):
    """Shannon entropy H = -sum p_i log2 p_i"""
    probs = np.array([c/total for tid, c in topic_counts.items() if tid != -1])
    if len(probs) == 0: return 0, 0
    probs = probs / probs.sum()
    H = -np.sum(probs * np.log2(probs))
    eff_n = 2**H
    return round(H, 4), round(eff_n, 1)

H_sampled, eff_sampled = compute_entropy(topic_counts_sampled, total_sampled)
H_full, eff_full = compute_entropy(topic_counts_full, total_full)

# Compare macro-topic proportions
macro = pd.read_csv(resolve_path("analysis_output/topics/macro_topic_assignments.csv"))
t1_macro = macro[macro["tier"] == "T1_fine"]

# Map topic_id → macro_id
topic_to_macro = {}
for _, r in t1_macro.iterrows():
    topic_to_macro[int(r["topic_id"])] = int(r["macro_id"])

def macro_proportions(topic_counts, total, mapping):
    macro_counts = Counter()
    for tid, count in topic_counts.items():
        if tid in mapping:
            macro_counts[mapping[tid]] += count
    return {mid: count/total for mid, count in macro_counts.items()}

macro_sampled = macro_proportions(topic_counts_sampled, total_sampled, topic_to_macro)
macro_full = macro_proportions(topic_counts_full, total_full, topic_to_macro)

# ═══ Report ═══
results = {
    "corpus": {
        "full_T1_papers": total_full,
        "sampled_T1_papers": total_sampled,
        "noise_rate_full": round(noise_rate_full, 4),
        "noise_rate_sampled": round(len(t1_assign[t1_assign["topic_id"]==-1]) / total_sampled, 4),
    },
    "entropy": {
        "H_sampled": H_sampled, "effective_n_sampled": eff_sampled,
        "H_full": H_full, "effective_n_full": eff_full,
        "delta_H": round(H_full - H_sampled, 4),
        "sampling_bias_pct": round((H_sampled - H_full) / H_full * 100, 1) if H_full > 0 else 0,
    },
    "macro_topic_proportions": {
        "sampled": {str(k): round(v, 4) for k, v in sorted(macro_sampled.items())},
        "full": {str(k): round(v, 4) for k, v in sorted(macro_full.items())},
    }
}

with open(str(OUT / "full_corpus_comparison.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Print summary
print("\n" + "="*70)
print("FULL CORPUS vs SAMPLING-BASED COMPARISON")
print("="*70)
print(f"\nCorpus: {total_full:,} (full) vs {total_sampled:,} (sampled)")
print(f"Noise rate: {noise_rate_full*100:.1f}% (full) vs {results['corpus']['noise_rate_sampled']*100:.1f}% (sampled)")
print(f"\nEntropy (H): {H_full:.4f} (full) vs {H_sampled:.4f} (sampled)")
print(f"Effective topics: {eff_full:.1f} (full) vs {eff_sampled:.1f} (sampled)")
print(f"Sampling bias: +{results['entropy']['sampling_bias_pct']:.1f}% (sampled overestimates entropy by this much)")
print(f"\nMacro-topic proportion comparison (top differences):")
diffs = []
for mid in sorted(macro_full.keys()):
    sf = macro_sampled.get(mid, 0)
    ff = macro_full.get(mid, 0)
    if ff > 0.001:  # only show non-trivial
        diff = sf - ff
        diffs.append((mid, ff, sf, diff))
diffs.sort(key=lambda x: abs(x[3]), reverse=True)
for mid, ff, sf, diff in diffs[:10]:
    arrow = "↑" if diff > 0 else "↓"
    print(f"  M{mid}: {ff*100:5.1f}% (full) vs {sf*100:5.1f}% (sampled)  Δ={diff*100:+5.1f}pp {arrow}")

print(f"\nSaved to {OUT}/full_corpus_comparison.json")
