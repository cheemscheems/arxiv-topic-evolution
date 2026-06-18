#!/usr/bin/env python3
"""MC2: Fast BERTopic UMAP grid + NPMI using pre-existing sampled T1 data."""
import sys, os, time, json
from pathlib import Path
import numpy as np, pandas as pd
from collections import Counter
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import resolve_path

def compute_npmi(docs, topics, top_n=10):
    """NPMI: avg over all topic word-pairs. Clipped to [-1,1]."""
    tokenized = [set(str(d).lower().split()) for d in docs]
    n_docs = len(tokenized)
    df_counts = Counter()
    for tokens in tokenized: df_counts.update(tokens)
    
    # Stopwords
    sw = set("the a an is are was were be been being have has had do does did will would could should may might can shall to of in for on with at by from as into through during before after above below between and or not but if while where when that this it its we they he she which who whom these those using based via et al also however therefore paper propose method approach model models data results show demonstrate performance task tasks learning training new novel state art work".split())
    
    # Top words per topic by TF
    topic_words = {}
    for tid in sorted(set(topics) - {-1}):
        idxs = [i for i,t in enumerate(topics) if t==tid]
        if len(idxs) < 5: continue
        tf = Counter()
        for i in idxs: tf.update(tokenized[i])
        for w in sw: tf.pop(w, None)
        topic_words[tid] = [w for w,_ in tf.most_common(top_n)]
    
    scores = []
    for tid, words in topic_words.items():
        for i, w1 in enumerate(words):
            for w2 in words[i+1:]:
                joint = sum(1 for t in tokenized if w1 in t and w2 in t)
                p_joint = max(joint/n_docs, 1e-12)
                p1 = max(df_counts.get(w1,0)/n_docs, 1e-12)
                p2 = max(df_counts.get(w2,0)/n_docs, 1e-12)
                pmi = np.log(p_joint/(p1*p2))
                npmi = max(-1.0, min(1.0, pmi/(-np.log(p_joint))))
                scores.append(npmi)
    return np.mean(scores) if scores else 0.0, len(topic_words), len(scores)

# ═══ Main ═══
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic.vectorizers import ClassTfidfTransformer
    
    OUT = resolve_path("analysis_output/coherence_grid"); OUT.mkdir(parents=True, exist_ok=True)
    
    # Load ALREADY-SAMPLED T1 data (fast — just 55K rows)
    t1 = pd.read_csv(resolve_path("data/results/topic_assignments_T1_fine.csv"))
    print(f"T1 assignments: {len(t1):,} rows, cols: {list(t1.columns)}")
    
    # Get docs from the original sampled file
    # topic_assignments may not have text — load the sampled CSV directly
    # Find the sampled data file
    sampled_files = list(Path(resolve_path("data")).glob("**/*sampled*T1*"))
    print(f"Sampled files: {[str(f) for f in sampled_files]}")
    
    # Use raw T1 filtered + pre-existing sampling
    raw = pd.read_csv(resolve_path("data/raw/arxiv_raw_full.csv"), on_bad_lines="skip", nrows=300000)
    raw["published_dt"] = pd.to_datetime(raw["published"], errors="coerce", utc=True)
    raw = raw[(raw["published_dt"]>="2023-06-01") & (raw["published_dt"]<="2026-05-31")]
    raw = raw[raw["query_category"].isin(["cs.AI","cs.CL","cs.LG","cs.CV"])]
    raw["text"] = raw["title"].fillna("") + ". " + raw["abstract"].fillna("")
    
    # Simple stratified random sample (FAST — not semantic diversity)
    raw["month"] = raw["published_dt"].dt.strftime("%Y-%m")
    sampled = raw.groupby(["month","query_category"], group_keys=False).apply(
        lambda x: x.sample(n=min(400,len(x)), random_state=42)
    ).reset_index(drop=True)
    docs = sampled["text"].tolist()
    print(f"Sampled: {len(docs):,} docs")
    
    # Cache embeddings ONCE
    print("Computing embeddings (once, cached)...")
    t0 = time.time()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(docs, batch_size=256, show_progress_bar=True)
    print(f"  {embeddings.shape} in {time.time()-t0:.0f}s")
    
    # Fast grid — fewer combos
    grid = [(5,0.0),(8,0.0),(8,0.05),(15,0.1)]  # representative subset
    results = []
    
    for n_neighbors, min_dist in grid:
        label = f"n{n_neighbors}_d{min_dist}"
        print(f"\n─── {label} ───")
        
        umap_model = UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=min_dist,
                         metric="cosine", random_state=42, n_jobs=1)
        hdbscan_model = HDBSCAN(min_cluster_size=30, min_samples=10, core_dist_n_jobs=1)
        vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,2), min_df=2)
        ctfidf = ClassTfidfTransformer()
        
        model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                        vectorizer_model=vectorizer, ctfidf_model=ctfidf,
                        nr_topics=None, calculate_probabilities=False, verbose=False)
        
        t0 = time.time()
        topics, _ = model.fit_transform(docs, embeddings)
        elapsed = time.time() - t0
        
        ti = model.get_topic_info()
        n_topics = len(ti[ti["Topic"]!=-1])
        noise = ti[ti["Topic"]==-1]["Count"].values[0] if -1 in ti["Topic"].values else 0
        noise_rate = noise / len(topics)
        
        npmi, nv, npairs = compute_npmi(docs, topics, top_n=10)
        
        results.append({"label":label,"n_neighbors":n_neighbors,"min_dist":min_dist,
                       "n_topics":n_topics,"noise_rate":round(noise_rate,4),
                       "npmi":round(npmi,4),"n_valid":nv,"n_pairs":npairs,
                       "time_sec":round(elapsed,1)})
        print(f"  Topics={n_topics} Noise={noise_rate:.1%} NPMI={npmi:.4f} ({npairs} pairs) Time={elapsed:.0f}s")
    
    rdf = pd.DataFrame(results)
    rdf.to_csv(str(OUT/"grid_results.csv"), index=False)
    
    print("\n" + "="*65)
    print(f"{'Label':<12} {'Topics':>7} {'Noise%':>8} {'NPMI':>8} {'Time':>6}")
    print("-"*45)
    for _,r in rdf.iterrows():
        print(f"{r['label']:<12} {r['n_topics']:>7} {r['noise_rate']*100:>7.1f}% {r['npmi']:>8.4f} {r['time_sec']:>5.0f}s")
    
    best = rdf.loc[rdf["npmi"].idxmax()]
    print(f"\n→ Best NPMI: {best['label']} (noise={best['noise_rate']*100:.1f}%, NPMI={best['npmi']:.4f})")
    print(f"→ Current (n8,d0.0): noise={rdf[rdf['label']=='n8_d0.0']['noise_rate'].values[0]*100:.1f}%, NPMI={rdf[rdf['label']=='n8_d0.0']['npmi'].values[0]:.4f}")
    print(f"→ Default (n15,d0.1): noise={rdf[rdf['label']=='n15_d0.1']['noise_rate'].values[0]*100:.1f}%, NPMI={rdf[rdf['label']=='n15_d0.1']['npmi'].values[0]:.4f}")
    print(f"\nSaved to {OUT}/grid_results.csv")
