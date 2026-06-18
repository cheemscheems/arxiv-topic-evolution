"""Data cleaning, deduplication, text preparation, and stratified sampling."""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, ensure_dirs, setup_logging, clean_text, safe_save_csv, resolve_path

logger = setup_logging("preprocess")


def parse_date_col(df: pd.DataFrame, col: str = "published") -> pd.DataFrame:
    """Parse published date column and extract year-month."""
    df["published_dt"] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df["month"] = df["published_dt"].dt.strftime("%Y-%m")
    return df


def filter_target_categories(df: pd.DataFrame, target_cats: list) -> pd.DataFrame:
    """Keep only papers whose primary_category is in the target list."""
    before = len(df)
    df = df[df["primary_category"].isin(target_cats)].copy()
    logger.info(f"Primary category filter: {before} -> {len(df)} (target: {target_cats})")
    return df


def filter_withdrawn(df: pd.DataFrame) -> pd.DataFrame:
    """Remove withdrawn papers."""
    before = len(df)
    text_cols = ["title", "abstract"]
    mask = pd.Series(False, index=df.index)
    for col in text_cols:
        if col in df.columns:
            mask |= df[col].fillna("").str.lower().str.contains("withdrawn", na=False)
    df = df[~mask].copy()
    logger.info(f"Withdrawn filter: {before} -> {len(df)} (removed {before - len(df)})")
    return df


def build_text(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate title and abstract into the 'text' column."""
    df["title_clean"] = df["title"].apply(clean_text)
    df["abstract_clean"] = df["abstract"].apply(clean_text)
    df["text"] = df["title_clean"] + ". " + df["abstract_clean"]
    return df


def filter_short_text(df: pd.DataFrame, min_chars: int = 300) -> pd.DataFrame:
    """Remove papers whose combined text is too short."""
    before = len(df)
    df = df[df["text"].str.len() >= min_chars].copy()
    logger.info(f"Min text length ({min_chars}) filter: {before} -> {len(df)} (removed {before - len(df)})")
    return df


def filter_near_duplicate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Remove near-duplicate papers (same paper, different versions/IDs).

    Normalizes titles (lowercase, strip, first 100 chars) and keeps the
    LATEST version (max published date) for each near-duplicate group.
    This handles cases where the same paper is re-submitted with a new arxiv_id.
    """
    before = len(df)
    df = df.copy()
    df["_title_key"] = (
        df["title"]
        .fillna("")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str[:100]
    )
    # Sort by published descending, then drop duplicates keeping the first (latest)
    df = df.sort_values("published_dt", ascending=False)
    df = df.drop_duplicates(subset="_title_key", keep="first")
    df = df.drop(columns=["_title_key"])
    removed = before - len(df)
    logger.info(f"Near-duplicate title filter: {before} -> {len(df)} (removed {removed})")
    return df


def stratified_sample(df: pd.DataFrame, sample_per_group: int, random_seed: int) -> pd.DataFrame:
    """Stratified sampling with semantic diversity within each group.

    Within each month×category group, uses TF-IDF + MiniBatchKMeans to cluster
    papers into micro-clusters, then samples proportionally from each cluster.
    This maximizes topic coverage and prevents random sampling from missing
    small but distinct sub-topics. Falls back to simple random for small groups.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import MiniBatchKMeans

    groups = df.groupby(["month", "primary_category"], group_keys=False)
    sampled_parts = []
    rng = np.random.RandomState(random_seed)

    for (month, cat), group in groups:
        pool_size = len(group)
        target_n = min(pool_size, sample_per_group)

        if target_n >= pool_size:
            sampled_parts.append(group)
            continue

        # Only use diversity sampling when pool is large enough
        if pool_size < target_n * 2 or pool_size < 100:
            sampled_parts.append(group.sample(n=target_n, random_state=random_seed))
            continue

        n_clusters = min(20, target_n // 15)  # ~15 papers per micro-cluster
        texts = group["text"].fillna("").tolist()

        try:
            vectorizer = TfidfVectorizer(
                max_features=2000, stop_words="english", ngram_range=(1, 2)
            )
            X = vectorizer.fit_transform(texts)

            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, random_state=random_seed,
                n_init=3, batch_size=min(256, pool_size),
            )
            labels = kmeans.fit_predict(X)

            cluster_samples = []
            for c in range(n_clusters):
                cluster = group[labels == c]
                n_c = max(1, int(target_n * len(cluster) / pool_size))
                n_c = min(n_c, len(cluster))
                cluster_samples.append(
                    cluster.sample(n=n_c, random_state=random_seed)
                )

            result = pd.concat(cluster_samples, ignore_index=True)
            if len(result) > target_n:
                result = result.sample(n=target_n, random_state=random_seed)
            sampled_parts.append(result)

        except Exception:
            sampled_parts.append(group.sample(n=target_n, random_state=random_seed))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    logger.info(
        f"Semantic diversity sampling: {len(groups)} groups → {len(sampled)} papers "
        f"(simple random for small groups, TF-IDF+KMeans for others)"
    )
    return sampled


def compute_sampling_stats(df_clean: pd.DataFrame, df_sampled: pd.DataFrame) -> pd.DataFrame:
    """Compute per-group sampling statistics."""
    raw_counts = df_clean.groupby(["month", "primary_category"]).size().reset_index(name="raw_count")
    sampled_counts = df_sampled.groupby(["month", "primary_category"]).size().reset_index(name="sampled_count")

    stats = raw_counts.merge(sampled_counts, on=["month", "primary_category"], how="left")
    stats["sampled_count"] = stats["sampled_count"].fillna(0).astype(int)
    stats["sampling_rate"] = (stats["sampled_count"] / stats["raw_count"]).round(4)
    return stats


def compute_dataset_stats(df_raw: pd.DataFrame, df_clean: pd.DataFrame, df_sampled: pd.DataFrame,
                          config: dict) -> pd.DataFrame:
    """Compute overall dataset statistics."""
    stats = {
        "total_raw_records": len(df_raw),
        "total_unique_papers": df_clean["arxiv_id"].nunique() if "arxiv_id" in df_clean.columns else len(df_clean),
        "total_clean_papers": len(df_clean),
        "total_sampled_papers": len(df_sampled),
        "start_date": config["start_date"],
        "end_date": config["end_date"],
    }
    return pd.DataFrame([stats])


def main():
    config = load_config()
    ensure_dirs()

    raw_path = resolve_path("data/raw/arxiv_raw.csv")
    if not raw_path.exists():
        logger.error(f"Raw data not found: {raw_path}")
        logger.info("Run 01_fetch_arxiv.py first.")
        return

    # 1. Load raw data
    df_raw = pd.read_csv(raw_path)
    logger.info(f"Loaded raw data: {len(df_raw)} records")

    # 2. Parse dates
    df = parse_date_col(df_raw)
    logger.info(f"Date range parsed: {df['month'].min()} to {df['month'].max()}")

    # 3. Deduplicate by arxiv_id
    before = len(df)
    df = df.drop_duplicates(subset="arxiv_id", keep="first").copy()
    logger.info(f"Deduplication (arxiv_id): {before} -> {len(df)}")

    # 4. Filter target categories
    df = filter_target_categories(df, config["target_categories"])

    # 5. Filter withdrawn
    df = filter_withdrawn(df)

    # 6. Build text
    df = build_text(df)

    # 7. Near-duplicate title dedup (keep latest version)
    df_clean = filter_near_duplicate_titles(df)

    # 8. Save clean data
    clean_cols = ["arxiv_id", "title", "abstract", "text", "published", "month",
                  "primary_category", "categories", "authors", "entry_url"]
    clean_cols = [c for c in clean_cols if c in df_clean.columns]
    safe_save_csv(df_clean[clean_cols], str(resolve_path("data/processed/arxiv_clean.csv")))

    # 9. Stratified sampling
    sample_per_group = config["sample_per_group"]
    random_seed = config["random_seed"]
    df_sampled = stratified_sample(df_clean, sample_per_group, random_seed)
    logger.info(f"Sampled: {len(df_sampled)} papers ({sample_per_group} per group max)")

    save_cols = ["arxiv_id", "title", "abstract", "text", "published", "month",
                 "primary_category", "categories", "authors", "entry_url"]
    save_cols = [c for c in save_cols if c in df_sampled.columns]
    safe_save_csv(df_sampled[save_cols], str(resolve_path("data/processed/arxiv_sampled.csv")))

    # 10. Compute and save stats
    sampling_stats = compute_sampling_stats(df_clean, df_sampled)
    safe_save_csv(sampling_stats, str(resolve_path("data/results/group_sampling_stats.csv")))

    dataset_stats = compute_dataset_stats(df_raw, df_clean, df_sampled, config)
    safe_save_csv(dataset_stats, str(resolve_path("data/results/dataset_stats.csv")))

    logger.info("=== Preprocessing Complete ===")
    logger.info(f"  Raw records:     {len(df_raw)}")
    logger.info(f"  Clean papers:    {len(df_clean)}")
    logger.info(f"  Sampled papers:  {len(df_sampled)}")
    logger.info(f"  Groups:          {len(sampling_stats)}")

    # Print per-category summary
    cat_summary = df_sampled.groupby("primary_category").size()
    for cat, cnt in cat_summary.items():
        logger.info(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()
