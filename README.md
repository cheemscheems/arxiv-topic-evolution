# ArXiv AI Topic Evolution (1998–2026)

Code repository for the paper **"从分化到耦合：arXiv人工智能、机器学习、自然语言处理与计算机视觉主题演化的28年"**  
(*From Fragmentation to Coupling: 28 Years of Topic Evolution in arXiv AI, Machine Learning, NLP, and Vision*)

## Overview

This repository contains the complete computational pipeline for analyzing topic evolution across **547,130 arXiv preprints** from four CS categories (cs.AI, cs.CL, cs.LG, cs.CV) spanning **June 1998 – May 2026**.

**Core methods**: Hierarchical BERTopic modeling (4 temporal tiers, 929 sub-topics aggregated into 16 macro meta-topics), PELT + CUSUM change-point detection, threshold-based burst detection, three-variant Jaccard similarity analysis, Shannon entropy decomposition, detrended CV volatility analysis, and Granger-predictive lead-lag analysis.

## Repository Structure

```
.
├── README.md
├── Makefile                  # Pipeline orchestration (8 targets)
├── config.yaml               # All configurable parameters
├── requirements.txt          # Python dependencies
├── CITATION.cff
├── .gitignore
├── src/
│   ├── utils.py              # Shared utilities + config helpers
│   ├── 02_prepare_data.py    # Data cleaning + stratified sampling
│   ├── 03_train_bertopic.py  # 4-tier BERTopic + change-point detection
│   ├── 04_macro_topics.py    # 929 sub-topics → 16 macro meta-topics
│   ├── 05_comprehensive_analysis.py  # Growth, volatility, seasonality, anomalies
│   ├── 06_advanced_analytics.py      # Paradigm half-life, entropy, Jaccard, bursts
│   ├── 07_final_analytics.py         # A1–A8 final analyses (Paper §4–§6)
│   └── experiments/
│       ├── topic_coherence_grid.py   # NPMI grid search (§3.3, Table 3)
│       └── full_corpus_transform.py  # Full-corpus sampling bias validation (§3.2, Table 4)
├── data/
│   ├── README.md             # Data access instructions
│   ├── raw/                  # Raw arXiv API output (not committed)
│   ├── processed/            # Cleaned + sampled data (not committed)
│   └── results/              # Topic model outputs
│       ├── topic_info_T1~T4.csv       # Per-tier topic statistics
│       ├── topic_keywords_T1~T4.csv   # Per-topic top-10 keywords
│       ├── macro_topic_assignments.csv # 929 sub-topics→16 macro
│       ├── change_points.csv          # 33 structural change-points
│       ├── cross_tier_trends.csv      # Cross-tier topic trends
│       └── topic_evolution.csv        # Topic evolution data
└── outputs/
    └── sample/               # Representative outputs for verification
        └── analysis_output/  # 15 CSV/JSON files matching paper tables
```

## Requirements

- **Python 3.10+** with packages listed in `requirements.txt`
- **Network access** required only for `make data` (arXiv API)
- **GPU optional** — SentenceTransformer auto-detects CUDA; full-corpus transform takes ~10 min on V100 vs. several hours on CPU
- **Tested on Debian 13 (amd64)** — compatibility with other environments (WSL, macOS, etc.) is not guaranteed and should be verified independently

## Installation

```bash
git clone https://github.com/cheemscheems/arxiv-topic-evolution.git
cd arxiv-topic-mining
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
make smoke    # Verify config loading + all scripts compile (~10 sec)
```

If you have the raw data already available (`data/raw/arxiv_raw_full.csv`):

```bash
make prepare     # Step 2: Clean + sample (~30 sec)
make topics      # Step 3: 4-tier BERTopic (~10 min on 56 cores)
```

## Complete Pipeline

All parameters are configured in `config.yaml`. Each step reads its inputs from the previous step's outputs.

| Step | Command | Description | Input | Output | Time (approx.) |
|:----:|---------|-------------|-------|--------|:---:|
| 1 | `make data` | Fetch from arXiv API | — | `data/raw/arxiv_raw_full.csv` | Several hours |
| 2 | `make prepare` | Clean + stratified semantic-diversity sampling | Raw CSV | `data/processed/sampled_T*.csv` | 30 sec |
| 3 | `make topics` | 4-tier BERTopic + change-point detection | Sampled CSVs | `data/results/topic_info_T*.csv` | 10 min |
| 4 | `make macro` | 929 sub-topics → 16 macro meta-topics | Topic keywords | `analysis_output/topics/macro_topic_assignments.csv` | 1 min |
| 5 | `make analysis` | Growth, volatility, seasonality, anomalies, correlations | Raw CSV | `analysis_output/growth/`, `seasonality/`, etc. | 5 min |
| 6 | `make advanced` | Paradigm half-life, entropy, Jaccard, burst detection | Raw CSV | `analysis_output/advanced/` | 10 min |
| 7 | `make final` | A1–A8 final analyses (Paper §4–§6 core results) | Multiple inputs | `analysis_output/final/` | 5 min |
| — | `make all` | Run steps 1–7 sequentially | — | All outputs | Variable |

## Data Access

### What is included

The following **analysis outputs** are included in this repository (see `data/results/` and `outputs/sample/`):

| File | Size | Description |
|------|------|-------------|
| `topic_info_T1~T4.csv` | 4.3 MB total | Per-tier topic statistics (topic ID, paper count, noise rate) |
| `topic_keywords_T1~T4.csv` | 229 KB total | Top-10 c-TF-IDF keywords for each topic |
| `macro_topic_assignments.csv` | 79 KB | 929 sub-topics mapped to 16 macro meta-topics |
| `change_points.csv` | 689 B | 33 structural change-points (PELT + CUSUM) |
| `cross_tier_trends.csv` | 1.8 KB | Cross-tier topic trend data |
| `topic_evolution.csv` | 9 KB | Topic evolution across tiers |
| `outputs/sample/` | ~1 MB | 15 representative analysis output files |

These files are our **original analytical products** (aggregate statistics, clustering results, keyword extractions). They contain no raw paper text and are freely redistributable.

### What is NOT included (and why)

Three categories of data are intentionally excluded:

**1. Raw arXiv metadata (`data/raw/arxiv_raw_full.csv`, 779 MB)**

The [arXiv API Terms of Use](https://info.arxiv.org/help/api/tou.html) prohibit bulk redistribution of metadata. With 547,130 records containing titles, abstracts, and author lists, this file constitutes a "substantial portion" under the API terms. Each user must obtain it independently.

→ **How to obtain**: Run `make data`, which queries the public arXiv API with exponential-backoff retry logic (~864 queries, several hours).

**2. Sampled intermediate data (`data/processed/sampled_T*.csv`, 42–158 MB × 4)**

These files are derived from the raw arXiv metadata through text cleaning and stratified semantic-diversity sampling. Since the upstream raw data cannot be redistributed, these derivative files are also excluded.

→ **How to regenerate**: Run `make prepare`. Produces identical output from the same raw data and random seed (42).

**3. Per-paper topic assignments (`data/results/topic_assignments_T*.csv`, 42–158 MB × 4)**

These files store every paper's topic ID alongside its full title and abstract text. They are both too large for practical Git storage and largely redundant — the essential information they contain is already condensed in `topic_info_*.csv` (topic-level aggregate statistics) and `topic_keywords_*.csv` (c-TF-IDF keyword representations), both of which are included.

→ **How to regenerate**: Run `make topics`. Produces identical output with the same random seed.

**4. Quick start without raw data**:
   - Inspect pre-computed outputs in `outputs/sample/`
   - Run `make smoke` to verify code integrity
   - Examine `config.yaml` to understand parameter choices
   - Contact the author for BERTopic model checkpoints (~500 MB/tier)

## Configuration

All key parameters are in `config.yaml`. To adapt for different data:

| To change | Edit config key |
|-----------|----------------|
| Date range | `start_date`, `end_date` |
| Categories | `categories` (list of arXiv category IDs) |
| Tier boundaries | `tiers.T*_*.start`, `tiers.T*_*.end` |
| Sampling intensity | `tiers.T*_*.sample_per_group` |
| BERTopic UMAP params | `bertopic.umap.*` |
| Jaccard umbrella terms | `jaccard.umbrella_terms` |
| Burst target keywords | `burst.target_terms` |
| Paradigm keyword lists | `paradigms.*.terms` |
| CPU parallelism | `parallel_workers` |

## Sample Outputs

The `outputs/sample/` directory contains 15 representative CSV/JSON files matching the paper's tables. These allow verification of the analysis pipeline without running the full computation.

| File | Paper Reference |
|------|----------------|
| `growth/monthly_paper_counts.csv` | Fig 1, §4.1 |
| `growth/volatility_by_era.csv` | Table 5, §6.2 |
| `anomalies/structural_breaks.csv` | §4.2 |
| `seasonality/cvpr_march_effect.csv` | Fig 7, §6.4 |
| `advanced/discipline_convergence.csv` | Fig 4, §5.5 |
| `advanced/keyword_bursts.csv` | §4.3 |
| `advanced/paradigm_halflife.csv` | Table 2, §4.4 |
| `final/A1_generic_term_jaccard.csv` | Table 3 (Appendix) |
| `final/A1_fixed_vocab_jaccard.csv` | Table 3 (Appendix) |
| `final/A2_entropy_decomposition.csv` | Fig 5, §6.1 |
| `final/A3_primary_category_jaccard.csv` | Table 3, §5.5 |
| `final/A6_granger_causality.csv` | Table 6, §5.6 |
| `coherence_grid/grid_results.csv` | Table 3, §3.3 |
| `full_corpus/all_tiers_comparison.csv` | Table 4, §3.2 |
| `full_corpus/emerging_topics.csv` | Table A5 (Appendix) |

## Robustness Experiments

```bash
make coherence-grid           # UMAP parameter grid search + NPMI (~10 min, §3.3)
make full-corpus-transform    # Full-corpus sampling bias validation (~5 min GPU, §3.2)
```

## Reproducibility Notes

- **Random seed**: 42 — The Answer to the Ultimate Question of Life, The Universe, and Everything (also: all numpy, sklearn, UMAP, HDBSCAN, MiniBatchKMeans)
- **Embedding model**: `all-MiniLM-L6-v2` (exact version pinned in `requirements.txt`)
- **Expected row counts**: raw ~547,130; post-sampling 4-tier total ~148,675
- **arXiv API**: Subject to rate limiting (HTTP 429). Scripts use exponential backoff with checkpoint resume
- **BERTopic model checkpoints**: Available upon request (~500 MB per tier)
- **GPU acceleration**: SentenceTransformer auto-detects CUDA; UMAP/HDBSCAN use CPU parallelism (`parallel_workers` in config)

## Troubleshooting

**"No such file: data/raw/arxiv_raw_full.csv"**  
→ Run `make data` first, or place your data file at that path.

**"Repo id must be in the form 'namespace/repo_name'"**  
→ BERTopic model checkpoint not found. The script will auto-train a new model. Ignore this warning.

**Out of memory during full-corpus transform**  
→ Reduce `sample_per_group` in config, or use `make topics` without `make full-corpus-transform`.

**arXiv API rate limited (HTTP 429)**  
→ Scripts auto-retry with exponential backoff. Increase `arxiv.sleep_seconds` in config if persistent.

## Citation

```bibtex
@software{arxiv-topic-evolution,
  title   = {ArXiv AI Topic Evolution (1998--2026)},
  url     = {https://github.com/cheemscheems/arxiv-topic-evolution},
  year    = {2026},
}
```

## License

MIT License. Data from arXiv is subject to [arXiv's terms of use](https://info.arxiv.org/help/api/tou.html).
