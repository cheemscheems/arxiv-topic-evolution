"""Common utilities: config loading, directory creation, logging, text cleaning, CSV I/O."""

import os
import yaml
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_tiers(config: dict = None) -> dict:
    """Extract tiers configuration, falling back to defaults."""
    if config is None:
        config = load_config()
    return config.get("tiers", {})

def get_categories(config: dict = None) -> list:
    """Extract target categories."""
    if config is None:
        config = load_config()
    return config.get("categories", ["cs.AI","cs.CL","cs.LG","cs.CV"])

def get_eras(config: dict = None) -> dict:
    """Extract era definitions."""
    if config is None:
        config = load_config()
    return config.get("eras", {})

def get_paradigms(config: dict = None) -> dict:
    """Extract paradigm keyword definitions."""
    if config is None:
        config = load_config()
    return config.get("paradigms", {})

def get_burst_terms(config: dict = None) -> list:
    """Extract burst detection target terms."""
    if config is None:
        config = load_config()
    return config.get("burst", {}).get("target_terms", [])

def get_jaccard_config(config: dict = None) -> dict:
    """Extract Jaccard analysis configuration."""
    if config is None:
        config = load_config()
    return config.get("jaccard", {})

def get_pelt_config(config: dict = None) -> dict:
    """Extract PELT changepoint configuration."""
    if config is None:
        config = load_config()
    return config.get("pelt", {})

def get_parallel_workers(config: dict = None) -> int:
    """Extract parallel worker count."""
    if config is None:
        config = load_config()
    return config.get("parallel_workers", 1)

def load_config(path: str = None) -> dict:
    """Load YAML configuration file."""
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    """Create all required output directories."""
    dirs = [
        "data/raw",
        "data/processed",
        "data/results",
        "figures",
        "report",
        "report/assets",
        "logs",
    ]
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "arxiv_topic") -> logging.Logger:
    """Configure and return a logger that writes to both console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def clean_text(text: str) -> str:
    """Clean text: normalize whitespace, strip, keep professional terms intact."""
    if not isinstance(text, str):
        return ""
    # Replace newlines and tabs with space
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Collapse multiple whitespace
    import re
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """Save DataFrame to CSV, creating parent directories as needed."""
    full_path = Path(path)
    if not full_path.is_absolute():
        full_path = PROJECT_ROOT / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=index, encoding="utf-8-sig")
    return full_path


def resolve_path(rel_path: str) -> Path:
    """Resolve a relative path against PROJECT_ROOT."""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def month_range(start: str, end: str) -> list:
    """Generate list of 'YYYY-MM' strings between start and end dates (inclusive)."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    months = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        months.append(current.strftime("%Y-%m"))
        # Advance one month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def timestamp_str() -> str:
    """Return current timestamp string for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
