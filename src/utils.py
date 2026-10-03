"""
Shared helpers used by every pipeline step: config loading, logging,
path resolution, and small I/O conveniences.

Keeping this logic in one place means each step script
(src/01_..py ... src/08_..py) stays focused on its own task instead of
re-implementing "how do I find the config" / "how do I log".
"""
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load the single YAML config that drives the whole pipeline.

    Resolution order: explicit `config_path` arg > SBA_XAI_CONFIG_PATH env
    var (set by `main.py --config ...`) > default config/config.yaml.
    """
    if config_path is None:
        config_path = os.environ.get(
            "SBA_XAI_CONFIG_PATH", PROJECT_ROOT / "config" / "config.yaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_path(cfg: dict, key: str, create: bool = True) -> Path:
    """Resolve a paths.<key> entry from config relative to the project root."""
    rel = cfg["paths"][key]
    p = PROJECT_ROOT / rel
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logger(name: str, cfg: dict, filename: str | None = None) -> logging.Logger:
    """Console + rotating-file logger, consistent across every step script."""
    log_dir = get_path(cfg, "log_dir")
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on re-import / re-run
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_dir / (filename or f"{name}.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def find_recode_file(cfg: dict, dataset_key: str) -> Path:
    """
    Locate a DHS recode file by glob pattern instead of a hardcoded folder
    name — the raw archive's subfolder suffixes are inconsistent
    (ETBR8ADT/ vs ETIR8ADT/ vs ETGE8AFL/), so we search by file stem.
    """
    data_root = get_path(cfg, "data_root", create=False)
    pattern = cfg["datasets"][dataset_key]["glob"]
    matches = sorted(glob.glob(str(data_root / pattern), recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' found under {data_root}. "
            f"Check that the DHS archive for '{dataset_key}' is unzipped "
            f"under {cfg['paths']['data_root']}."
        )
    if len(matches) > 1:
        logging.getLogger(__name__).warning(
            "Multiple files matched %s for '%s'; using the first: %s",
            pattern, dataset_key, matches[0],
        )
    return Path(matches[0])


def save_parquet(df: pd.DataFrame, path: Path, logger: logging.Logger | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    if logger:
        logger.info("Wrote %s rows x %s cols -> %s", df.shape[0], df.shape[1], path)


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Did you run the earlier pipeline step that produces it?"
        )
    return pd.read_parquet(path)


def assert_no_excluded_datasets_used(cfg: dict, loaded_dataset_keys: list[str]) -> None:
    """
    Guardrail matching protocol section 3.1: HR/KR/MR/PR are explicitly out
    of scope. If a future edit accidentally wires one in, fail loudly.
    """
    excluded = set(cfg["datasets"]["excluded"])
    used = set(loaded_dataset_keys)
    leaked = used & excluded
    if leaked:
        raise ValueError(
            f"Datasets {leaked} are explicitly excluded by the study protocol "
            f"(section 3.1) but were loaded. This is a scope violation."
        )
