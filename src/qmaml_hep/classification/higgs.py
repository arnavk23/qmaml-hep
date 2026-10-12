"""
UCI HIGGS dataset loader (https://archive.ics.uci.edu/ml/datasets/HIGGS), streamed
and truncated to a manageable subsample -- avoids downloading the full 11M-row /
2.8GB-compressed file. The gzip stream is read sequentially and closed early once
enough rows are collected, so network traffic is roughly proportional to the
fraction of rows requested, not the full file size.

Columns (per UCI documentation): label, then 21 low-level kinematic features,
then 7 high-level derived features (m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb).
"""

import gzip
import os
from typing import Tuple

import numpy as np
import pandas as pd

HIGGS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

LOW_LEVEL_COLS = [
    "lepton_pT",
    "lepton_eta",
    "lepton_phi",
    "missing_energy_magnitude",
    "missing_energy_phi",
    "jet1_pt",
    "jet1_eta",
    "jet1_phi",
    "jet1_b-tag",
    "jet2_pt",
    "jet2_eta",
    "jet2_phi",
    "jet2_b-tag",
    "jet3_pt",
    "jet3_eta",
    "jet3_phi",
    "jet3_b-tag",
    "jet4_pt",
    "jet4_eta",
    "jet4_phi",
    "jet4_b-tag",
]
HIGH_LEVEL_COLS = ["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]
ALL_FEATURE_COLS = LOW_LEVEL_COLS + HIGH_LEVEL_COLS
COLUMNS = ["label"] + ALL_FEATURE_COLS


def stream_subsample(n_rows: int, cache_path: str, chunksize: int = 20_000) -> str:
    """Stream the first `n_rows` of HIGGS.csv.gz and cache them as a small local CSV.
    Returns cache_path. If cache_path already has >= n_rows, skips the network entirely."""
    if os.path.isfile(cache_path):
        cached_len = sum(1 for _ in open(cache_path, encoding="utf-8")) - 1  # minus header
        if cached_len >= n_rows:
            print(f"[higgs_data] Using cached {cache_path} ({cached_len} rows).")
            return cache_path

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(
        f"[higgs_data] Streaming {n_rows} rows from {HIGGS_URL} (stops early, does not download the full 2.8GB file)..."
    )

    import urllib.request

    req = urllib.request.Request(HIGGS_URL, headers={"User-Agent": "Mozilla/5.0"})
    rows_written = 0
    with urllib.request.urlopen(req) as resp, gzip.GzipFile(fileobj=resp) as gz:
        with open(cache_path, "w", encoding="utf-8", newline="") as out:
            out.write(",".join(COLUMNS) + "\n")
            for line in gz:
                out.write(line.decode("utf-8"))
                rows_written += 1
                if rows_written >= n_rows:
                    break
                if rows_written % chunksize == 0:
                    print(f"[higgs_data]   {rows_written}/{n_rows} rows...")
    print(f"[higgs_data] Done: {rows_written} rows written to {cache_path}")
    return cache_path


def load_higgs(n_rows: int, cache_path: str, seed: int = 42) -> pd.DataFrame:
    stream_subsample(n_rows, cache_path)
    df = pd.read_csv(cache_path, nrows=n_rows)
    df["label"] = df["label"].astype(np.int64)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_test_split_df(df: pd.DataFrame, test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_test = int(len(df) * test_frac)
    return df.iloc[n_test:].reset_index(drop=True), df.iloc[:n_test].reset_index(drop=True)


__all__ = ["load_higgs", "train_test_split_df", "LOW_LEVEL_COLS", "HIGH_LEVEL_COLS", "ALL_FEATURE_COLS"]
