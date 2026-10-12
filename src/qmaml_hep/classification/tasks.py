"""
Few-shot task generation for tabular HEP data (HIGGS, quark-gluon jets) by quantile binning.

Each task = one quantile bin of a physics-motivated high-level feature
(m_bb: reconstructed Higgs->bb candidate mass). Support/query sets are drawn
class-balanced from within that bin. This gives many distinct-but-related
binary classification "tasks" for the Learner to meta-train across, the
same way each Hamiltonian is a distinct "task" in the Q-MAML paper.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


def make_tasks(
    df: pd.DataFrame,
    feature_cols: List[str],
    bin_col: str,
    bin_count: int,
    support_size: int,
    query_size: int,
    tasks_per_bin: int,
    seed: int = 42,
    min_per_class: int = None,
) -> List[Dict[str, Any]]:
    assert support_size % 2 == 0 and query_size % 2 == 0
    half_s, half_q = support_size // 2, query_size // 2
    need_per_class = half_s + half_q
    min_per_class = min_per_class or need_per_class

    rng = np.random.default_rng(seed)
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    bin_vals = df[bin_col].values.astype(np.float32)

    edges = np.quantile(bin_vals, np.linspace(0, 1, bin_count + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_idx = np.digitize(bin_vals, edges[1:-1], right=False)

    tasks = []
    for b in range(bin_count):
        mask = bin_idx == b
        idx_c0 = np.where(mask & (y == 0))[0]
        idx_c1 = np.where(mask & (y == 1))[0]
        if len(idx_c0) < min_per_class or len(idx_c1) < min_per_class:
            continue
        for _ in range(tasks_per_bin):
            pick0 = rng.choice(idx_c0, size=need_per_class, replace=False)
            pick1 = rng.choice(idx_c1, size=need_per_class, replace=False)
            s_idx = np.concatenate([pick0[:half_s], pick1[:half_s]])
            q_idx = np.concatenate([pick0[half_s:], pick1[half_s:]])
            rng.shuffle(s_idx)
            rng.shuffle(q_idx)
            tasks.append(
                {
                    "support_X": torch.from_numpy(X[s_idx]),
                    "support_y": torch.from_numpy(y[s_idx]),
                    "query_X": torch.from_numpy(X[q_idx]),
                    "query_y": torch.from_numpy(y[q_idx]),
                    "bin": b,
                }
            )
    return tasks


__all__ = ["make_tasks"]
