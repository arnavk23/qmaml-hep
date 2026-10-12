"""
Quark-gluon jet dataset loader (Komiske, Metodiev & Thaler, Pythia8 quark/gluon
jets for energy flow, Zenodo record 3164691, https://zenodo.org/records/3164691).
100,000 particle-level jets (50/50 quark/gluon, quark label=1), each a padded
(M, 4) array of (pt, rapidity, phi, PDG id) per constituent particle -- not the
933k-image CMS Open Data set (Andrews et al.) other QMLHEP GSoC projects use,
which is two orders of magnitude larger and not tractable to download/process
here; this is the standard lighter alternative used throughout the energyflow/
"quark-gluon tagging with EFPs" literature.

We reduce each jet's variable-length particle list to a fixed set of high-level
jet-substructure observables, mirroring classification/higgs.py's role (UCI HIGGS's 7
"high-level derived features" analogous to these): jet transverse momentum,
invariant mass, particle multiplicity, girth (pT-weighted mean angular width),
pT dispersion, and leading-particle momentum fraction. These are standard,
well-established quark/gluon-discriminating observables (e.g. quark jets are
narrower and lower-multiplicity than gluon jets), not a novel feature set.

Four-vectors are built from (pt, y, phi) assuming massless constituents
(E = pt*cosh(y), pz = pt*sinh(y), px = pt*cos(phi), py = pt*sin(phi)), the
standard convention for this dataset (particle masses are not provided).
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

QG_URL = "https://zenodo.org/api/records/3164691/files/QG_jets.npz/content"

FEATURE_COLS = ["jet_pt", "jet_mass", "n_particles", "girth", "pt_dispersion", "leading_z", "jet_eta", "jet_phi"]


def _download_if_missing(cache_path: str) -> str:
    if os.path.isfile(cache_path):
        print(f"[quarkgluon_data] Using cached {cache_path}.")
        return cache_path
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"[quarkgluon_data] Downloading {QG_URL} (~107MB, 100k jets)...")
    import urllib.request

    urllib.request.urlretrieve(QG_URL, cache_path)
    print(f"[quarkgluon_data] Done: saved to {cache_path}")
    return cache_path


def _jet_features(particles: np.ndarray) -> np.ndarray:
    """particles: (M, 4) array of (pt, y, phi, pid), zero-padded. Returns an
    8-vector of high-level jet observables for one jet."""
    pt, y, phi = particles[:, 0], particles[:, 1], particles[:, 2]
    mask = pt > 0
    n_particles = float(mask.sum())
    if n_particles == 0:
        return np.zeros(len(FEATURE_COLS), dtype=np.float32)

    pt, y, phi = pt[mask], y[mask], phi[mask]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(y)
    E = pt * np.cosh(y)

    jet_px, jet_py, jet_pz, jet_E = px.sum(), py.sum(), pz.sum(), E.sum()
    jet_pt = float(np.hypot(jet_px, jet_py))
    jet_p = float(np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2))
    jet_mass2 = jet_E**2 - jet_p**2
    jet_mass = float(np.sqrt(max(jet_mass2, 0.0)))
    jet_phi = float(np.arctan2(jet_py, jet_px))
    jet_eta = float(np.arctanh(np.clip(jet_pz / max(jet_p, 1e-9), -1 + 1e-9, 1 - 1e-9)))

    dphi = np.remainder(phi - jet_phi + np.pi, 2 * np.pi) - np.pi
    dR = np.sqrt((y - jet_eta) ** 2 + dphi**2)
    girth = float((pt * dR).sum() / max(pt.sum(), 1e-9))
    pt_dispersion = float(np.sqrt((pt**2).sum()) / max(pt.sum(), 1e-9))
    leading_z = float(pt.max() / max(jet_pt, 1e-9))

    return np.array(
        [jet_pt, jet_mass, n_particles, girth, pt_dispersion, leading_z, jet_eta, jet_phi], dtype=np.float32
    )


def load_quarkgluon(n_jets: int, npz_cache_path: str, features_cache_path: str, seed: int = 42) -> pd.DataFrame:
    """Returns a DataFrame with columns ['label'] + FEATURE_COLS, `label`=1 for
    quark, 0 for gluon (matching the source dataset's y array convention)."""
    if os.path.isfile(features_cache_path):
        df = pd.read_csv(features_cache_path)
        if len(df) >= n_jets:
            print(f"[quarkgluon_data] Using cached features {features_cache_path} ({len(df)} jets).")
            return df.iloc[:n_jets].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    _download_if_missing(npz_cache_path)
    print(f"[quarkgluon_data] Extracting jet-substructure features for {n_jets} jets...")
    with np.load(npz_cache_path) as data:
        X, y = data["X"][:n_jets], data["y"][:n_jets]

    rows = [_jet_features(X[i]) for i in range(len(X))]
    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    df.insert(0, "label", y.astype(np.int64))
    os.makedirs(os.path.dirname(features_cache_path), exist_ok=True)
    df.to_csv(features_cache_path, index=False)
    print(f"[quarkgluon_data] Cached features -> {features_cache_path}")
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_test_split_df(df: pd.DataFrame, test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_test = int(len(df) * test_frac)
    return df.iloc[n_test:].reset_index(drop=True), df.iloc[:n_test].reset_index(drop=True)


__all__ = ["load_quarkgluon", "train_test_split_df", "FEATURE_COLS"]
