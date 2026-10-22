"""LHC Olympics 2020 (m_X, m_Y) signal grid against QCD: a classification task family with a physical descriptor.

Signal: W'(3.5 TeV) -> X(qq) Y(qq), (m_X, m_Y) on a 12 x 12 grid from 50 to 600 GeV, about 100k events per point
(Zenodo 11188685, the PAWS extended signal). Background: 22.5M Pythia QCD dijet events generated with the same cards
(Zenodo 22685072). Both files store high-level features of the two leading R = 1 jets.

Each task is "signal at (m_X, m_Y) vs QCD" in the signal region 3.3 < m_jj < 3.7 TeV, on six features: the lighter
jet mass, the mass difference, and tau21 and tau32 of the lighter and heavier jet. The n-subjettiness normalisation
differs between the files; the ratios cancel it (checked: KS <= 0.031 against the original R&D signal at
(500, 100), KS <= 0.005 against the original 1M-event QCD in the signal region).

``load()`` returns the per-point signal arrays and the QCD array, built once from the raw files and cached as
``data/lhco/lhco_sr_features.npz`` (about 200 MB).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request

import numpy as np
import pandas as pd

from qmaml_hep.paths import DATA

SR = (3.3e3, 3.7e3)
FEATS = ["mj_lo", "dm", "tau21_lo", "tau21_hi", "tau32_lo", "tau32_hi"]
RAW = [
    "pxj1",
    "pyj1",
    "pzj1",
    "mj1",
    "tau1j1",
    "tau2j1",
    "tau3j1",
    "pxj2",
    "pyj2",
    "pzj2",
    "mj2",
    "tau1j2",
    "tau2j2",
    "tau3j2",
]
FILES = {
    "events_anomalydetection_Z_XY_qq_parametric.h5": (
        "https://zenodo.org/api/records/11188685/files/events_anomalydetection_Z_XY_qq_parametric.h5/content",
        "ef7b613d8abd9c0a5958bf526df98f30",
    ),
    "Pythia_QCD_Dijet_Events.h5": (
        "https://zenodo.org/api/records/22685072/files/Pythia_QCD_Dijet_Events.h5/content",
        "b8601daa798e276e96c3af101a706089",
    ),
}
STRAY = {(50, 750)}  # an off-grid point with 2.4k events in the signal file


def _md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch(name, root):
    """Download one raw file into ``root`` if missing and verify its md5. Single stream; for speed, fetch the
    URLs in ``FILES`` with a parallel downloader (e.g. ``aria2c -x 16``) into ``data/lhco`` first."""
    path = os.path.join(root, name)
    url, md5 = FILES[name]
    if not os.path.isfile(path):
        os.makedirs(root, exist_ok=True)
        print(f"[lhco] downloading {name} (several GB) ...")
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    if _md5(path) != md5:
        raise RuntimeError(f"{path}: md5 mismatch, delete it and download again")
    return path


def features(d: pd.DataFrame) -> pd.DataFrame:
    """Raw two-jet columns -> m_jj and the six task features (massless-jet four-vectors from 3-momentum + mass)."""
    e = [np.sqrt(d[f"pxj{j}"] ** 2 + d[f"pyj{j}"] ** 2 + d[f"pzj{j}"] ** 2 + d[f"mj{j}"] ** 2) for j in "12"]
    px, py, pz = (d[f"p{c}j1"] + d[f"p{c}j2"] for c in "xyz")
    m1, m2 = d["mj1"].to_numpy(), d["mj2"].to_numpy()
    first_is_lo = m1 <= m2
    eps = 1e-9
    t21 = [(d[f"tau2j{j}"] / (d[f"tau1j{j}"] + eps)).to_numpy() for j in "12"]
    t32 = [(d[f"tau3j{j}"] / (d[f"tau2j{j}"] + eps)).to_numpy() for j in "12"]
    out = pd.DataFrame(
        {
            "mjj": np.sqrt(np.clip((e[0] + e[1]) ** 2 - px**2 - py**2 - pz**2, 0, None)).to_numpy(),
            "mj_lo": np.minimum(m1, m2),
            "dm": np.abs(m1 - m2),
            "tau21_lo": np.where(first_is_lo, t21[0], t21[1]),
            "tau21_hi": np.where(first_is_lo, t21[1], t21[0]),
            "tau32_lo": np.where(first_is_lo, t32[0], t32[1]),
            "tau32_hi": np.where(first_is_lo, t32[1], t32[0]),
        },
        index=d.index,
    )
    for c in ("mx", "my"):
        if c in d:
            out[c] = d[c].to_numpy()
    return out


def in_sr(f: pd.DataFrame) -> pd.DataFrame:
    return f[(f.mjj > SR[0]) & (f.mjj < SR[1])]


def _qcd_sr(path, chunk=1_000_000):
    """The QCD file is a fixed-format pandas frame of 22.5M rows; read it in row slices with PyTables."""
    import tables

    out = []
    with tables.open_file(path, "r") as h:
        node = h.root.particles
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in node.block0_items.read()]
        lab_names = [x.decode() if isinstance(x, bytes) else str(x) for x in node.block2_items.read()]
        assert lab_names == ["label"], lab_names
        n = node.block0_values.shape[0]
        for s in range(0, n, chunk):
            b0 = node.block0_values[s : s + chunk]
            lab = node.block2_values[s : s + chunk][:, 0]
            df = pd.DataFrame({nm: b0[:, j].astype(np.float64) for j, nm in enumerate(names) if nm in RAW})
            out.append(in_sr(features(df[lab == 0]))[FEATS].to_numpy(np.float32))
    return np.concatenate(out)


def load(root: str | None = None):
    """-> (signal: {(m_X, m_Y): float32 array (n, 6)}, qcd: float32 array (n, 6)), signal region only."""
    root = root or os.path.join(DATA, "lhco")
    cache = os.path.join(root, "lhco_sr_features.npz")
    if not os.path.isfile(cache):
        sig_path = fetch("events_anomalydetection_Z_XY_qq_parametric.h5", root)
        qcd_path = fetch("Pythia_QCD_Dijet_Events.h5", root)
        grid = in_sr(features(pd.read_hdf(sig_path)))
        arrays = {}
        for (mx, my), g in grid.groupby(["mx", "my"]):
            if (int(mx), int(my)) not in STRAY:
                arrays[f"sig_{int(mx)}_{int(my)}"] = g[FEATS].to_numpy(np.float32)
        arrays["qcd"] = _qcd_sr(qcd_path)
        np.savez(cache, **arrays)
    z = np.load(cache)
    sig = {tuple(int(v) for v in k.split("_")[1:]): z[k] for k in z.files if k.startswith("sig_")}
    return sig, z["qcd"]


__all__ = ["FEATS", "SR", "features", "in_sr", "load", "fetch"]
