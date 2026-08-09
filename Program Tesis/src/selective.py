# -*- coding: utf-8 -*-
"""Gatekeeper + perakitan prediksi hierarkis 3-kelas.

Kalibrasi ambang BERSARANG (anti kebocoran):
  1. hitung distribusi ketidakpastian pada INNER-VAL,
  2. ambil nilai absolut pada persentil (100 - X),
  3. BEKUKAN, terapkan ke test fold,
  4. laporkan COVERAGE AKTUAL (tidak akan persis 100-X% karena distribusi bergeser).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
def calibrate_threshold(inner_scores: np.ndarray, reject_percentile: float) -> float:
    """Ambang absolut dari inner-val. reject_percentile = X (buang top X%)."""
    s = np.asarray(inner_scores, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return float("inf")
    if reject_percentile <= 0:
        return float("inf")
    return float(np.percentile(s, 100.0 - reject_percentile))


def apply_gatekeeper(scores: np.ndarray, threshold: float,
                     eligible: np.ndarray = None) -> np.ndarray:
    """True = DITOLAK (dirujuk ke dermatolog).

    eligible: mask sampel yang boleh ditolak. Untuk mode 'miv_only',
    hanya sampel yang diprediksi Miv yang boleh ditolak.
    """
    s = np.asarray(scores, dtype=float)
    rejected = s > threshold
    if eligible is not None:
        rejected = rejected & np.asarray(eligible, dtype=bool)
    return rejected


def gatekeeper_eligibility(pred_miv: np.ndarray, mode: str) -> np.ndarray:
    """mode 'symmetric' -> semua sampel boleh ditolak.
    mode 'miv_only'  -> hanya yang diprediksi Miv.

    Catatan desain: mode simetris direkomendasikan karena eror paling fatal
    secara klinis justru Miv>=0.8 yang diprediksi Mis -- pasien kehilangan SLNB.
    """
    pred_miv = np.asarray(pred_miv, dtype=bool)
    return np.ones_like(pred_miv, dtype=bool) if mode == "symmetric" else pred_miv


# --------------------------------------------------------------------------- #
def assemble_hierarchical(p_miv: np.ndarray, p_depth: np.ndarray,
                          thr1: float = 0.5, thr2: float = 0.5) -> np.ndarray:
    """Gabungkan Stage 1 + Stage 2 menjadi prediksi 3-kelas.

    0 = Mis, 1 = Miv<cut, 2 = Miv>=cut
    p_depth boleh NaN untuk sampel yang diprediksi Mis (tidak dipakai).
    """
    p_miv = np.asarray(p_miv, dtype=float)
    p_depth = np.asarray(p_depth, dtype=float)
    pred = np.zeros(len(p_miv), dtype=int)
    is_miv = p_miv >= thr1
    pred[is_miv] = np.where(np.nan_to_num(p_depth[is_miv], nan=0.0) >= thr2, 2, 1)
    return pred


def hierarchical_prob_matrix(p_miv: np.ndarray, p_depth: np.ndarray) -> np.ndarray:
    """Probabilitas 3-kelas dari aturan rantai (untuk AUC one-vs-rest).

    P(Mis)      = 1 - P(Miv)
    P(Miv<cut)  = P(Miv) * (1 - P(>=cut))
    P(Miv>=cut) = P(Miv) * P(>=cut)
    """
    pm = np.asarray(p_miv, dtype=float)
    pd_ = np.nan_to_num(np.asarray(p_depth, dtype=float), nan=0.5)
    return np.stack([1 - pm, pm * (1 - pd_), pm * pd_], axis=1)


# --------------------------------------------------------------------------- #
def sweep_percentiles(correct: np.ndarray, scores: np.ndarray,
                      percentiles) -> pd.DataFrame:
    """Kurva Accuracy vs Rejection untuk Bab 4."""
    correct = np.asarray(correct).astype(float)
    scores = np.asarray(scores, dtype=float)
    n = len(correct)
    rows = []
    for x in percentiles:
        if n == 0:
            continue
        thr = float(np.percentile(scores, 100.0 - x)) if x > 0 else float("inf")
        keep = scores <= thr
        if keep.sum() == 0:
            continue
        rows.append({
            "reject_percentile": x,
            "threshold_abs": thr if np.isfinite(thr) else np.nan,
            "coverage": keep.mean(),
            "n_kept": int(keep.sum()),
            "n_rejected": int((~keep).sum()),
            "accuracy": float(correct[keep].mean()),
            "risk": 1 - float(correct[keep].mean()),
            "accuracy_on_rejected": (float(correct[~keep].mean())
                                     if (~keep).any() else np.nan),
        })
    return pd.DataFrame(rows)


def random_rejection_baseline(correct: np.ndarray, percentiles,
                              n_rep: int = 50, seed: int = 0) -> pd.DataFrame:
    """Lantai absolut: buang X% secara ACAK. Kalau metode Anda tidak jauh
    di atas ini, klaim novelty gugur."""
    correct = np.asarray(correct).astype(float)
    rng = np.random.RandomState(seed)
    n = len(correct)
    rows = []
    for x in percentiles:
        k = int(round(n * (1 - x / 100.0)))
        if k <= 0:
            continue
        accs = [correct[rng.choice(n, k, replace=False)].mean() for _ in range(n_rep)]
        rows.append({"reject_percentile": x, "coverage": k / n,
                     "accuracy": float(np.mean(accs)),
                     "accuracy_sd": float(np.std(accs)),
                     "risk": 1 - float(np.mean(accs))})
    return pd.DataFrame(rows)
