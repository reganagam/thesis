# -*- coding: utf-8 -*-
"""Metrik evaluasi + selective prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, cohen_kappa_score,
                             accuracy_score, confusion_matrix)


# --------------------------------------------------------------------------- #
def binary_metrics(y_true, y_prob, thr: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {
        "AUC": auc,
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "N": int(len(y_true)),
    }


def multiclass_metrics(y_true, y_pred, y_prob=None, n_classes: int = 3) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "MacroF1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MacroPrecision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "MacroRecall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "N": int(len(y_true)),
    }
    # per-kelas recall (sensitivitas) -- penting untuk kelas >=0.8 mm
    for c in range(n_classes):
        m = (y_true == c)
        out[f"Recall_c{c}"] = float((y_pred[m] == c).mean()) if m.any() else float("nan")
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            out["AUC_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                           average="macro",
                                           labels=list(range(n_classes)))
        except Exception:
            out["AUC_ovr"] = float("nan")
        for c in range(n_classes):
            try:
                out[f"AUC_c{c}"] = roc_auc_score((y_true == c).astype(int), y_prob[:, c])
            except Exception:
                out[f"AUC_c{c}"] = float("nan")
    # eror paling fatal secara klinis: Miv>=0.8 diprediksi Mis
    m = (y_true == 2)
    out["FatalMiss_c2_to_c0"] = float((y_pred[m] == 0).mean()) if m.any() else float("nan")
    return out


# --------------------------------------------------------------------------- #
def risk_coverage_curve(correct, uncertainty, n_points: int = 101) -> pd.DataFrame:
    """Urutkan dari paling yakin, hitung risk (1-akurasi) pada tiap coverage."""
    correct = np.asarray(correct).astype(float)
    unc = np.asarray(uncertainty, dtype=float)
    order = np.argsort(unc, kind="mergesort")     # menaik = paling yakin dulu
    c_sorted = correct[order]
    n = len(c_sorted)
    if n == 0:
        return pd.DataFrame(columns=["coverage", "risk", "accuracy", "n_kept"])
    cum_correct = np.cumsum(c_sorted)
    ks = np.unique(np.clip(np.linspace(1, n, min(n_points, n)).astype(int), 1, n))
    cov = ks / n
    acc = cum_correct[ks - 1] / ks
    return pd.DataFrame({"coverage": cov, "risk": 1 - acc,
                         "accuracy": acc, "n_kept": ks})


def aurc(correct, uncertainty) -> float:
    """Area Under Risk-Coverage Curve. Makin KECIL makin baik."""
    df = risk_coverage_curve(correct, uncertainty, n_points=1000)
    if df.empty:
        return float("nan")
    return float(np.trapezoid(df["risk"].values, df["coverage"].values)
                 / max(df["coverage"].values[-1] - df["coverage"].values[0], 1e-9))


def error_detection_auroc(correct, uncertainty) -> float:
    """Seberapa baik skor ketidakpastian membedakan prediksi salah vs benar.

    Ini BUKTI MEKANISTIK bahwa gatekeeper menangkap eror, bukan sekadar
    membuang kasus sulit secara acak.
    """
    err = 1 - np.asarray(correct).astype(int)
    if err.sum() == 0 or err.sum() == len(err):
        return float("nan")
    return float(roc_auc_score(err, np.asarray(uncertainty, dtype=float)))


def accuracy_at_coverage(correct, uncertainty, coverage: float) -> float:
    correct = np.asarray(correct).astype(float)
    unc = np.asarray(uncertainty, dtype=float)
    n_keep = max(1, int(round(len(correct) * coverage)))
    order = np.argsort(unc, kind="mergesort")[:n_keep]
    return float(correct[order].mean())


def sensitivity_at_specificity(y_true, y_prob, target_spec: float = 0.80) -> float:
    from sklearn.metrics import roc_curve
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, np.asarray(y_prob, dtype=float))
    ok = np.where((1 - fpr) >= target_spec)[0]
    return float(tpr[ok].max()) if len(ok) else float("nan")


# --------------------------------------------------------------------------- #
def summarize_folds(rows, group_cols=("scenario",)) -> pd.DataFrame:
    """mean ± sd lintas fold/seed."""
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
           and c not in group_cols and c not in ("fold", "seed")]
    g = df.groupby(list(group_cols))[num]
    out = g.mean().round(4).add_suffix("_mean").join(
        g.std().round(4).add_suffix("_sd"))
    return out.reset_index()


def wilcoxon_pairs(df: pd.DataFrame, metric: str, key="scenario",
                   block=("fold", "seed")) -> pd.DataFrame:
    """Wilcoxon signed-rank berpasangan antar-skenario pada fold/seed yang sama."""
    from scipy.stats import wilcoxon
    from itertools import combinations
    piv = df.pivot_table(index=list(block), columns=key, values=metric)
    rows = []
    for a, b in combinations(piv.columns, 2):
        s = piv[[a, b]].dropna()
        if len(s) < 3:
            continue
        try:
            stat, p = wilcoxon(s[a], s[b])
        except Exception:
            stat, p = float("nan"), float("nan")
        rows.append({"metric": metric, "A": a, "B": b, "n_pairs": len(s),
                     "median_A": s[a].median(), "median_B": s[b].median(),
                     "stat": stat, "p_value": p,
                     "significant_0.05": bool(p < 0.05) if p == p else False})
    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    means = [rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))
