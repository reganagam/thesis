# -*- coding: utf-8 -*-
"""Semua figur Bab 4. Dibuat dari CSV prediksi -- TIDAK butuh GPU."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc as sk_auc

import config as C
from src import wb

sns.set_theme(style="whitegrid", context="paper")
DPI = 300


def _save(fig, fname: str, run=None, key=None):
    os.makedirs(C.DIR_FIG, exist_ok=True)
    path = os.path.join(C.DIR_FIG, fname)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    wb.log_image(run, path, key or os.path.splitext(fname)[0])
    print(f"    [fig] {path}")
    return path


# --------------------------------------------------------------------------- #
def plot_confusion(y_true, y_pred, class_names, fname, title="", run=None,
                   normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        cm = cm.astype(float) / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(cm, annot=True, fmt=".3f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar=True, square=True, vmin=0, vmax=1 if normalize else None)
    ax.set_xlabel("Prediksi model"); ax.set_ylabel("Ground truth")
    ax.set_title(title)
    return _save(fig, fname, run)


def plot_roc_binary(y_true, y_prob, fname, title="ROC", run=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    a = sk_auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {a:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right")
    return _save(fig, fname, run)


def plot_roc_multiclass(y_true, y_prob, class_names, fname,
                        title="ROC (one-vs-rest)", run=None):
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    for c, name in enumerate(class_names):
        yt = (np.asarray(y_true) == c).astype(int)
        if yt.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yt, np.asarray(y_prob)[:, c])
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={sk_auc(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=8)
    return _save(fig, fname, run)


# --------------------------------------------------------------------------- #
def plot_uncertainty_hist(scores, threshold, fname, run=None,
                          title="Distribusi ketidakpastian epistemik",
                          correct=None):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    if correct is not None:
        correct = np.asarray(correct).astype(bool)
        ax.hist(np.asarray(scores)[correct], bins=45, alpha=.65,
                label="Prediksi benar", color="#2a9d8f")
        ax.hist(np.asarray(scores)[~correct], bins=45, alpha=.65,
                label="Prediksi salah", color="#e76f51")
    else:
        ax.hist(scores, bins=45, color="#4a7fb5", alpha=.85)
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(threshold, color="crimson", ls="--", lw=2,
                   label=f"Ambang = {threshold:.5f}")
    ax.set_xlabel("Varians epistemik (MC-Dropout)")
    ax.set_ylabel("Jumlah sampel"); ax.set_title(title); ax.legend()
    return _save(fig, fname, run)


def plot_accuracy_vs_rejection(sweeps: dict, fname, run=None,
                               title="Akurasi vs Tingkat Rejeksi",
                               random_df=None):
    """sweeps: {'MC variance': df, 'MSP': df, ...} dari selective.sweep_percentiles."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, df in sweeps.items():
        if df is None or df.empty:
            continue
        ax.plot(df["reject_percentile"], df["accuracy"], marker="o", ms=4,
                lw=1.9, label=name)
    if random_df is not None and not random_df.empty:
        ax.plot(random_df["reject_percentile"], random_df["accuracy"],
                ls=":", color="grey", lw=2, marker="x", ms=4,
                label="Penolakan acak (lantai)")
        if "accuracy_sd" in random_df:
            ax.fill_between(random_df["reject_percentile"],
                            random_df["accuracy"] - random_df["accuracy_sd"],
                            random_df["accuracy"] + random_df["accuracy_sd"],
                            color="grey", alpha=.15)
    ax.set_xlabel("Persentase data ditolak / dirujuk ke dermatolog (%)")
    ax.set_ylabel("Akurasi pada data yang diterima")
    ax.set_title(title); ax.legend(fontsize=8)
    return _save(fig, fname, run)


def plot_risk_coverage(curves: dict, fname, run=None,
                       title="Kurva Risk-Coverage"):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, df in curves.items():
        if df is None or df.empty:
            continue
        ax.plot(df["coverage"], df["risk"], lw=1.9, label=name)
    ax.set_xlabel("Coverage (proporsi kasus yang tetap diprediksi otomatis)")
    ax.set_ylabel("Risk (1 - akurasi)")
    ax.set_title(title + "  — makin rendah makin baik")
    ax.legend(fontsize=8)
    return _save(fig, fname, run)


# --------------------------------------------------------------------------- #
def plot_history(hist: dict, fname_prefix: str, run=None, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(hist["train_loss"], label="train")
    axes[0].plot(hist["val_loss"], label="val")
    if hist.get("best_epoch"):
        axes[0].axvline(hist["best_epoch"] - 1, color="crimson", ls="--", lw=1.2,
                        label="checkpoint terbaik")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[1].plot(hist["val_auc"], label="val AUC")
    axes[1].plot(hist["val_bacc"], label="val balanced acc")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Metrik"); axes[1].legend()
    fig.suptitle(title)
    return _save(fig, f"{fname_prefix}_history.png", run)


def plot_scenario_bars(df: pd.DataFrame, metric: str, fname, run=None,
                       title=None):
    """df: kolom scenario, <metric>_mean, <metric>_sd."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(df))
    ax.bar(x, df[f"{metric}_mean"], yerr=df.get(f"{metric}_sd"),
           capsize=4, color=sns.color_palette("Set2", len(df)))
    ax.set_xticks(x); ax.set_xticklabels(df["scenario"], rotation=15, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} per skenario (mean ± sd lintas fold × seed)")
    for i, v in enumerate(df[f"{metric}_mean"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    return _save(fig, fname, run)


def plot_per_source(df: pd.DataFrame, metric: str, fname, run=None):
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    sns.barplot(data=df, x="source", y=metric, hue="scenario", ax=ax,
                errorbar="sd", palette="Set2")
    ax.set_title(f"{metric} per sumber dataset")
    ax.legend(fontsize=8, title=None)
    return _save(fig, fname, run)


def plot_calibration(y_true, y_prob, fname, run=None, n_bins=10,
                     title="Reliability diagram"):
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(y_prob[m].mean()); ys.append(y_true[m].mean()); ns.append(m.sum())
    ece = float(np.sum(np.array(ns) * np.abs(np.array(xs) - np.array(ys))) /
                max(len(y_true), 1))
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="kalibrasi sempurna")
    ax.plot(xs, ys, marker="o", lw=1.8, label=f"model (ECE={ece:.4f})")
    ax.set_xlabel("Probabilitas prediksi"); ax.set_ylabel("Frekuensi observasi")
    ax.set_title(title); ax.legend()
    return _save(fig, fname, run)
