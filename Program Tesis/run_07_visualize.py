# -*- coding: utf-8 -*-
"""LANGKAH 7 — Semua FIGUR Bab 4. TIDAK BUTUH GPU.

Dibuat sepenuhnya dari CSV prediksi + CSV hasil analisis, sehingga bisa
dijalankan berkali-kali kapan saja tanpa training ulang.

Menghasilkan:
  - confusion matrix 3x3 untuk A / B / C
  - ROC one-vs-rest & ROC biner Stage 1 / Stage 2
  - histogram ketidakpastian (benar vs salah) + garis ambang
  - kurva Accuracy vs Rejection (semua skor + lantai acak)
  - kurva Risk-Coverage
  - bar chart perbandingan skenario
  - reliability diagram (kalibrasi)

    python run_07_visualize.py
    python run_07_visualize.py --split external
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

import config as C
from src import plots, wb
from src.selective import (assemble_hierarchical, hierarchical_prob_matrix,
                           calibrate_threshold)


def _read(pattern):
    files = sorted(glob.glob(os.path.join(C.DIR_PRED, pattern)))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files \
        else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "external"], default="test")
    ap.add_argument("--s2-mode", default="oof")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = C.CFG
    tag = args.tag or cfg.tag()
    split = args.split
    sfx = f"{split}_{args.s2_mode}_{tag}"
    os.makedirs(C.DIR_FIG, exist_ok=True)

    run = wb.init(cfg, name=f"figures_{sfx}", group="figures", job_type="viz")
    print("=" * 70)
    print(f"MEMBUAT FIGUR — {sfx}")
    print("=" * 70)

    s1 = _read(f"stage1_{tag}_{split}_f*_s*.csv")
    s2 = _read(f"stage2_{tag}_{args.s2_mode}_{split}_f*_s*.csv")
    fl = _read(f"flat_{tag}_{split}_f*_s*.csv")
    s1_iv = _read(f"stage1_{tag}_innerval_f*_s*.csv")

    score_col = "bald" if cfg.epistemic_measure == "bald" else "var_epistemic"

    # ---------------- Skenario A ----------------
    if not fl.empty:
        plots.plot_confusion(fl["y_true_3class"], fl["y_pred_3class"],
                             C.CLASS_NAMES_3, f"fig_cm_A_flat_{sfx}.png",
                             "Skenario A — Flat multiclass", run)
        plots.plot_roc_multiclass(fl["y_true_3class"],
                                  fl[["p_c0", "p_c1", "p_c2"]].values,
                                  C.CLASS_NAMES_3,
                                  f"fig_roc_A_flat_{sfx}.png",
                                  "Skenario A — ROC one-vs-rest", run)

    # ---------------- Stage 1 ----------------
    if not s1.empty:
        plots.plot_roc_binary(s1["y_true_mis_miv"], s1["p_miv_mean"],
                              f"fig_roc_stage1_{sfx}.png",
                              "Stage 1 — Mis vs Miv", run)
        plots.plot_calibration(s1["y_true_mis_miv"], s1["p_miv_mean"],
                               f"fig_calibration_stage1_{sfx}.png", run,
                               title="Kalibrasi Stage 1")

    # ---------------- Stage 2 ----------------
    if not s2.empty:
        ev = s2[s2["y_true_bt"].notna()]
        if len(ev) and ev["y_true_bt"].nunique() > 1:
            plots.plot_roc_binary(ev["y_true_bt"].astype(int), ev["p_bt_mean"],
                                  f"fig_roc_stage2_{sfx}.png",
                                  "Stage 2 — <0.8 mm vs >=0.8 mm", run)

    # ---------------- Skenario B & C ----------------
    if not s1.empty and not s2.empty:
        h = s1.merge(s2[["img_path", "fold", "seed", "p_bt_mean"]],
                     on=["img_path", "fold", "seed"], how="left")
        yp = assemble_hierarchical(h["p_miv_mean"].values, h["p_bt_mean"].values)
        yt = h["y_true_3class"].values
        prob = hierarchical_prob_matrix(h["p_miv_mean"].values, h["p_bt_mean"].values)

        plots.plot_confusion(yt, yp, C.CLASS_NAMES_3,
                             f"fig_cm_B_hier_{sfx}.png",
                             "Skenario B — Hierarkis tanpa gatekeeper", run)
        plots.plot_roc_multiclass(yt, prob, C.CLASS_NAMES_3,
                                  f"fig_roc_B_hier_{sfx}.png",
                                  "Skenario B — ROC one-vs-rest", run)

        # ambang bersarang dari inner-val
        thr = (calibrate_threshold(s1_iv[score_col].values, cfg.reject_percentile)
               if not s1_iv.empty else
               float(np.percentile(h[score_col].values, 100 - cfg.reject_percentile)))
        keep = h[score_col].values <= thr
        if keep.sum() > 5:
            plots.plot_confusion(yt[keep], yp[keep], C.CLASS_NAMES_3,
                                 f"fig_cm_C_gate_{sfx}.png",
                                 f"Skenario C — Gatekeeper (coverage {keep.mean():.1%})",
                                 run)

        correct = (yt == yp).astype(int)
        plots.plot_uncertainty_hist(h[score_col].values, thr,
                                    f"fig_uncertainty_hist_{sfx}.png", run,
                                    correct=correct)

        # ---- Accuracy vs Rejection & Risk-Coverage ----
        sweeps, rcs = {}, {}
        for label, pat in [("MC variance", "sweep_MC_variance"),
                           ("BALD", "sweep_BALD"),
                           ("Predictive entropy", "sweep_Predictive_entropy"),
                           ("MSP", "sweep_MSP")]:
            f = glob.glob(os.path.join(C.DIR_RESULT, f"{pat}*_{sfx}.csv"))
            if f:
                d = pd.read_csv(f[0])
                sweeps[label] = d.groupby("reject_percentile", as_index=False)[
                    ["accuracy", "coverage"]].mean()
        rnd = glob.glob(os.path.join(C.DIR_RESULT, f"sweep_Penolakan_acak*_{sfx}.csv"))
        rnd_df = (pd.read_csv(rnd[0]).groupby("reject_percentile", as_index=False)
                  [["accuracy", "accuracy_sd"]].mean()) if rnd else None
        if sweeps:
            plots.plot_accuracy_vs_rejection(sweeps,
                                             f"fig_accuracy_vs_rejection_{sfx}.png",
                                             run, random_df=rnd_df)

        for label, pat in [("MC variance", "riskcov_MC_variance"),
                           ("BALD", "riskcov_BALD"),
                           ("Predictive entropy", "riskcov_Predictive_entropy"),
                           ("MSP", "riskcov_MSP")]:
            f = glob.glob(os.path.join(C.DIR_RESULT, f"{pat}*_{sfx}.csv"))
            if f:
                d = pd.read_csv(f[0])
                d["cov_bin"] = (d["coverage"] * 100).round().astype(int)
                rcs[label] = d.groupby("cov_bin", as_index=False)[
                    ["coverage", "risk"]].mean()
        if rcs:
            plots.plot_risk_coverage(rcs, f"fig_risk_coverage_{sfx}.png", run)

    # ---------------- bar chart skenario ----------------
    f = os.path.join(C.DIR_RESULT, f"tabel_ablasi_{sfx}.csv")
    if os.path.exists(f):
        tab = pd.read_csv(f)
        tab = tab[tab.scenario.isin(C.SCENARIOS)]
        for metric in ["BalancedAcc", "MacroF1", "Kappa"]:
            if f"{metric}_mean" in tab:
                plots.plot_scenario_bars(tab, metric,
                                         f"fig_bar_{metric}_{sfx}.png", run)

    wb.finish(run)
    print("\nSemua figur -> ", C.DIR_FIG)
    print("LANGKAH BERIKUTNYA: python run_08_gradcam.py")


if __name__ == "__main__":
    main()
