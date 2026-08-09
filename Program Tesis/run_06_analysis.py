# -*- coding: utf-8 -*-
"""LANGKAH 6 — ANALISIS OFFLINE (TANPA GPU). Inti Bab 4.

Membaca CSV prediksi dari run_02..run_05 dan menghasilkan:
  1. Tabel ablasi A / A-rej / B / C pada ruang 3-kelas yang sama
  2. Kalibrasi ambang BERSARANG (inner-val -> test) + coverage aktual
  3. Kurva Accuracy vs Rejection & Risk-Coverage untuk SEMUA skor
     ketidakpastian (acak, MSP, entropy, variance, BALD) + AURC
  4. AUROC deteksi eror (bukti mekanistik gatekeeper)
  5. Breakdown per sumber dataset
  6. Uji Wilcoxon berpasangan antar-skenario
  7. Hasil eksternal Kawahara

    python run_06_analysis.py
    python run_06_analysis.py --split external
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

import config as C
from src.metrics import (multiclass_metrics, aurc, error_detection_auroc,
                         risk_coverage_curve, summarize_folds, wilcoxon_pairs,
                         bootstrap_ci)
from src.selective import (calibrate_threshold, apply_gatekeeper,
                           gatekeeper_eligibility, assemble_hierarchical,
                           hierarchical_prob_matrix, sweep_percentiles,
                           random_rejection_baseline)
from src import wb

UNC_SCORES = {
    "MC variance (epistemik)": "var_epistemic",
    "BALD (mutual information)": "bald",
    "Predictive entropy": "entropy_total",
    "MSP (1 forward pass)": "msp_inv",     # dibalik: makin kecil MSP = makin ragu
}


def _read(pattern):
    files = sorted(glob.glob(os.path.join(C.DIR_PRED, pattern)))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def add_msp_inv(df):
    if "msp" in df:
        df["msp_inv"] = 1.0 - df["msp"]
    return df


# --------------------------------------------------------------------------- #
def build_hierarchical(s1, s2, fold, seed, split):
    """Gabungkan Stage 1 & Stage 2 pada gambar yang sama."""
    a = s1[(s1.fold == fold) & (s1.seed == seed) & (s1.split == split)]
    b = s2[(s2.fold == fold) & (s2.seed == seed) & (s2.split == split)]
    if a.empty or b.empty:
        return pd.DataFrame()
    b = b[["img_path", "p_bt_mean", "var_epistemic_s2", "msp_s2"]]
    m = a.merge(b, on="img_path", how="left")
    m["y_pred_3class_hier"] = assemble_hierarchical(m["p_miv_mean"].values,
                                                    m["p_bt_mean"].values)
    return m


def evaluate_scenarios(cfg, split="test", tag=None, s2_mode="oof"):
    tag = tag or cfg.tag()
    s1 = add_msp_inv(_read(f"stage1_{tag}_{split}_f*_s*.csv"))
    s2 = add_msp_inv(_read(f"stage2_{tag}_{s2_mode}_{split}_f*_s*.csv"))
    fl = add_msp_inv(_read(f"flat_{tag}_{split}_f*_s*.csv"))

    s1_iv = add_msp_inv(_read(f"stage1_{tag}_innerval_f*_s*.csv"))
    fl_iv = add_msp_inv(_read(f"flat_{tag}_innerval_f*_s*.csv"))

    if s1.empty and fl.empty:
        raise SystemExit(f"Tidak ada CSV prediksi untuk tag='{tag}' split='{split}'. "
                         "Jalankan run_02..run_05 dulu.")

    rows, sweeps_all, rc_all, thr_rows = [], {}, {}, []
    combos = sorted(set(zip(s1.fold, s1.seed))) if not s1.empty \
        else sorted(set(zip(fl.fold, fl.seed)))

    for fold, seed in combos:
        # ---------------- Skenario A : flat multiclass ----------------
        fa = fl[(fl.fold == fold) & (fl.seed == seed) & (fl.split == split)]
        if not fa.empty:
            yt = fa["y_true_3class"].values
            yp = fa["y_pred_3class"].values
            prob = fa[["p_c0", "p_c1", "p_c2"]].values
            m = multiclass_metrics(yt, yp, prob, 3)
            m.update({"scenario": "A_flat", "fold": fold, "seed": seed,
                      "coverage": 1.0, "n_rejected": 0})
            rows.append(m)

            # ---- Skenario A-rej : flat + penolakan (KONTROL WAJIB) ----
            iv = fl_iv[(fl_iv.fold == fold) & (fl_iv.seed == seed)]
            score_col = ("bald" if cfg.epistemic_measure == "bald" else "var_epistemic")
            if not iv.empty:
                thr = calibrate_threshold(iv[score_col].values, cfg.reject_percentile)
                keep = fa[score_col].values <= thr
                if keep.sum() > 5:
                    m2 = multiclass_metrics(yt[keep], yp[keep], prob[keep], 3)
                    m2.update({"scenario": "A_flat_reject", "fold": fold, "seed": seed,
                               "coverage": float(keep.mean()),
                               "n_rejected": int((~keep).sum()),
                               "threshold_abs": thr})
                    rows.append(m2)

        # ---------------- Skenario B & C : hierarkis ----------------
        h = build_hierarchical(s1, s2, fold, seed, split)
        if h.empty:
            continue
        yt = h["y_true_3class"].values
        yp = h["y_pred_3class_hier"].values
        prob = hierarchical_prob_matrix(h["p_miv_mean"].values, h["p_bt_mean"].values)

        m = multiclass_metrics(yt, yp, prob, 3)
        m.update({"scenario": "B_hier", "fold": fold, "seed": seed,
                  "coverage": 1.0, "n_rejected": 0})
        rows.append(m)

        # --- kalibrasi ambang BERSARANG dari inner-val ---
        iv = s1_iv[(s1_iv.fold == fold) & (s1_iv.seed == seed)]
        score_col = "bald" if cfg.epistemic_measure == "bald" else "var_epistemic"
        if iv.empty:
            continue
        thr = calibrate_threshold(iv[score_col].values, cfg.reject_percentile)
        elig = gatekeeper_eligibility(h["p_miv_mean"].values >= 0.5, cfg.gatekeeper_mode)
        rej = apply_gatekeeper(h[score_col].values, thr, elig)
        keep = ~rej
        thr_rows.append({"fold": fold, "seed": seed, "threshold_abs": thr,
                         "reject_percentile_target": cfg.reject_percentile,
                         "coverage_actual": float(keep.mean()),
                         "n_rejected": int(rej.sum()),
                         "gatekeeper_mode": cfg.gatekeeper_mode,
                         "epistemic_measure": cfg.epistemic_measure})

        if keep.sum() > 5:
            mc = multiclass_metrics(yt[keep], yp[keep], prob[keep], 3)
            mc.update({"scenario": "C_hier_gate", "fold": fold, "seed": seed,
                       "coverage": float(keep.mean()),
                       "n_rejected": int(rej.sum()), "threshold_abs": thr})
            rows.append(mc)

        # --- perbandingan SEMUA skor ketidakpastian pada coverage sama ---
        correct = (yt == yp).astype(int)
        for label, col in UNC_SCORES.items():
            if col not in h:
                continue
            sw = sweep_percentiles(correct, h[col].values, cfg.percentile_sweep) \
                if hasattr(cfg, "percentile_sweep") else \
                sweep_percentiles(correct, h[col].values,
                                  [0, 2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30])
            sw["fold"], sw["seed"], sw["score"] = fold, seed, label
            sweeps_all.setdefault(label, []).append(sw)

            rc = risk_coverage_curve(correct, h[col].values)
            rc["fold"], rc["seed"], rc["score"] = fold, seed, label
            rc_all.setdefault(label, []).append(rc)

            rows.append({"scenario": f"unc::{label}", "fold": fold, "seed": seed,
                         "AURC": aurc(correct, h[col].values),
                         "ErrorDetection_AUROC": error_detection_auroc(
                             correct, h[col].values),
                         "N": len(correct)})

        # --- lantai acak ---
        rnd = random_rejection_baseline(correct, [0, 2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30],
                                        seed=seed)
        rnd["fold"], rnd["seed"], rnd["score"] = fold, seed, "Penolakan acak"
        sweeps_all.setdefault("Penolakan acak", []).append(rnd)

        # --- breakdown per sumber ---
        for src in h["source"].unique():
            ms = h["source"] == src
            if ms.sum() < 5:
                continue
            msrc = multiclass_metrics(yt[ms.values], yp[ms.values], prob[ms.values], 3)
            msrc.update({"scenario": "B_hier", "fold": fold, "seed": seed,
                         "source": src})
            rows.append(msrc)

    return (pd.DataFrame(rows),
            {k: pd.concat(v, ignore_index=True) for k, v in sweeps_all.items()},
            {k: pd.concat(v, ignore_index=True) for k, v in rc_all.items()},
            pd.DataFrame(thr_rows))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "external"], default="test")
    ap.add_argument("--s2-mode", default="oof")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = C.CFG
    os.makedirs(C.DIR_RESULT, exist_ok=True)
    sfx = f"{args.split}_{args.s2_mode}_{args.tag or cfg.tag()}"

    print("=" * 70)
    print(f"ANALISIS — split={args.split} | Stage2 mode={args.s2_mode}")
    print("=" * 70)

    res, sweeps, rcs, thr = evaluate_scenarios(cfg, args.split, args.tag, args.s2_mode)
    run = wb.init(cfg, name=f"analysis_{sfx}", group="analysis", job_type="analysis")

    # ---------- 1. tabel ablasi utama ----------
    main_res = res[res.scenario.isin(C.SCENARIOS) & res.get("source", pd.Series(dtype=object)).isna()] \
        if "source" in res else res[res.scenario.isin(C.SCENARIOS)]
    if "source" in res.columns:
        main_res = res[res.scenario.isin(C.SCENARIOS) & res["source"].isna()]
    else:
        main_res = res[res.scenario.isin(C.SCENARIOS)]

    if not main_res.empty:
        tab = summarize_folds(main_res.drop(columns=["source"], errors="ignore"),
                              group_cols=("scenario",))
        p = os.path.join(C.DIR_RESULT, f"tabel_ablasi_{sfx}.csv")
        tab.to_csv(p, index=False)
        print("\n--- TABEL ABLASI UTAMA (mean ± sd) ---")
        show = [c for c in ["scenario", "BalancedAcc_mean", "BalancedAcc_sd",
                            "MacroF1_mean", "AUC_ovr_mean", "Kappa_mean",
                            "coverage_mean", "FatalMiss_c2_to_c0_mean"] if c in tab]
        print(tab[show].round(4).to_string(index=False))
        print(f"[SAVED] {p}")
        wb.log_table(run, tab, f"ablasi_{args.split}")

        # bootstrap CI
        ci = []
        for sc in main_res.scenario.unique():
            v = main_res.loc[main_res.scenario == sc, "BalancedAcc"]
            lo, hi = bootstrap_ci(v)
            ci.append({"scenario": sc, "BalancedAcc_mean": v.mean(),
                       "CI95_low": lo, "CI95_high": hi, "n": len(v)})
        pd.DataFrame(ci).to_csv(
            os.path.join(C.DIR_RESULT, f"bootstrap_ci_{sfx}.csv"), index=False)

    # ---------- 2. ambang & coverage aktual ----------
    if not thr.empty:
        p = os.path.join(C.DIR_RESULT, f"gatekeeper_threshold_{sfx}.csv")
        thr.to_csv(p, index=False)
        print("\n--- KALIBRASI AMBANG BERSARANG ---")
        print(f"  target rejeksi   : {cfg.reject_percentile}%")
        print(f"  ambang absolut   : {thr.threshold_abs.mean():.6f} "
              f"± {thr.threshold_abs.std():.6f}")
        print(f"  COVERAGE AKTUAL  : {thr.coverage_actual.mean():.4f} "
              f"± {thr.coverage_actual.std():.4f}   <-- laporkan angka ini")
        print(f"[SAVED] {p}")
        wb.log_table(run, thr, "gatekeeper_threshold")

    # ---------- 3. perbandingan skor ketidakpastian ----------
    unc = res[res.scenario.astype(str).str.startswith("unc::")].copy()
    if not unc.empty:
        unc["score"] = unc.scenario.str.replace("unc::", "", regex=False)
        agg = unc.groupby("score")[["AURC", "ErrorDetection_AUROC"]] \
                 .agg(["mean", "std"]).round(4)
        p = os.path.join(C.DIR_RESULT, f"tabel_uncertainty_scores_{sfx}.csv")
        agg.to_csv(p)
        print("\n--- PERBANDINGAN SKOR KETIDAKPASTIAN ---")
        print("  (AURC makin KECIL makin baik; ErrorDetection AUROC makin BESAR makin baik)")
        print(agg.to_string())
        print(f"[SAVED] {p}")
        wb.log_table(run, agg.reset_index(), "uncertainty_scores")

    # ---------- 4. kurva ----------
    for name, df in sweeps.items():
        df.to_csv(os.path.join(C.DIR_RESULT,
                  f"sweep_{name.split('(')[0].strip().replace(' ','_')}_{sfx}.csv"),
                  index=False)
    for name, df in rcs.items():
        df.to_csv(os.path.join(C.DIR_RESULT,
                  f"riskcov_{name.split('(')[0].strip().replace(' ','_')}_{sfx}.csv"),
                  index=False)
    print(f"\n[SAVED] kurva sweep & risk-coverage -> {C.DIR_RESULT}")

    # ---------- 5. per sumber ----------
    if "source" in res.columns:
        per_src = res[res["source"].notna()]
        if not per_src.empty:
            t = per_src.groupby("source")[["BalancedAcc", "MacroF1", "Kappa"]] \
                       .agg(["mean", "std"]).round(4)
            p = os.path.join(C.DIR_RESULT, f"tabel_per_sumber_{sfx}.csv")
            t.to_csv(p)
            print("\n--- BREAKDOWN PER SUMBER DATASET ---")
            print(t.to_string()); print(f"[SAVED] {p}")
            wb.log_table(run, t.reset_index(), "per_source")

    # ---------- 6. uji statistik ----------
    if not main_res.empty and main_res.scenario.nunique() > 1:
        for metric in ["BalancedAcc", "MacroF1"]:
            if metric not in main_res:
                continue
            w = wilcoxon_pairs(main_res, metric)
            if not w.empty:
                p = os.path.join(C.DIR_RESULT, f"wilcoxon_{metric}_{sfx}.csv")
                w.to_csv(p, index=False)
                print(f"\n--- WILCOXON SIGNED-RANK ({metric}) ---")
                print(w.round(4).to_string(index=False))
                print(f"[SAVED] {p}")
                wb.log_table(run, w, f"wilcoxon_{metric}")

    res.to_csv(os.path.join(C.DIR_RESULT, f"raw_all_metrics_{sfx}.csv"), index=False)
    wb.finish(run)
    print("\nLANGKAH BERIKUTNYA: python run_07_visualize.py")


if __name__ == "__main__":
    main()
