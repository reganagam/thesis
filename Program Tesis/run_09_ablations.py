# -*- coding: utf-8 -*-
"""LANGKAH 9 — ABLASI TAMBAHAN + DIAGNOSTIK SOURCE SHORTCUT.

(a) Diagnostik source shortcut
    Melatih classifier untuk menebak SUMBER DATASET dari gambar. Kalau AUC-nya
    sangat tinggi, sumber sangat mudah dikenali -> Anda WAJIB menunjukkan bahwa
    model utama tidak memanfaatkan sinyal itu (lewat performa per-sumber).
    Satu paragraf di Bab 4 yang mematikan pertanyaan penguji paling tajam.

(b) Ablasi otomatis: preprocessing, backbone, komposisi ISIC, sensitivitas
    Polesie, mode gatekeeper.

    python run_09_ablations.py --what shortcut
    python run_09_ablations.py --what plan      # cetak daftar perintah
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch

import config as C
from src.data import load_manifest, split_fold, build_transforms, make_loader
from src.models import build_model
from src.engine import train_model, set_seed
from src.uncertainty import deterministic_predict
from src.metrics import multiclass_metrics
from src import wb


def source_shortcut(cfg, fold=0, seed=42):
    """Bisakah CNN menebak sumber dataset hanya dari gambar?"""
    df = load_manifest(cfg)
    df = df[~df["is_external"]].copy()
    srcs = sorted(df["source"].unique())
    mapping = {s: i for i, s in enumerate(srcs)}
    df["src_label"] = df["source"].map(mapping)
    print(f"Diagnostik source shortcut: {mapping}")

    tr, iva, te = split_fold(df, fold, cfg, inner_seed=0)
    train_tf, eval_tf = build_transforms(cfg)
    set_seed(seed, cfg.deterministic)

    run = wb.init(cfg, name=f"shortcut_diag_f{fold}", group="diagnostics",
                  job_type="diagnostic")
    model = build_model(cfg, n_out=len(srcs))
    model, hist = train_model(model, tr, iva, cfg, "src_label", len(srcs),
                              train_tf, eval_tf, seed=seed, run=run,
                              tag="shortcut")

    loader = make_loader(te, "src_label", eval_tf, cfg, shuffle=False)
    d = deterministic_predict(model, loader, cfg, n_out=len(srcs))
    yt = te.reset_index(drop=True).iloc[d["row_index"]]["src_label"].values
    m = multiclass_metrics(yt, d["mean_prob"].argmax(1), d["mean_prob"], len(srcs))

    print("\n--- HASIL DIAGNOSTIK SOURCE SHORTCUT ---")
    print(f"  Balanced accuracy menebak SUMBER : {m['BalancedAcc']:.4f}")
    print(f"  AUC (ovr)                        : {m.get('AUC_ovr', float('nan')):.4f}")
    if m["BalancedAcc"] > 0.85:
        print("  [!] Sumber SANGAT mudah dikenali. WAJIB laporkan performa per-sumber")
        print("      dan diskusikan risiko source-label confounding di Bab 4/5.")
    else:
        print("  [OK] Sumber tidak trivial dikenali -> risiko shortcut lebih rendah.")

    out = pd.DataFrame([dict(m, fold=fold, seed=seed, classes=str(mapping))])
    p = os.path.join(C.DIR_RESULT, "diagnostik_source_shortcut.csv")
    out.to_csv(p, index=False)
    print(f"[SAVED] {p}")
    wb.summary(run, {f"shortcut/{k}": v for k, v in m.items()})
    wb.finish(run)
    return m


PLAN = r"""
================= RENCANA ABLASI (jalankan berurutan) =================

# --- 0. Baseline utama (sudah dilakukan di run_02..run_07) ---
python run_02_train_stage1.py
python run_03_train_stage2.py --mode both
python run_04_train_flat.py
python run_05_external_kawahara.py
python run_06_analysis.py
python run_07_visualize.py

# --- 1. Ablasi PREPROCESSING (raw vs proc) ---
#     ubah use_preprocessing di config.py, atau:
python run_02_train_stage1.py --no-preprocessing
python run_03_train_stage2.py --mode oof
python run_04_train_flat.py
python run_06_analysis.py --tag densenet121_raw_subsample

# --- 2. Ablasi BACKBONE ---
python run_02_train_stage1.py --backbone resnet50
python run_03_train_stage2.py --backbone resnet50 --mode oof
python run_04_train_flat.py  --backbone resnet50
python run_06_analysis.py --tag resnet50_proc_subsample

# --- 3. Ablasi KOMPOSISI ISIC ---
python run_00_build_manifest.py --isic-mode full   --out <path>/manifest_isicfull.csv
python run_00_build_manifest.py --isic-mode none   --out <path>/manifest_noisic.csv
#     salin varian ke manifest_frozen.csv lalu ulangi run_02..run_06

# --- 4. Sensitivitas POLESIE (buang bucket 0.6-0.8 mm) ---
python run_00_build_manifest.py --drop-polesie-ambiguous

# --- 5. Ablasi GATEKEEPER (ubah di config.py lalu jalankan ulang run_06 SAJA) ---
#     gatekeeper_mode : 'symmetric' vs 'miv_only'
#     epistemic_measure: 'variance' vs 'bald'
#     reject_percentile: 5 / 10 / 15 / 20
python run_06_analysis.py        # murni offline, tidak perlu training ulang

# --- 6. Diagnostik source shortcut ---
python run_09_ablations.py --what shortcut

CATATAN: ablasi nomor 5 GRATIS — semua sudah tersimpan di CSV prediksi.
========================================================================
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["shortcut", "plan"], default="plan")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.what == "plan":
        print(PLAN)
    else:
        source_shortcut(C.CFG, args.fold, args.seed)


if __name__ == "__main__":
    main()
