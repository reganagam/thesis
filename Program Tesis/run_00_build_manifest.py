# -*- coding: utf-8 -*-
"""LANGKAH 0 — Bangun manifest_frozen.csv (WAJIB PERTAMA).

Menggabungkan VRUH + Polesie + ISIC (latih/CV) dan Kawahara (eksternal),
menurunkan label, menetapkan patient_id, menyubsampel ISIC, dan MEMBEKUKAN
pembagian fold ke disk.

    python run_00_build_manifest.py

Setelah ini, COMMIT manifest_frozen.csv ke git. Tidak ada skrip lain yang
boleh memanggil KFold lagi.
"""
import argparse
import pandas as pd

import config as C
from src import manifest as M
from src import wb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isic-mode", choices=["subsample", "full", "none"], default=None)
    ap.add_argument("--drop-polesie-ambiguous", action="store_true",
                    help="analisis sensitivitas: buang bucket 0.6-0.8 mm")
    ap.add_argument("--out", default=None, help="simpan ke path lain (varian ablasi)")
    args = ap.parse_args()

    cfg = C.CFG
    if args.isic_mode:
        cfg.isic_mode = args.isic_mode
    if args.drop_polesie_ambiguous:
        cfg.drop_polesie_ambiguous = True

    df = M.build(cfg, save=False)

    out = args.out or C.MANIFEST
    df.to_csv(out, index=False)
    print(f"\n[SAVED] {out}")

    # ringkasan tabel untuk Bab 3
    names = {0: "Mis", 1: "Miv<0.8", 2: "Miv>=0.8"}
    tab = (df.assign(kelas=df["multiclass"].map(names))
             .pivot_table(index="source", columns="kelas", values="img_path",
                          aggfunc="count", fill_value=0))
    tab.to_csv(f"{C.DIR_RESULT}/tabel_komposisi_dataset.csv")
    print(f"[SAVED] {C.DIR_RESULT}/tabel_komposisi_dataset.csv")

    run = wb.init(cfg, name="build_manifest", group="data", job_type="preprocess")
    wb.log_table(run, df.head(200), "manifest_preview")
    wb.log_table(run, tab.reset_index(), "komposisi_dataset")
    wb.summary(run, {"n_images": len(df),
                     "n_patients": int(df.patient_id.nunique()),
                     "n_external": int(df.is_external.sum())})
    wb.log_artifact(run, out, "manifest_frozen", "dataset")
    wb.finish(run)

    print("\nLANGKAH BERIKUTNYA: python run_01_preprocess_cache.py")


if __name__ == "__main__":
    main()
