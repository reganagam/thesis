# -*- coding: utf-8 -*-
"""LANGKAH 5 — Inferensi ZERO-SHOT pada dataset eksternal KAWAHARA.

Kawahara TIDAK PERNAH dipakai saat training maupun kalibrasi ambang.
Skrip ini hanya MENDUMP prediksi; evaluasi & perakitan skenario dilakukan
di run_06 memakai ambang yang sudah dibekukan dari inner-val internal.

Catatan: Kawahara memakai proxy AJCC-7 (0.76 mm). Kasus riil di pita
0.76-0.80 mm akan salah dilabeli sebagai >=0.8 -> tulis sebagai limitasi.

    python run_05_external_kawahara.py
"""
import argparse
import os
import pandas as pd
import torch

import config as C
from src.data import load_manifest, external_set, build_transforms, make_loader
from src.uncertainty import mc_predict, serialize_samples
from src.checkpoint import load_checkpoint, exists
from src import wb


def dump(part, mc, cfg, fold, seed, stage, extra="", n_out=1):
    d = part.reset_index(drop=True).iloc[mc["row_index"]].reset_index(drop=True)
    base = {
        "img_path": d["img_path"], "image_id": d["image_id"],
        "patient_id": d["patient_id"], "source": d["source"],
        "fold": fold, "seed": seed, "split": "external",
        "stage": stage, "backbone": cfg.backbone, "tag": cfg.tag(),
        "y_true_mis_miv": d["is_invasive"].astype(int),
        "y_true_bt": d["depth_ge_cut"],
        "y_true_3class": d["multiclass"].astype(int),
        "ambiguous_cutoff": d["ambiguous_cutoff"],
        "var_epistemic": mc["var_epistemic"], "bald": mc["bald"],
        "entropy_total": mc["entropy_total"],
        "entropy_aleatoric": mc["entropy_aleatoric"], "msp": mc["msp"],
        "mc_samples": serialize_samples(mc["mc_samples"]),
    }
    if n_out == 1:
        base["p_miv_mean" if stage == "stage1" else "p_bt_mean"] = mc["mean_prob"]
    else:
        P = mc["mean_prob"]
        base.update({"p_c0": P[:, 0], "p_c1": P[:, 1], "p_c2": P[:, 2],
                     "y_pred_3class": P.argmax(1)})
    out = pd.DataFrame(base)
    ex = f"_{extra}" if extra else ""
    fn = f"{stage}_{cfg.tag()}{ex}_external_f{fold}_s{seed}.csv"
    path = os.path.join(C.DIR_PRED, fn)
    out.to_csv(path, index=False)
    print(f"    [pred] {path}  ({len(out)} baris)")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--s2-modes", nargs="*", default=["oof", "gt"])
    ap.add_argument("--backbone", default=None)
    args = ap.parse_args()

    cfg = C.CFG
    if args.backbone:
        cfg.backbone = args.backbone
    seeds = args.seeds if args.seeds else list(cfg.seeds)
    folds = args.folds if args.folds is not None else list(range(cfg.n_folds))

    df = load_manifest(cfg)
    ext = external_set(df)
    print("=" * 70)
    print(f"UJI EKSTERNAL ZERO-SHOT — Kawahara: {len(ext)} gambar / "
          f"{ext.patient_id.nunique()} kasus")
    print(f"Distribusi 3-kelas: {ext['multiclass'].value_counts().sort_index().to_dict()}")
    print("=" * 70)

    _, eval_tf = build_transforms(cfg)
    run = wb.init(cfg, name=f"external_{cfg.tag()}", group="external",
                  job_type="eval", tags=["external", "kawahara"])

    for seed in seeds:
        for fold in folds:
            print(f"\n### seed {seed} | fold {fold} ###")

            # ---- Stage 1 ----
            if exists("stage1", cfg, fold, seed):
                m1, _ = load_checkpoint("stage1", cfg, fold, seed)
                loader = make_loader(ext, "is_invasive", eval_tf, cfg, shuffle=False)
                dump(ext, mc_predict(m1, loader, cfg, n_out=1), cfg, fold, seed,
                     "stage1", n_out=1)
                del m1; torch.cuda.empty_cache()
            else:
                print("  [!] Stage 1 tidak ada")

            # ---- Stage 2 (tiap mode) ----
            for mode in args.s2_modes:
                if exists("stage2", cfg, fold, seed, extra=mode):
                    m2, _ = load_checkpoint("stage2", cfg, fold, seed, extra=mode)
                    loader = make_loader(ext, "depth_ge_cut", eval_tf, cfg, shuffle=False)
                    dump(ext, mc_predict(m2, loader, cfg, n_out=1), cfg, fold, seed,
                         "stage2", extra=mode, n_out=1)
                    del m2; torch.cuda.empty_cache()

            # ---- Flat ----
            if exists("flat", cfg, fold, seed):
                mf, _ = load_checkpoint("flat", cfg, fold, seed)
                loader = make_loader(ext, "multiclass", eval_tf, cfg, shuffle=False)
                dump(ext, mc_predict(mf, loader, cfg, n_out=3), cfg, fold, seed,
                     "flat", n_out=3)
                del mf; torch.cuda.empty_cache()

    wb.finish(run)
    print("\nLANGKAH BERIKUTNYA: python run_06_analysis.py")


if __name__ == "__main__":
    main()
