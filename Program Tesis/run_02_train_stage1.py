# -*- coding: utf-8 -*-
"""LANGKAH 2 — Latih STAGE 1 (Mis vs Miv) + dump prediksi MC-Dropout.

Untuk setiap (seed, fold):
  * latih model biner is_invasive
  * simpan checkpoint BERMETADATA (termasuk history -> kurva loss)
  * jalankan MC-Dropout T kali pada INNER-VAL (untuk kalibrasi ambang) dan
    pada TEST FOLD (untuk evaluasi) -> tulis CSV per-sampel

CSV inilah yang membuat seluruh Bab 4 bisa dikerjakan tanpa GPU.

    python run_02_train_stage1.py
    python run_02_train_stage1.py --seeds 42 --folds 0 1
    python run_02_train_stage1.py --backbone resnet50
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch

import config as C
from src.data import load_manifest, split_fold, build_transforms, make_loader
from src.models import build_model, count_params
from src.engine import train_model, set_seed
from src.uncertainty import mc_predict, serialize_samples
from src.metrics import binary_metrics
from src.checkpoint import save_checkpoint, exists
from src.plots import plot_history
from src import wb

STAGE = "stage1"
LABEL = "is_invasive"


def dump_predictions(df_split, mc, split_name, cfg, fold, seed, extra=""):
    """Tulis satu baris per gambar: prediksi + SEMUA skor ketidakpastian."""
    d = df_split.reset_index(drop=True).iloc[mc["row_index"]].reset_index(drop=True)
    out = pd.DataFrame({
        "img_path": d["img_path"], "image_id": d["image_id"],
        "patient_id": d["patient_id"], "source": d["source"],
        "fold": fold, "seed": seed, "split": split_name,
        "stage": STAGE, "backbone": cfg.backbone, "tag": cfg.tag(),
        "y_true_mis_miv": d["is_invasive"].astype(int),
        "y_true_bt": d["depth_ge_cut"],
        "y_true_3class": d["multiclass"].astype(int),
        "ambiguous_cutoff": d["ambiguous_cutoff"],
        "p_miv_mean": mc["mean_prob"],
        "var_epistemic": mc["var_epistemic"],
        "bald": mc["bald"],
        "entropy_total": mc["entropy_total"],
        "entropy_aleatoric": mc["entropy_aleatoric"],
        "msp": mc["msp"],
        "mc_samples": serialize_samples(mc["mc_samples"]),
    })
    ex = f"_{extra}" if extra else ""
    fn = f"{STAGE}_{cfg.tag()}{ex}_{split_name}_f{fold}_s{seed}.csv"
    path = os.path.join(C.DIR_PRED, fn)
    out.to_csv(path, index=False)
    print(f"    [pred] {path}  ({len(out)} baris)")
    return path, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--no-preprocessing", action="store_true")
    ap.add_argument("--force", action="store_true", help="latih ulang walau ckpt ada")
    args = ap.parse_args()

    cfg = C.CFG
    if args.backbone:
        cfg.backbone = args.backbone
    if args.no_preprocessing:
        cfg.use_preprocessing = False

    seeds = args.seeds if args.seeds else list(cfg.seeds)
    folds = args.folds if args.folds is not None else list(range(cfg.n_folds))

    df = load_manifest(cfg)
    train_tf, eval_tf = build_transforms(cfg)
    rows = []

    print("=" * 70)
    print(f"STAGE 1 — Mis vs Miv | backbone={cfg.backbone} | tag={cfg.tag()}")
    print(f"seeds={seeds} folds={folds}")
    print("=" * 70)

    for seed in seeds:
        for fold in folds:
            print(f"\n### seed {seed} | fold {fold} ###")
            if exists(STAGE, cfg, fold, seed) and not args.force:
                print("  checkpoint sudah ada -> lewati training, langsung inferensi.")
            set_seed(seed, cfg.deterministic)

            tr, iva, te = split_fold(df, fold, cfg, inner_seed=seed % 10)
            print(f"  train {len(tr)} | inner-val {len(iva)} | test {len(te)}")
            print(f"  distribusi train: {tr[LABEL].value_counts().to_dict()}")

            run = wb.init(cfg, name=f"{STAGE}_{cfg.tag()}_f{fold}_s{seed}",
                          group=f"{STAGE}_{cfg.tag()}", job_type="train",
                          tags=[STAGE, cfg.backbone, f"seed{seed}", f"fold{fold}"])

            model = build_model(cfg, n_out=1)
            if exists(STAGE, cfg, fold, seed) and not args.force:
                from src.checkpoint import load_checkpoint
                model, blob = load_checkpoint(STAGE, cfg, fold, seed)
                hist = blob.get("history", {})
            else:
                print(f"  parameter terlatih: {count_params(model):,}")
                model, hist = train_model(model, tr, iva, cfg, LABEL, 1,
                                          train_tf, eval_tf, seed=seed,
                                          run=run, tag=STAGE)
                save_checkpoint(model, cfg, STAGE, fold, seed, history=hist,
                                val_img_paths=te["img_path"].tolist(),
                                best_metric=hist.get("best_val_loss"), n_out=1)
                if hist.get("train_loss"):
                    plot_history(hist, f"{STAGE}_{cfg.tag()}_f{fold}_s{seed}",
                                 run=run, title=f"Stage 1 fold {fold} seed {seed}")

            # ---------- MC-Dropout: inner-val (kalibrasi) & test (evaluasi) ----------
            for name, part in [("innerval", iva), ("test", te)]:
                loader = make_loader(part, LABEL, eval_tf, cfg, shuffle=False)
                mc = mc_predict(model, loader, cfg, n_out=1)
                _, pred_df = dump_predictions(part, mc, name, cfg, fold, seed)
                if name == "test":
                    m = binary_metrics(pred_df["y_true_mis_miv"], pred_df["p_miv_mean"])
                    m.update({"fold": fold, "seed": seed, "stage": STAGE})
                    rows.append(m)
                    print(f"    TEST  AUC={m['AUC']:.4f}  bACC={m['BalancedAcc']:.4f}  "
                          f"Sens={m['Recall']:.4f}  Spec={m['Specificity']:.4f}")
                    wb.summary(run, {f"test/{k}": v for k, v in m.items()})
            wb.finish(run)

    res = pd.DataFrame(rows)
    if len(res):
        path = os.path.join(C.DIR_RESULT, f"stage1_metrics_{cfg.tag()}.csv")
        res.to_csv(path, index=False)
        print("\n" + "=" * 70)
        print("RINGKASAN STAGE 1 (mean ± sd lintas fold × seed)")
        num = res.select_dtypes("number").drop(columns=["fold", "seed"], errors="ignore")
        print(pd.concat([num.mean().rename("mean"), num.std().rename("sd")],
                        axis=1).round(4).to_string())
        print(f"\n[SAVED] {path}")

    print("\nLANGKAH BERIKUTNYA: python run_03_train_stage2.py")


if __name__ == "__main__":
    main()
