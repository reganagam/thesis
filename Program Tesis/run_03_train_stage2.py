# -*- coding: utf-8 -*-
"""LANGKAH 3 — Latih STAGE 2 (<0.8 mm vs >=0.8 mm).

DUA MODE PELATIHAN (ablasi penting):
  gt  : dilatih pada Miv menurut LABEL ASLI (naif)
  oof : dilatih pada sampel yang DIPREDIKSI Miv oleh Stage 1 secara
        out-of-fold. Ini menutup mismatch train/inference — saat inferensi
        Stage 2 menerima Miv hasil prediksi, bukan Miv ground truth.

Mode 'oof' membutuhkan checkpoint Stage 1 (jalankan run_02 dulu).

    python run_03_train_stage2.py --mode oof
    python run_03_train_stage2.py --mode gt
    python run_03_train_stage2.py --mode both
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
from src.uncertainty import mc_predict, serialize_samples
from src.metrics import binary_metrics
from src.checkpoint import save_checkpoint, load_checkpoint, exists
from src.plots import plot_history
from src import wb

STAGE = "stage2"
LABEL = "depth_ge_cut"


def stage1_predict_df(model, part, cfg, eval_tf):
    """Prediksi Stage 1 pada satu subset (deterministik, cepat)."""
    from src.uncertainty import deterministic_predict
    loader = make_loader(part, "is_invasive", eval_tf, cfg, shuffle=False)
    d = deterministic_predict(model, loader, cfg, n_out=1)
    out = part.reset_index(drop=True).iloc[d["row_index"]].reset_index(drop=True)
    out["p_miv"] = d["mean_prob"]
    return out


def build_stage2_train(tr, cfg, fold, seed, eval_tf, mode):
    """Kembalikan subset latih Stage 2 sesuai mode."""
    if mode == "gt":
        sub = tr[tr["is_invasive"] == 1].copy()
        print(f"  mode 'gt' : {len(sub)} sampel Miv (label asli)")
        return sub

    # mode 'oof': pakai Stage 1 fold ini untuk menyaring
    try:
        s1, _ = load_checkpoint("stage1", cfg, fold, seed)
    except FileNotFoundError:
        print("  [!] checkpoint Stage 1 tidak ada -> fallback ke mode 'gt'")
        return tr[tr["is_invasive"] == 1].copy()

    pr = stage1_predict_df(s1, tr, cfg, eval_tf)
    sub = pr[pr["p_miv"] >= 0.5].copy()
    # sampel yang diprediksi Miv tapi sebenarnya Mis: tidak punya label depth.
    # Dibuang dari LOSS tetapi keberadaannya sudah tercermin di distribusi.
    n_fp = int((sub["is_invasive"] == 0).sum())
    sub = sub[sub["is_invasive"] == 1].copy()
    print(f"  mode 'oof': {len(sub)} sampel diprediksi Miv & benar Miv "
          f"({n_fp} false-positive Stage 1 dibuang karena tak berlabel depth)")
    del s1
    torch.cuda.empty_cache()
    return sub


def dump_predictions(part, mc, split_name, cfg, fold, seed, mode):
    d = part.reset_index(drop=True).iloc[mc["row_index"]].reset_index(drop=True)
    out = pd.DataFrame({
        "img_path": d["img_path"], "image_id": d["image_id"],
        "patient_id": d["patient_id"], "source": d["source"],
        "fold": fold, "seed": seed, "split": split_name,
        "stage": STAGE, "s2_mode": mode, "backbone": cfg.backbone, "tag": cfg.tag(),
        "y_true_mis_miv": d["is_invasive"].astype(int),
        "y_true_bt": d["depth_ge_cut"],
        "y_true_3class": d["multiclass"].astype(int),
        "ambiguous_cutoff": d["ambiguous_cutoff"],
        "p_bt_mean": mc["mean_prob"],
        "var_epistemic_s2": mc["var_epistemic"],
        "bald_s2": mc["bald"],
        "entropy_total_s2": mc["entropy_total"],
        "msp_s2": mc["msp"],
        "mc_samples_s2": serialize_samples(mc["mc_samples"]),
    })
    fn = f"{STAGE}_{cfg.tag()}_{mode}_{split_name}_f{fold}_s{seed}.csv"
    path = os.path.join(C.DIR_PRED, fn)
    out.to_csv(path, index=False)
    print(f"    [pred] {path}  ({len(out)} baris)")
    return path, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gt", "oof", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = C.CFG
    if args.backbone:
        cfg.backbone = args.backbone
    modes = ["gt", "oof"] if args.mode == "both" else [args.mode]
    seeds = args.seeds if args.seeds else list(cfg.seeds)
    folds = args.folds if args.folds is not None else list(range(cfg.n_folds))

    df = load_manifest(cfg)
    train_tf, eval_tf = build_transforms(cfg)
    rows = []

    print("=" * 70)
    print(f"STAGE 2 — <0.8 vs >=0.8 mm | modes={modes} | tag={cfg.tag()}")
    print("=" * 70)

    for mode in modes:
        for seed in seeds:
            for fold in folds:
                print(f"\n### mode {mode} | seed {seed} | fold {fold} ###")
                set_seed(seed, cfg.deterministic)
                tr, iva, te = split_fold(df, fold, cfg, inner_seed=seed % 10)

                s2_tr = build_stage2_train(tr, cfg, fold, seed, eval_tf, mode)
                s2_iva = iva[iva["is_invasive"] == 1].copy()
                # TEST: semua sampel (perakitan hierarkis dilakukan di run_06)
                s2_te = te.copy()

                if len(s2_tr) < 10 or s2_tr[LABEL].nunique() < 2:
                    print("  [!] data latih Stage 2 tidak memadai -> dilewati.")
                    continue
                print(f"  train {len(s2_tr)} | inner-val {len(s2_iva)} | test {len(s2_te)}")

                run = wb.init(cfg, name=f"{STAGE}_{mode}_{cfg.tag()}_f{fold}_s{seed}",
                              group=f"{STAGE}_{mode}_{cfg.tag()}", job_type="train",
                              tags=[STAGE, mode, cfg.backbone, f"seed{seed}", f"fold{fold}"])

                model = build_model(cfg, n_out=1)
                if exists(STAGE, cfg, fold, seed, extra=mode) and not args.force:
                    print("  checkpoint sudah ada -> muat.")
                    model, blob = load_checkpoint(STAGE, cfg, fold, seed, extra=mode)
                    hist = blob.get("history", {})
                else:
                    model, hist = train_model(model, s2_tr, s2_iva, cfg, LABEL, 1,
                                              train_tf, eval_tf, seed=seed,
                                              run=run, tag=f"{STAGE}_{mode}")
                    save_checkpoint(model, cfg, STAGE, fold, seed, history=hist,
                                    val_img_paths=s2_te["img_path"].tolist(),
                                    best_metric=hist.get("best_val_loss"),
                                    extra=mode, n_out=1)
                    if hist.get("train_loss"):
                        plot_history(hist, f"{STAGE}_{mode}_{cfg.tag()}_f{fold}_s{seed}",
                                     run=run, title=f"Stage 2 ({mode}) fold {fold}")

                for name, part in [("innerval", s2_iva), ("test", s2_te)]:
                    if len(part) == 0:
                        continue
                    loader = make_loader(part, LABEL, eval_tf, cfg, shuffle=False)
                    mc = mc_predict(model, loader, cfg, n_out=1)
                    _, pdf = dump_predictions(part, mc, name, cfg, fold, seed, mode)
                    if name == "test":
                        ev = pdf[pdf["y_true_bt"].notna()]
                        if len(ev) and ev["y_true_bt"].nunique() > 1:
                            m = binary_metrics(ev["y_true_bt"].astype(int),
                                               ev["p_bt_mean"])
                            m.update({"fold": fold, "seed": seed, "mode": mode})
                            rows.append(m)
                            print(f"    TEST(Miv sejati) AUC={m['AUC']:.4f}  "
                                  f"bACC={m['BalancedAcc']:.4f}")
                            wb.summary(run, {f"test/{k}": v for k, v in m.items()})
                wb.finish(run)

    if rows:
        res = pd.DataFrame(rows)
        path = os.path.join(C.DIR_RESULT, f"stage2_metrics_{cfg.tag()}.csv")
        res.to_csv(path, index=False)
        print("\n" + "=" * 70)
        print("RINGKASAN STAGE 2 per mode")
        print(res.groupby("mode")[["AUC", "BalancedAcc", "F1", "Recall",
                                   "Specificity"]].agg(["mean", "std"]).round(4).to_string())
        print(f"\n[SAVED] {path}")

    print("\nLANGKAH BERIKUTNYA: python run_04_train_flat.py")


if __name__ == "__main__":
    main()
