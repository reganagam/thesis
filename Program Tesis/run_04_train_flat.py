# -*- coding: utf-8 -*-
"""LANGKAH 4 — Latih BASELINE FLAT MULTICLASS (Skenario A).

Satu CNN, 3 kelas {Mis, Miv<0.8, Miv>=0.8}. Ini replikasi jalur SUPERVISED
paper Dominguez-Morales dkk. sekaligus baseline pembanding.

MC-Dropout juga dijalankan di sini supaya Skenario A-rej (flat + penolakan)
bisa dievaluasi. Tanpa A-rej, penguji bisa berargumen bahwa seluruh kenaikan
Skenario C berasal dari penolakan semata, bukan dari struktur hierarkis.

    python run_04_train_flat.py
"""
import argparse
import os
import numpy as np
import pandas as pd

import config as C
from src.data import load_manifest, split_fold, build_transforms, make_loader
from src.models import build_model
from src.engine import train_model, set_seed
from src.uncertainty import mc_predict, serialize_samples
from src.metrics import multiclass_metrics
from src.checkpoint import save_checkpoint, load_checkpoint, exists
from src.plots import plot_history
from src import wb

STAGE = "flat"
LABEL = "multiclass"


def dump_predictions(part, mc, split_name, cfg, fold, seed):
    d = part.reset_index(drop=True).iloc[mc["row_index"]].reset_index(drop=True)
    P = mc["mean_prob"]                    # (N, 3)
    out = pd.DataFrame({
        "img_path": d["img_path"], "image_id": d["image_id"],
        "patient_id": d["patient_id"], "source": d["source"],
        "fold": fold, "seed": seed, "split": split_name,
        "stage": STAGE, "backbone": cfg.backbone, "tag": cfg.tag(),
        "y_true_3class": d["multiclass"].astype(int),
        "y_true_mis_miv": d["is_invasive"].astype(int),
        "y_true_bt": d["depth_ge_cut"],
        "ambiguous_cutoff": d["ambiguous_cutoff"],
        "p_c0": P[:, 0], "p_c1": P[:, 1], "p_c2": P[:, 2],
        "y_pred_3class": P.argmax(1),
        "var_epistemic": mc["var_epistemic"],
        "bald": mc["bald"],
        "entropy_total": mc["entropy_total"],
        "entropy_aleatoric": mc["entropy_aleatoric"],
        "msp": mc["msp"],
        "mc_samples": serialize_samples(mc["mc_samples"]),
    })
    fn = f"{STAGE}_{cfg.tag()}_{split_name}_f{fold}_s{seed}.csv"
    path = os.path.join(C.DIR_PRED, fn)
    out.to_csv(path, index=False)
    print(f"    [pred] {path}  ({len(out)} baris)")
    return path, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = C.CFG
    if args.backbone:
        cfg.backbone = args.backbone
    seeds = args.seeds if args.seeds else list(cfg.seeds)
    folds = args.folds if args.folds is not None else list(range(cfg.n_folds))

    df = load_manifest(cfg)
    train_tf, eval_tf = build_transforms(cfg)
    rows = []

    print("=" * 70)
    print(f"SKENARIO A — FLAT MULTICLASS (3 kelas) | tag={cfg.tag()}")
    print("=" * 70)

    for seed in seeds:
        for fold in folds:
            print(f"\n### seed {seed} | fold {fold} ###")
            set_seed(seed, cfg.deterministic)
            tr, iva, te = split_fold(df, fold, cfg, inner_seed=seed % 10)
            print(f"  train {len(tr)} | inner-val {len(iva)} | test {len(te)}")

            run = wb.init(cfg, name=f"{STAGE}_{cfg.tag()}_f{fold}_s{seed}",
                          group=f"{STAGE}_{cfg.tag()}", job_type="train",
                          tags=[STAGE, cfg.backbone, f"seed{seed}", f"fold{fold}"])

            model = build_model(cfg, n_out=3)
            if exists(STAGE, cfg, fold, seed) and not args.force:
                print("  checkpoint sudah ada -> muat.")
                model, blob = load_checkpoint(STAGE, cfg, fold, seed)
                hist = blob.get("history", {})
            else:
                model, hist = train_model(model, tr, iva, cfg, LABEL, 3,
                                          train_tf, eval_tf, seed=seed,
                                          run=run, tag=STAGE)
                save_checkpoint(model, cfg, STAGE, fold, seed, history=hist,
                                val_img_paths=te["img_path"].tolist(),
                                best_metric=hist.get("best_val_loss"), n_out=3)
                if hist.get("train_loss"):
                    plot_history(hist, f"{STAGE}_{cfg.tag()}_f{fold}_s{seed}",
                                 run=run, title=f"Flat multiclass fold {fold}")

            for name, part in [("innerval", iva), ("test", te)]:
                loader = make_loader(part, LABEL, eval_tf, cfg, shuffle=False)
                mc = mc_predict(model, loader, cfg, n_out=3)
                _, pdf = dump_predictions(part, mc, name, cfg, fold, seed)
                if name == "test":
                    m = multiclass_metrics(pdf["y_true_3class"], pdf["y_pred_3class"],
                                           pdf[["p_c0", "p_c1", "p_c2"]].values, 3)
                    m.update({"fold": fold, "seed": seed})
                    rows.append(m)
                    print(f"    TEST bACC={m['BalancedAcc']:.4f}  "
                          f"macroF1={m['MacroF1']:.4f}  AUC={m.get('AUC_ovr', float('nan')):.4f}")
                    wb.summary(run, {f"test/{k}": v for k, v in m.items()})
            wb.finish(run)

    if rows:
        res = pd.DataFrame(rows)
        path = os.path.join(C.DIR_RESULT, f"flat_metrics_{cfg.tag()}.csv")
        res.to_csv(path, index=False)
        num = res.select_dtypes("number").drop(columns=["fold", "seed"], errors="ignore")
        print("\n" + "=" * 70)
        print("RINGKASAN SKENARIO A")
        print(pd.concat([num.mean().rename("mean"), num.std().rename("sd")],
                        axis=1).round(4).to_string())
        print(f"\n[SAVED] {path}")

    print("\nLANGKAH BERIKUTNYA: python run_05_external_kawahara.py")


if __name__ == "__main__":
    main()
