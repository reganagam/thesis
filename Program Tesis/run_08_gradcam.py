# -*- coding: utf-8 -*-
"""LANGKAH 8 — Grad-CAM++ (Explainable AI).

Hanya butuh checkpoint + gambar. Bisa dijalankan KAPAN SAJA, berkali-kali,
tanpa training ulang.

Menghasilkan tiga grid yang berguna untuk sidang:
  1. Prediksi BENAR berkeyakinan tinggi   -> model melihat lesi
  2. Prediksi SALAH                        -> di mana model tersesat
  3. Kasus DITOLAK GATEKEEPER              -> apakah heatmap-nya memang kacau?
     (kalau ya, ini bukti visual kuat bahwa gatekeeper menangkap ambiguitas nyata)

    python run_08_gradcam.py
    python run_08_gradcam.py --fold 0 --seed 42 --n 8
"""
import argparse
import glob
import os
import cv2
import numpy as np
import pandas as pd
import torch

import config as C
from src.data import build_transforms
from src.models import target_layer
from src.checkpoint import load_checkpoint
from src.gradcam import GradCAMpp, render_grid
from src.selective import calibrate_threshold
from src import wb


def load_tensor(path, tf, device):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = tf(image=rgb)["image"].unsqueeze(0).to(device)
    return x, cv2.resize(rgb, (C.CFG.img_size, C.CFG.img_size))


def make_grid(cam_obj, rows, tf, device, fname, title, n_out=1, run=None):
    items = []
    for _, r in rows.iterrows():
        x, rgb = load_tensor(r["img_path"], tf, device)
        if x is None:
            continue
        cam = cam_obj(x, n_out=n_out)[0]
        cap = (f"{r['source']} | GT={C.CLASS_NAMES_3[int(r['y_true_3class'])]}\n"
               f"p(Miv)={r.get('p_miv_mean', float('nan')):.2f} "
               f"var={r.get('var_epistemic', float('nan')):.4f}")
        items.append((rgb, cam, cap))
    return render_grid(items, fname, ncol=min(4, max(1, len(items))),
                       run=run, suptitle=title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = C.CFG
    tag = args.tag or cfg.tag()
    seed = args.seed if args.seed is not None else list(cfg.seeds)[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, eval_tf = build_transforms(cfg)

    f = os.path.join(C.DIR_PRED, f"stage1_{tag}_test_f{args.fold}_s{seed}.csv")
    if not os.path.exists(f):
        raise SystemExit(f"CSV prediksi tidak ada: {f}\nJalankan run_02 dulu.")
    d = pd.read_csv(f)
    d["pred_miv"] = (d["p_miv_mean"] >= 0.5).astype(int)
    d["correct"] = (d["pred_miv"] == d["y_true_mis_miv"]).astype(int)

    score_col = "bald" if cfg.epistemic_measure == "bald" else "var_epistemic"
    iv = os.path.join(C.DIR_PRED, f"stage1_{tag}_innerval_f{args.fold}_s{seed}.csv")
    thr = (calibrate_threshold(pd.read_csv(iv)[score_col].values,
                               cfg.reject_percentile) if os.path.exists(iv)
           else float(np.percentile(d[score_col], 100 - cfg.reject_percentile)))
    print(f"Ambang gatekeeper (dari inner-val) = {thr:.6f}")

    run = wb.init(cfg, name=f"gradcam_{tag}_f{args.fold}_s{seed}",
                  group="xai", job_type="viz")

    model, _ = load_checkpoint("stage1", cfg, args.fold, seed, device=device)
    cam = GradCAMpp(model, target_layer(model, cfg))

    conf = d[(d.correct == 1) & (d[score_col] <= thr)].nsmallest(args.n, score_col)
    make_grid(cam, conf, eval_tf, device,
              f"fig_gradcam_stage1_confident_correct_f{args.fold}.png",
              "Grad-CAM++ Stage 1 — prediksi BENAR & yakin", run=run)

    wrong = d[d.correct == 0].nlargest(args.n, score_col)
    make_grid(cam, wrong, eval_tf, device,
              f"fig_gradcam_stage1_errors_f{args.fold}.png",
              "Grad-CAM++ Stage 1 — prediksi SALAH", run=run)

    rej = d[d[score_col] > thr].nlargest(args.n, score_col)
    make_grid(cam, rej, eval_tf, device,
              f"fig_gradcam_stage1_rejected_f{args.fold}.png",
              "Grad-CAM++ Stage 1 — kasus DITOLAK gatekeeper "
              "(dirujuk ke dermatolog)", run=run)
    cam.remove(); del model; torch.cuda.empty_cache()

    # ---- Stage 2 ----
    for mode in ["oof", "gt"]:
        f2 = os.path.join(C.DIR_PRED,
                          f"stage2_{tag}_{mode}_test_f{args.fold}_s{seed}.csv")
        if not os.path.exists(f2):
            continue
        try:
            m2, _ = load_checkpoint("stage2", cfg, args.fold, seed,
                                    extra=mode, device=device)
        except FileNotFoundError:
            continue
        d2 = pd.read_csv(f2)
        d2 = d2[d2["y_true_bt"].notna()]
        if d2.empty:
            continue
        d2["correct"] = ((d2["p_bt_mean"] >= .5).astype(int)
                         == d2["y_true_bt"].astype(int)).astype(int)
        cam2 = GradCAMpp(m2, target_layer(m2, cfg))
        make_grid(cam2, d2[d2.correct == 1].head(args.n), eval_tf, device,
                  f"fig_gradcam_stage2_{mode}_correct_f{args.fold}.png",
                  f"Grad-CAM++ Stage 2 ({mode}) — prediksi benar", run=run)
        cam2.remove(); del m2; torch.cuda.empty_cache()

    wb.finish(run)
    print("\nSELESAI. Semua figur di", C.DIR_FIG)


if __name__ == "__main__":
    main()
