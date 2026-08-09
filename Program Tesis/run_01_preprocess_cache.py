# -*- coding: utf-8 -*-
"""LANGKAH 1 — Prapemrosesan + cache citra ke disk.

Membuat DUA cache supaya ablasi preprocessing bisa apple-to-apple:
    raw  : hanya resize 224 (sesuai paper rujukan)
    proc : black-hat hair removal + Shades of Gray (p=6) + border crop + resize

    python run_01_preprocess_cache.py            # keduanya
    python run_01_preprocess_cache.py --only raw
    python run_01_preprocess_cache.py --sample 12   # cetak contoh visual saja
"""
import argparse
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import config as C
from src.data import load_manifest, cache_path
from src.preprocessing import process_file, resize_only, preprocess_image
from src import wb


def build_cache(df, key, cfg, workers=8):
    todo = []
    for p in df["img_path"].tolist():
        dst = cache_path(p, key, cfg.img_size)
        if not os.path.exists(dst):
            todo.append((p, dst))
    print(f"  cache '{key}': {len(df)-len(todo)} sudah ada, {len(todo)} akan dibuat")
    if not todo:
        return 0

    def work(t):
        src, dst = t
        try:
            if key == "raw":
                return resize_only(src, dst, cfg.img_size)
            return process_file(src, dst, cfg.img_size,
                                do_hair=True, do_cc=True, do_crop=True, sog_p=6)
        except Exception as e:
            print(f"    [!] gagal {src}: {e}")
            return False

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(ex.map(work, todo), total=len(todo), desc=f"  {key}"):
            ok += bool(r)
    print(f"  cache '{key}': {ok}/{len(todo)} berhasil")
    return ok


def sample_figure(df, cfg, n=8):
    """Figur perbandingan sebelum/sesudah preprocessing untuk Bab 3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = df.sample(min(n, len(df)), random_state=7)
    fig, axes = plt.subplots(2, len(sub), figsize=(2.4 * len(sub), 5.2))
    axes = np.atleast_2d(axes)
    for j, (_, r) in enumerate(sub.iterrows()):
        img = cv2.imread(r["img_path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        proc = preprocess_image(img, True, True, True, 6)
        for i, im in enumerate([img, proc]):
            axes[i, j].imshow(cv2.cvtColor(cv2.resize(im, (224, 224)),
                                           cv2.COLOR_BGR2RGB))
            axes[i, j].axis("off")
        axes[0, j].set_title(r["source"], fontsize=8)
    axes[0, 0].set_ylabel("Asli"); axes[1, 0].set_ylabel("Diproses")
    fig.suptitle("Prapemrosesan: hair removal + Shades of Gray + border crop")
    os.makedirs(C.DIR_FIG, exist_ok=True)
    path = os.path.join(C.DIR_FIG, "fig_preprocessing_examples.png")
    fig.savefig(path, dpi=250, bbox_inches="tight"); plt.close(fig)
    print(f"  [fig] {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["raw", "proc"], default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample", type=int, default=8)
    args = ap.parse_args()

    cfg = C.CFG
    df = load_manifest(cfg)
    print(f"Manifest: {len(df)} gambar")

    run = wb.init(cfg, name="preprocess_cache", group="data", job_type="preprocess")

    keys = [args.only] if args.only else ["raw", "proc"]
    for k in keys:
        print(f"\n== Membangun cache '{k}' ==")
        build_cache(df, k, cfg, args.workers)

    if args.sample:
        p = sample_figure(df, cfg, args.sample)
        wb.log_image(run, p, "preprocessing_examples")
    wb.finish(run)

    print("\nLANGKAH BERIKUTNYA: python run_02_train_stage1.py")


if __name__ == "__main__":
    main()
