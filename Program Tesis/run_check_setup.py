# -*- coding: utf-8 -*-
"""PEMERIKSAAN AWAL — jalankan SEBELUM apa pun.

Memverifikasi: GPU, versi library, keberadaan CSV metadata, dan apakah
gambar dari tiap sumber benar-benar bisa ditemukan di disk.

    python run_check_setup.py
"""
import os
import sys
import importlib

import config as C

OK, BAD = "  [OK] ", "  [!!] "


def check_libs():
    print("\n== LIBRARY ==")
    need = ["torch", "torchvision", "albumentations", "cv2", "sklearn",
            "pandas", "numpy", "matplotlib", "seaborn", "scipy", "tqdm"]
    ok = True
    for n in need:
        try:
            m = importlib.import_module(n)
            print(OK + f"{n:16s} {getattr(m, '__version__', '?')}")
        except Exception as e:
            print(BAD + f"{n:16s} TIDAK ADA ({e})"); ok = False
    try:
        import wandb
        print(OK + f"{'wandb':16s} {wandb.__version__}")
    except Exception:
        print("  [--] wandb tidak ada -> logging dilewati (tidak fatal)")
    return ok


def check_gpu():
    print("\n== GPU ==")
    import torch
    print(f"  torch {torch.__version__}")
    if not torch.cuda.is_available():
        print(BAD + "CUDA TIDAK TERSEDIA — training akan sangat lambat di CPU.")
        return False
    cc = torch.cuda.get_device_capability()
    print(OK + f"{torch.cuda.get_device_name(0)}  compute capability {cc}")
    print(OK + f"VRAM {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    if cc[0] >= 8 and cc[1] >= 9:
        print(OK + "Ada Lovelace (sm_89) terdeteksi — butuh CUDA 11.8+/PyTorch 2.x. Aman.")
    try:
        x = torch.randn(8, 3, 224, 224, device="cuda")
        with torch.amp.autocast("cuda"):
            y = (x * 2).sum()
        print(OK + f"uji AMP berhasil ({float(y):.1f})")
    except Exception as e:
        print(BAD + f"uji GPU gagal: {e}"); return False
    return True


def check_csv():
    print("\n== CSV METADATA ==")
    ok = True
    for name, p in [("VRUH", C.VRUH_CSV), ("ISIC", C.ISIC_CSV),
                    ("Polesie", C.POLESIE_CSV), ("Kawahara", C.KAWAHARA_CSV)]:
        if os.path.exists(p):
            import pandas as pd
            print(OK + f"{name:9s} {len(pd.read_csv(p)):5d} baris  {p}")
        else:
            print(BAD + f"{name:9s} TIDAK DITEMUKAN: {p}"); ok = False
    return ok


def check_images():
    print("\n== RESOLUSI PATH GAMBAR (sampel 5 per sumber) ==")
    import pandas as pd
    from src import paths as P
    ok = True

    def probe(label, series, fn):
        hits = sum(1 for v in series if fn(v))
        mark = OK if hits == len(series) else BAD
        print(mark + f"{label:9s} {hits}/{len(series)} ditemukan")
        return hits == len(series)

    if os.path.exists(C.VRUH_CSV):
        d = pd.read_csv(C.VRUH_CSV).head(5)
        ok &= probe("VRUH", d["filepath"], lambda v: P.resolve_vruh(v) is not None)
    if os.path.exists(C.ISIC_CSV):
        d = pd.read_csv(C.ISIC_CSV).head(5)
        ok &= probe("ISIC", d["isic_id"], lambda v: P.resolve_isic(v) is not None)
    if os.path.exists(C.POLESIE_CSV):
        d = pd.read_csv(C.POLESIE_CSV).head(5)
        ok &= probe("Polesie", d["image_id"], lambda v: P.resolve_polesie(v) is not None)
    if os.path.exists(C.KAWAHARA_CSV):
        d = pd.read_csv(C.KAWAHARA_CSV).head(5)
        ok &= probe("Kawahara", d["derm"], lambda v: P.resolve_kawahara(v) is not None)
    if not ok:
        print("\n  Petunjuk: resolver otomatis mengindeks seluruh DATASET_ROOT")
        print("  berdasarkan NAMA FILE. Kalau masih gagal, periksa DATASET_ROOT")
        print("  dan SOURCE_DIRS di config.py.")
    return ok


def main():
    print("=" * 70)
    print("PEMERIKSAAN SETUP")
    print("=" * 70)
    print(f"  DATASET_ROOT = {C.DATASET_ROOT}")
    print(f"  PROJECT_ROOT = {C.PROJECT_ROOT}")
    r = [check_libs(), check_gpu(), check_csv(), check_images()]
    print("\n" + "=" * 70)
    if all(r):
        print("SEMUA SIAP. Jalankan: python run_00_build_manifest.py")
    else:
        print("ADA MASALAH DI ATAS — perbaiki dulu sebelum lanjut.")
    print("=" * 70)


if __name__ == "__main__":
    main()
