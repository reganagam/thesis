# -*- coding: utf-8 -*-
"""Resolver path gambar untuk 4 sumber dataset.

Layout folder tiap sumber bisa berbeda-beda. Strategi:
  1. coba path langsung (kandidat yang paling mungkin),
  2. kalau gagal, pakai INDEKS global: seluruh DATASET_ROOT di-scan sekali,
     lalu dicocokkan berdasarkan nama file (case-insensitive).
Ini membuat pipeline tahan terhadap perbedaan struktur folder.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional

import config as C

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG")


@lru_cache(maxsize=1)
def _global_index() -> Dict[str, str]:
    """{nama_file_lowercase: path_absolut} untuk seluruh DATASET_ROOT."""
    idx: Dict[str, str] = {}
    for root, _dirs, files in os.walk(C.DATASET_ROOT):
        for f in files:
            if f.lower().endswith(tuple(e.lower() for e in IMG_EXT)):
                idx.setdefault(f.lower(), os.path.join(root, f))
    print(f"[paths] indeks global: {len(idx)} berkas gambar di {C.DATASET_ROOT}")
    return idx


def _try(cands: List[str]) -> Optional[str]:
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def _by_name(fname: str) -> Optional[str]:
    return _global_index().get(os.path.basename(fname).lower())


def _with_ext(base: str, folder: str) -> List[str]:
    return [os.path.join(folder, base + e) for e in IMG_EXT]


# --------------------------------------------------------------------------- #
def resolve_vruh(filepath: str) -> Optional[str]:
    """CSV VRUH menyimpan path relatif, mis. 'VRUH\\data\\Tis\\<pid>\\<img>.jpg'."""
    fp = str(filepath).replace("\\", os.sep).replace("/", os.sep)
    cands = [
        os.path.join(C.DATASET_ROOT, fp),
        os.path.join(C.SOURCE_DIRS["VRUH"], fp),
        fp,
    ]
    return _try(cands) or _by_name(fp)


def resolve_isic(isic_id: str) -> Optional[str]:
    d = C.SOURCE_DIRS["ISIC"]
    cands = (_with_ext(isic_id, d)
             + _with_ext(isic_id, os.path.join(d, "labeled"))
             + _with_ext(isic_id, os.path.join(d, "images")))
    return _try(cands) or _by_name(isic_id + ".jpg")


def resolve_polesie(image_id: str) -> Optional[str]:
    d = C.SOURCE_DIRS["Polesie"]
    base = os.path.splitext(str(image_id))[0]
    cands = ([os.path.join(d, str(image_id))]
             + _with_ext(base, d)
             + _with_ext(base, os.path.join(d, "images")))
    return _try(cands) or _by_name(str(image_id))


def resolve_kawahara(derm_rel: str) -> Optional[str]:
    """Kolom 'derm' berisi path relatif seperti 'NEL/Nel026.jpg'."""
    rel = str(derm_rel).replace("/", os.sep).replace("\\", os.sep)
    d = C.SOURCE_DIRS["Kawahara"]
    cands = [
        os.path.join(d, "images", rel),
        os.path.join(d, rel),
        os.path.join(d, "release_v0", "images", rel),
        os.path.join(C.DATASET_ROOT, rel),
    ]
    return _try(cands) or _by_name(rel)


def report_missing(df, col="img_path", label=""):
    n_missing = int(df[col].isna().sum())
    if n_missing:
        print(f"  [!] {label}: {n_missing} gambar TIDAK ditemukan di disk (dibuang).")
    return n_missing
