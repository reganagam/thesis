# -*- coding: utf-8 -*-
"""Dataset & DataLoader. Albumentations 2.x."""
from __future__ import annotations

import os
import hashlib
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config as C


# --------------------------------------------------------------------------- #
def cache_path(img_path: str, key: str, size: int) -> str:
    """Path cache deterministik berdasar hash path asli."""
    h = hashlib.md5(img_path.encode("utf-8")).hexdigest()
    return os.path.join(C.DIR_CACHE, f"{key}_{size}", h[:2], h + ".jpg")


def build_transforms(cfg):
    size = cfg.img_size
    train_tf = A.Compose([
        A.Resize(size, size),
        A.RandomRotate90(p=0.5),          # 90/180/270 -- sesuai paper rujukan
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=C.IMAGENET_MEAN, std=C.IMAGENET_STD),
        ToTensorV2(),
    ])
    eval_tf = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=C.IMAGENET_MEAN, std=C.IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_tf, eval_tf


class LesionDataset(Dataset):
    """label_col: 'is_invasive' | 'depth_ge_cut' | 'multiclass' | None."""

    def __init__(self, df: pd.DataFrame, label_col, tf, cfg, use_cache=True):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.tf = tf
        self.cfg = cfg
        self.use_cache = use_cache
        self.key = cfg.cache_key()

    def __len__(self):
        return len(self.df)

    def _read(self, row) -> np.ndarray:
        if self.use_cache:
            cp = cache_path(row["img_path"], self.key, self.cfg.img_size)
            if os.path.exists(cp):
                img = cv2.imread(cp, cv2.IMREAD_COLOR)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.imread(row["img_path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row["img_path"])
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = self.tf(image=self._read(row))["image"]
        if self.label_col is None:
            y = torch.tensor(-1, dtype=torch.long)
        else:
            v = row[self.label_col]
            y = torch.tensor(float(v) if pd.notna(v) else -1.0, dtype=torch.float32)
        return img, y, i


def make_loader(df, label_col, tf, cfg, shuffle=False, bs=None):
    ds = LesionDataset(df, label_col, tf, cfg)
    return DataLoader(ds, batch_size=bs or cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=False)


# --------------------------------------------------------------------------- #
def load_manifest(cfg=None) -> pd.DataFrame:
    if not os.path.exists(C.MANIFEST):
        raise FileNotFoundError(
            f"{C.MANIFEST} belum ada. Jalankan run_00_build_manifest.py dulu.")
    df = pd.read_csv(C.MANIFEST)
    df["depth_ge_cut"] = pd.to_numeric(df["depth_ge_cut"], errors="coerce").astype("Int64")
    for c in ("is_external", "breslow_is_range", "ambiguous_cutoff"):
        df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def split_fold(df: pd.DataFrame, fold: int, cfg, inner_seed: int = 0):
    """Kembalikan (train, inner_val, test) untuk satu fold.

    inner_val dipakai untuk early stopping DAN kalibrasi ambang gatekeeper,
    supaya ambang tidak pernah "melihat" test fold (anti kebocoran).
    Split inner juga patient-level.
    """
    internal = df[~df["is_external"]]
    test = internal[internal["fold"] == fold].copy()
    pool = internal[internal["fold"] != fold].copy()

    rng = np.random.RandomState(1000 + inner_seed * 10 + fold)
    pats = pool["patient_id"].unique()
    rng.shuffle(pats)
    n_val = max(1, int(len(pats) * cfg.inner_val_frac))
    val_p = set(pats[:n_val])

    inner_val = pool[pool["patient_id"].isin(val_p)].copy()
    train = pool[~pool["patient_id"].isin(val_p)].copy()

    assert not (set(train.patient_id) & set(inner_val.patient_id))
    assert not (set(train.patient_id) & set(test.patient_id))
    return train, inner_val, test


def external_set(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_external"]].copy()
