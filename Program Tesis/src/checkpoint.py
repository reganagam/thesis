# -*- coding: utf-8 -*-
"""Checkpoint BERMETADATA.

Menyimpan state_dict SAJA tidak cukup: history training (kurva loss/akurasi
per epoch) adalah satu-satunya artefak yang HILANG PERMANEN kalau tidak
disimpan -- tidak bisa diregenerasi dari bobot. Fold & daftar sampel validasi
juga disimpan agar visualisasi berbulan-bulan kemudian tetap valid.
"""
from __future__ import annotations

import os
import json
import torch

import config as C


def ckpt_name(stage: str, cfg, fold: int, seed: int, extra: str = "") -> str:
    ex = f"_{extra}" if extra else ""
    return f"{stage}_{cfg.tag()}{ex}_f{fold}_s{seed}.pt"


def ckpt_path(stage: str, cfg, fold: int, seed: int, extra: str = "") -> str:
    return os.path.join(C.DIR_CKPT, ckpt_name(stage, cfg, fold, seed, extra))


def save_checkpoint(model, cfg, stage: str, fold: int, seed: int,
                    history=None, val_img_paths=None, best_metric=None,
                    extra: str = "", n_out: int = 1) -> str:
    path = ckpt_path(stage, cfg, fold, seed, extra)
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "stage": stage,
        "n_out": n_out,
        "fold": fold,
        "seed": seed,
        "history": history or {},
        "val_img_paths": list(val_img_paths) if val_img_paths is not None else [],
        "best_metric": best_metric,
        "torch_version": torch.__version__,
    }, path)
    return path


def load_checkpoint(stage: str, cfg, fold: int, seed: int, extra: str = "",
                    device=None, build=True):
    """Return (model_or_None, blob). build=False -> hanya metadata."""
    path = ckpt_path(stage, cfg, fold, seed, extra)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    if not build:
        return None, blob
    from src.models import build_model
    model = build_model(cfg, n_out=blob.get("n_out", 1))
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob


def exists(stage: str, cfg, fold: int, seed: int, extra: str = "") -> bool:
    return os.path.exists(ckpt_path(stage, cfg, fold, seed, extra))


def save_json(obj, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
