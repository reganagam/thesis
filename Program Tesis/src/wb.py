# -*- coding: utf-8 -*-
"""Pembungkus wandb yang TIDAK PERNAH menggagalkan training.

Kalau wandb tidak terpasang / offline / gagal login, semua fungsi jadi no-op.
"""
from __future__ import annotations

from typing import Optional, List, Any, Dict

import config as C

try:
    import wandb
    _HAS = True
except Exception:
    wandb = None
    _HAS = False


def available() -> bool:
    return _HAS


def init(cfg, name: str, group: str, job_type: str = "train",
         tags: Optional[List[str]] = None):
    if not _HAS or cfg.wandb_mode == "disabled":
        return None
    try:
        return wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                          name=name, group=group, job_type=job_type,
                          tags=tags or [], config=cfg.to_dict(),
                          mode=cfg.wandb_mode, reinit=True)
    except Exception as e:
        print(f"  [wandb] init gagal ({e}) -> dilewati.")
        return None


def log(run, payload: Dict[str, Any], step: Optional[int] = None):
    if run is None:
        return
    try:
        run.log(payload, step=step)
    except Exception:
        pass


def log_table(run, df, key: str):
    if run is None or not _HAS:
        return
    try:
        run.log({key: wandb.Table(dataframe=df)})
    except Exception:
        pass


def log_image(run, path: str, key: str, caption: str = ""):
    if run is None or not _HAS:
        return
    try:
        run.log({key: wandb.Image(path, caption=caption)})
    except Exception:
        pass


def log_artifact(run, path: str, name: str, type_: str = "dataset"):
    if run is None or not _HAS:
        return
    try:
        art = wandb.Artifact(name=name, type=type_)
        art.add_file(path)
        run.log_artifact(art)
    except Exception:
        pass


def summary(run, payload: Dict[str, Any]):
    if run is None:
        return
    try:
        for k, v in payload.items():
            run.summary[k] = v
    except Exception:
        pass


def finish(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
