# -*- coding: utf-8 -*-
"""MC-Dropout dan dekomposisi ketidakpastian.

Dengan T forward pass menghasilkan probabilitas p_t:
  epistemik (variance) : Var[p] = (1/T) * sum_t (p_t - p_bar)^2
  BALD / mutual info   : H[p_bar] - (1/T) * sum_t H[p_t]
  aleatorik            : (1/T) * sum_t H[p_t]
  total entropy        : H[p_bar]
  MSP                  : max softmax / max(p, 1-p)   <- pembanding murah

Gatekeeper memakai komponen EPISTEMIK karena ketidakpastian aleatorik
(ambiguitas melekat pada citra) tidak bisa dikurangi dengan merujuk ke dokter.
"""
from __future__ import annotations

import numpy as np
import torch

from src.models import enable_mc_dropout

EPS = 1e-12


def _ent(p: np.ndarray) -> np.ndarray:
    """Entropi biner elemen-wise."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _ent_multi(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


@torch.no_grad()
def mc_predict(model, loader, cfg, n_out: int = 1, device=None,
               return_samples: bool = True):
    """Jalankan T forward pass dengan dropout aktif.

    Return dict berisi array numpy sepanjang N (urutan == urutan loader,
    jadi loader HARUS shuffle=False).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    T = cfg.mc_samples
    n_drop = enable_mc_dropout(model)
    if n_drop == 0:
        print("  [!] PERINGATAN: tidak ada layer Dropout aktif -> MC tidak stokastik.")

    samples, ys, idxs = [], [], []
    for t in range(T):
        outs = []
        first = (t == 0)
        for x, y, i in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device == "cuda")):
                logits = model(x)
            if n_out == 1:
                p = torch.sigmoid(logits.float()).squeeze(-1)
            else:
                p = torch.softmax(logits.float(), dim=1)
            outs.append(p.detach().cpu().numpy())
            if first:
                ys.append(y.numpy())
                idxs.append(i.numpy())
        samples.append(np.concatenate(outs, axis=0))

    S = np.stack(samples, axis=0)           # (T, N) atau (T, N, C)
    mean_p = S.mean(axis=0)

    if n_out == 1:
        var = S.var(axis=0)
        total_ent = _ent(mean_p)
        alea = _ent(S).mean(axis=0)
        msp = np.maximum(mean_p, 1 - mean_p)
    else:
        var = S.var(axis=0).sum(axis=-1)    # jumlah varians antar-kelas
        total_ent = _ent_multi(mean_p)
        alea = _ent_multi(S).mean(axis=0)
        msp = mean_p.max(axis=-1)

    out = {
        "mean_prob": mean_p,
        "var_epistemic": var,
        "entropy_total": total_ent,
        "entropy_aleatoric": alea,
        "bald": np.clip(total_ent - alea, 0, None),
        "msp": msp,
        "y_true": np.concatenate(ys, axis=0),
        "row_index": np.concatenate(idxs, axis=0),
    }
    if return_samples:
        out["mc_samples"] = S
    return out


@torch.no_grad()
def deterministic_predict(model, loader, cfg, n_out: int = 1, device=None):
    """Satu forward pass tanpa dropout (untuk Skenario B / Stage 2)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ps, ys, idxs = [], [], []
    for x, y, i in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device == "cuda")):
            logits = model(x)
        p = (torch.sigmoid(logits.float()).squeeze(-1) if n_out == 1
             else torch.softmax(logits.float(), dim=1))
        ps.append(p.cpu().numpy()); ys.append(y.numpy()); idxs.append(i.numpy())
    mean_p = np.concatenate(ps, axis=0)
    msp = (np.maximum(mean_p, 1 - mean_p) if n_out == 1 else mean_p.max(axis=-1))
    ent = _ent(mean_p) if n_out == 1 else _ent_multi(mean_p)
    return {"mean_prob": mean_p, "msp": msp, "entropy_total": ent,
            "y_true": np.concatenate(ys), "row_index": np.concatenate(idxs)}


def epistemic_score(d: dict, cfg) -> np.ndarray:
    return d["bald"] if cfg.epistemic_measure == "bald" else d["var_epistemic"]


def serialize_samples(S: np.ndarray) -> list:
    """(T, N) -> daftar string 'p1;p2;...' per sampel, untuk disimpan ke CSV."""
    if S.ndim == 3:                       # multiclass: simpan prob kelas argmax mean
        S = S.max(axis=-1)
    return [";".join(f"{v:.5f}" for v in S[:, i]) for i in range(S.shape[1])]
