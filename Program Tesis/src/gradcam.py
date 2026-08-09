# -*- coding: utf-8 -*-
"""Grad-CAM++ untuk Explainable AI (Bab 4).

Hanya butuh bobot + gambar -> bisa dijalankan KAPAN SAJA dari checkpoint,
tanpa training ulang.
"""
from __future__ import annotations

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C


class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts = None
        self.grads = None
        self.h1 = target_layer.register_forward_hook(self._fwd)
        self.h2 = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def remove(self):
        self.h1.remove(); self.h2.remove()

    def __call__(self, x, class_idx=None, n_out=1):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if n_out == 1:
            score = logits.squeeze(-1).sum()
        else:
            class_idx = (logits.argmax(1) if class_idx is None
                         else torch.tensor([class_idx], device=x.device))
            score = logits.gather(1, class_idx.view(-1, 1)).sum()
        score.backward(retain_graph=True)

        g, a = self.grads, self.acts                     # (B,C,H,W)
        g2, g3 = g.pow(2), g.pow(3)
        denom = 2 * g2 + (a * g3).sum(dim=(2, 3), keepdim=True)
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        weights = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * a).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze(1).cpu().numpy()
        out = []
        for c in cam:
            c = c - c.min()
            out.append(c / (c.max() + 1e-8))
        return np.stack(out)


def overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha=0.42) -> np.ndarray:
    hm = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (cam.shape[1], cam.shape[0]))
    return np.uint8((1 - alpha) * img + alpha * hm)


def render_grid(items, fname, ncol=4, run=None, suptitle=""):
    """items: list of (img_rgb, cam, caption)."""
    n = len(items)
    if n == 0:
        return None
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol * 2, figsize=(3.0 * ncol * 2, 3.2 * nrow))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (img, cam, cap) in enumerate(items):
        r, c = divmod(k, ncol)
        axes[r, 2 * c].imshow(img); axes[r, 2 * c].set_title("Asli", fontsize=8)
        axes[r, 2 * c + 1].imshow(overlay(img, cam))
        axes[r, 2 * c + 1].set_title(cap, fontsize=7)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    os.makedirs(C.DIR_FIG, exist_ok=True)
    path = os.path.join(C.DIR_FIG, fname)
    fig.savefig(path, dpi=250, bbox_inches="tight"); plt.close(fig)
    from src import wb
    wb.log_image(run, path, os.path.splitext(fname)[0])
    print(f"    [fig] {path}")
    return path
