# -*- coding: utf-8 -*-
"""Backbone + head (Dense 128 -> output), sesuai paper Dominguez-Morales.

Dropout ditempatkan di head sebagai sumber MC-Dropout. Untuk DenseNet-121
kita juga menyalakan drop_rate internal dense block sehingga sampling MC
berasal dari ruang bobot yang lebih dalam, bukan hanya layer terakhir.
(ResNet-50 torchvision TIDAK punya dropout internal -> tulis sebagai limitasi.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


def build_model(cfg, n_out: int) -> nn.Module:
    bb = cfg.backbone.lower()
    p = cfg.dropout_p

    if bb == "densenet121":
        w = tvm.DenseNet121_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        m = tvm.densenet121(weights=w, drop_rate=cfg.densenet_drop_rate)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(
            nn.Linear(in_f, 128), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(128, n_out))

    elif bb == "resnet50":
        w = tvm.ResNet50_Weights.IMAGENET1K_V2 if cfg.pretrained else None
        m = tvm.resnet50(weights=w)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Linear(in_f, 128), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(128, n_out))

    elif bb == "efficientnet_b0":
        w = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        m = tvm.efficientnet_b0(weights=w)
        in_f = m.classifier[-1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p), nn.Linear(in_f, 128), nn.ReLU(inplace=True),
            nn.Dropout(p), nn.Linear(128, n_out))
    else:
        raise ValueError(f"backbone tidak dikenal: {cfg.backbone}")
    return m


def target_layer(model: nn.Module, cfg) -> nn.Module:
    """Layer konvolusi terakhir untuk Grad-CAM++."""
    bb = cfg.backbone.lower()
    if bb == "densenet121":
        return model.features.denseblock4
    if bb == "resnet50":
        return model.layer4
    if bb == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(cfg.backbone)


def enable_mc_dropout(model: nn.Module) -> int:
    """eval() untuk semua layer (BatchNorm BEKU), lalu HANYA Dropout diaktifkan.

    PENTING: memanggil model.train() secara naif akan merusak statistik
    running BatchNorm dan membuat estimasi ketidakpastian tidak valid.
    """
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            n += 1
    return n


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
