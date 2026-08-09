# -*- coding: utf-8 -*-
"""Prapemrosesan citra dermoskopi.

1. Black-hat hair removal + inpainting
2. Color constancy (Shades of Gray, Minkowski p=6)  -- lebih baik dari Gray-World
3. Pemotongan tepi hitam / vignette mikroskop

Dipakai sebagai TAHAP CACHE offline (run_01), bukan di dalam DataLoader,
supaya training tidak melambat dan hasilnya deterministik.
"""
from __future__ import annotations

import os
import cv2
import numpy as np


# --------------------------------------------------------------------------- #
def remove_hair(img_bgr: np.ndarray, kernel_size: int = 17,
                thresh: int = 10, inpaint_radius: int = 3) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(img_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)


def shades_of_gray(img_bgr: np.ndarray, p: int = 6) -> np.ndarray:
    """Normalisasi warna Minkowski-p. p=1 -> Gray-World, p=inf -> Max-RGB."""
    img = img_bgr.astype(np.float32)
    pow_mean = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p)
    pow_mean = np.where(pow_mean == 0, 1.0, pow_mean)
    scale = pow_mean.mean() / pow_mean
    out = img * scale.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_black_border(img_bgr: np.ndarray, thresh: int = 20,
                      min_ratio: float = 0.35) -> np.ndarray:
    """Buang tepi hitam / vignette lensa dengan bounding box area non-gelap."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > thresh).astype(np.uint8)
    if mask.sum() == 0:
        return img_bgr
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_bgr
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    H, W = gray.shape
    if w < min_ratio * W or h < min_ratio * H:   # crop terlalu agresif -> batalkan
        return img_bgr
    return img_bgr[y:y + h, x:x + w]


def preprocess_image(img_bgr: np.ndarray, do_hair=True, do_cc=True,
                     do_crop=True, sog_p: int = 6) -> np.ndarray:
    out = img_bgr
    if do_crop:
        out = crop_black_border(out)
    if do_hair:
        out = remove_hair(out)
    if do_cc:
        out = shades_of_gray(out, p=sog_p)
    return out


def process_file(src: str, dst: str, size: int = 224, **kw) -> bool:
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return False
    img = preprocess_image(img, **kw)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    return bool(cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95]))


def resize_only(src: str, dst: str, size: int = 224) -> bool:
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return False
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    return bool(cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95]))
