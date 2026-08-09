# -*- coding: utf-8 -*-
"""
config.py — SATU-SATUNYA sumber kebenaran untuk path & hyperparameter.

Ubah di sini saja; semua script membaca dari file ini.
Python 3.9 compatible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# =============================================================================
# 1. PATH  — EDIT BAGIAN INI SESUAI MESIN ANDA
# =============================================================================
DATASET_ROOT = os.environ.get("DS_ROOT", r"D:\REGAN-Projek\THESIS\Dataset")
PROJECT_ROOT = os.environ.get("PJ_ROOT", r"D:\REGAN-Projek\THESIS\Result")

# Metadata CSV (4 sumber). Ubah kalau nama/letak file berbeda.
VRUH_CSV     = os.path.join(DATASET_ROOT, "VRUH_dataset_cleaned.csv")
ISIC_CSV     = os.path.join(DATASET_ROOT, "ISIC_breslow_metadata.csv")
POLESIE_CSV  = os.path.join(DATASET_ROOT, "polesie_ground_truth.csv")
KAWAHARA_CSV = os.path.join(DATASET_ROOT, "kawahara_meta.csv")

# Folder gambar tiap sumber (dipakai resolver; kalau layout beda, resolver
# akan otomatis mengindeks seluruh DATASET_ROOT berdasarkan nama file).
SOURCE_DIRS = {
    "VRUH":     os.path.join(DATASET_ROOT, "VRUH"),
    "ISIC":     os.path.join(DATASET_ROOT, "ISIC"),
    "Polesie":  os.path.join(DATASET_ROOT, "Polesei"),   # ejaan folder Anda
    "Kawahara": os.path.join(DATASET_ROOT, "Kawahara"),
}

# Output
DIR_DATA   = os.path.join(PROJECT_ROOT, "data")
DIR_CACHE  = os.path.join(PROJECT_ROOT, "cache")
DIR_CKPT   = os.path.join(PROJECT_ROOT, "checkpoints")
DIR_PRED   = os.path.join(PROJECT_ROOT, "predictions")
DIR_RESULT = os.path.join(PROJECT_ROOT, "results")
DIR_FIG    = os.path.join(PROJECT_ROOT, "figures")
DIR_LOG    = os.path.join(PROJECT_ROOT, "logs")

MANIFEST = os.path.join(DIR_DATA, "manifest_frozen.csv")

for _d in (DIR_DATA, DIR_CACHE, DIR_CKPT, DIR_PRED, DIR_RESULT, DIR_FIG, DIR_LOG):
    os.makedirs(_d, exist_ok=True)


# =============================================================================
# 2. KONFIGURASI EKSPERIMEN
# =============================================================================
@dataclass
class Config:
    # ---- label ---------------------------------------------------------------
    breslow_cutoff_mm: float = 0.8      # ambang Stage 2 (AJCC ke-8 / EU 2022)
    kawahara_proxy_mm: float = 0.76     # proxy AJCC ke-7 untuk Derm7pt

    # ---- komposisi dataset ---------------------------------------------------
    # ISIC hanya berisi kasus invasif -> disubsampel agar tidak jadi
    # source-label shortcut. Lihat blueprint Bab 3.
    isic_mode: str = "subsample"        # 'subsample' | 'full' | 'none'
    isic_n_thin: int = 53               # ambil semua (<0.8 mm)
    isic_n_thick: int = 100             # acak dari >=0.8 mm
    isic_deprioritize_exact_cut: bool = True   # hindari nilai tepat 0.8 mm
    drop_polesie_ambiguous: bool = False       # analisis sensitivitas bucket 0.6-0.8

    # ---- preprocessing -------------------------------------------------------
    use_preprocessing: bool = True      # True -> pakai cache 'proc', False -> 'raw'
    img_size: int = 224

    # ---- backbone ------------------------------------------------------------
    backbone: str = "densenet121"       # 'densenet121' | 'resnet50'
    pretrained: bool = True
    dropout_p: float = 0.3              # dropout head (sumber MC)
    densenet_drop_rate: float = 0.2     # dropout DI DALAM dense block

    # ---- ketidakpastian ------------------------------------------------------
    mc_samples: int = 30               # T
    epistemic_measure: str = "variance"  # 'variance' | 'bald'
    reject_percentile: float = 10.0     # referral budget X%
    gatekeeper_mode: str = "symmetric"  # 'symmetric' | 'miv_only'

    # ---- training ------------------------------------------------------------
    epochs: int = 25
    lr: float = 1e-4
    batch_size: int = 32
    weight_decay: float = 1e-5
    early_stop_patience: int = 6
    num_workers: int = 0                # set 0 kalau bermasalah di Windows
    use_amp: bool = True
    inner_val_frac: float = 0.15        # split kalibrasi + early stopping

    # ---- cross-validation ----------------------------------------------------
    n_folds: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])

    # ---- Stage 2 -------------------------------------------------------------
    stage2_train_mode: str = "oof"      # 'gt' (label asli) | 'oof' (prediksi Stage 1)

    # ---- wandb ---------------------------------------------------------------
    wandb_project: str = "breslow-hierarchical-gatekeeper"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"             # 'online' | 'offline' | 'disabled'

    # ---- misc ----------------------------------------------------------------
    deterministic: bool = True

    def tag(self) -> str:
        """Tag pendek yang ikut ke nama checkpoint & CSV."""
        pp = "proc" if self.use_preprocessing else "raw"
        return f"{self.backbone}_{pp}_{self.isic_mode}"

    def cache_key(self) -> str:
        return "proc" if self.use_preprocessing else "raw"

    def to_dict(self):
        return asdict(self)


CFG = Config()


# =============================================================================
# 3. KONSTANTA
# =============================================================================
CLASS_NAMES_3 = ["Mis", "Miv<0.8", "Miv>=0.8"]
CLASS_NAMES_S1 = ["Mis", "Miv"]
CLASS_NAMES_S2 = ["<0.8mm", ">=0.8mm"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Skenario ablasi
SCENARIOS = ["A_flat", "A_flat_reject", "B_hier", "C_hier_gate"]
