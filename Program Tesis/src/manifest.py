# -*- coding: utf-8 -*-
"""Membangun manifest_frozen.csv — SUMBER KEBENARAN TUNGGAL.

Kolom keluaran:
    img_path, image_id, patient_id, source,
    breslow_mm, breslow_is_range, ambiguous_cutoff,
    is_invasive, depth_ge_cut, multiclass,
    is_external, fold

Aturan mutlak:
  * TIDAK ADA fillna(0) pada breslow — nilai hilang dibuang & dilaporkan.
  * fold ditulis SEKALI ke disk; skrip lain tidak boleh memanggil KFold lagi.
  * split memakai StratifiedGroupKFold pada patient_id (anti kebocoran).
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config as C
from src import paths as P

COLS = ["img_path", "image_id", "patient_id", "source", "breslow_mm",
        "breslow_is_range", "ambiguous_cutoff", "is_invasive",
        "depth_ge_cut", "multiclass", "is_external", "fold"]


# --------------------------------------------------------------------------- #
def _derive_labels(df: pd.DataFrame, cut: float) -> pd.DataFrame:
    df = df.copy()
    df["is_invasive"] = df["is_invasive"].astype(int)
    depth = np.where(df["is_invasive"] == 1,
                     (df["breslow_mm"] >= cut).astype(float), np.nan)
    df["depth_ge_cut"] = pd.array(depth, dtype="Int64")
    df["multiclass"] = np.where(df["is_invasive"] == 0, 0,
                                np.where(df["breslow_mm"] >= cut, 2, 1)).astype(int)
    return df


# ----------------------------- VRUH ---------------------------------------- #
def load_vruh(cfg) -> pd.DataFrame:
    d = pd.read_csv(C.VRUH_CSV)
    d = d.rename(columns={"breslow_thickness_mm": "breslow_mm"})
    d["breslow_mm"] = pd.to_numeric(d["breslow_mm"], errors="coerce")
    n_nan = int(d["breslow_mm"].isna().sum())
    if n_nan:
        print(f"  [!] VRUH: {n_nan} baris tanpa nilai Breslow -> DIBUANG "
              f"(tidak diasumsikan in situ).")
        d = d[d["breslow_mm"].notna()]
    d["img_path"] = d["filepath"].apply(P.resolve_vruh)
    P.report_missing(d, label="VRUH")
    d = d[d["img_path"].notna()].copy()

    d["source"] = "VRUH"
    d["breslow_is_range"] = False
    d["ambiguous_cutoff"] = False
    d["is_invasive"] = (d["breslow_mm"] > 0.0).astype(int)
    d = d[["img_path", "image_id", "patient_id", "source", "breslow_mm",
           "breslow_is_range", "ambiguous_cutoff", "is_invasive"]]
    return _derive_labels(d, cfg.breslow_cutoff_mm)


# ----------------------------- POLESIE ------------------------------------- #
POLESIE_MAP = {
    "0 mm (In situ)": (0.0, False),
    "0.1-0.5 mm":     (0.3, True),
    "0.6-0.8 mm":     (0.7, True),    # AMBIGU terhadap ambang 0.8
    "0.9-1.0 mm":     (0.95, True),
    "1.1-2.0 mm":     (1.5, True),
    "2.1-4.0 mm":     (3.0, True),
    "> 4mm":          (5.0, True),
}
POLESIE_AMBIGUOUS = {"0.6-0.8 mm"}


def load_polesie(cfg) -> pd.DataFrame:
    d = pd.read_csv(C.POLESIE_CSV)
    raw = d["raw_diagnosis"].astype(str).str.strip()

    unknown = sorted(set(raw) - set(POLESIE_MAP))
    if unknown:
        print(f"  [!] Polesie: kategori tak dikenal {unknown} -> DIBUANG.")
    keep = raw.isin(POLESIE_MAP)
    d, raw = d[keep].copy(), raw[keep]

    d["breslow_mm"] = raw.map(lambda x: POLESIE_MAP[x][0]).astype(float)
    d["breslow_is_range"] = raw.map(lambda x: POLESIE_MAP[x][1])
    d["ambiguous_cutoff"] = raw.isin(POLESIE_AMBIGUOUS)
    d["is_invasive"] = (d["breslow_mm"] > 0.0).astype(int)

    # patient_id dari pola 'case_page<N>_img<M>' (193 gambar -> 184 kasus)
    d["patient_id"] = ("POL_" + d["image_id"].astype(str)
                       .str.extract(r"case_page(\d+)_", expand=False)
                       .fillna(d["image_id"].astype(str)))
    d["img_path"] = d["image_id"].apply(P.resolve_polesie)
    P.report_missing(d, label="Polesie")
    d = d[d["img_path"].notna()].copy()
    d["source"] = "Polesie"

    n_amb = int(d["ambiguous_cutoff"].sum())
    print(f"  [i] Polesie: {n_amb} gambar pada bucket 0.6-0.8 mm (ambigu vs ambang).")
    if cfg.drop_polesie_ambiguous:
        d = d[~d["ambiguous_cutoff"]].copy()
        print("      -> dibuang (analisis sensitivitas aktif).")

    d = d[["img_path", "image_id", "patient_id", "source", "breslow_mm",
           "breslow_is_range", "ambiguous_cutoff", "is_invasive"]]
    return _derive_labels(d, cfg.breslow_cutoff_mm)


# ----------------------------- ISIC ---------------------------------------- #
def load_isic(cfg) -> pd.DataFrame:
    if cfg.isic_mode == "none":
        print("  [i] ISIC: dilewati (isic_mode='none').")
        return pd.DataFrame(columns=COLS[:-2])

    d = pd.read_csv(C.ISIC_CSV)
    d["breslow_mm"] = pd.to_numeric(d["breslow_mm"], errors="coerce")
    n_nan = int(d["breslow_mm"].isna().sum())
    if n_nan:
        print(f"  [!] ISIC: {n_nan} baris tanpa nilai Breslow -> DIBUANG.")
        d = d[d["breslow_mm"].notna()].copy()

    cut = cfg.breslow_cutoff_mm
    n_exact = int((d["breslow_mm"] == cut).sum())
    print(f"  [i] ISIC: {len(d)} gambar, {n_exact} ({100*n_exact/max(len(d),1):.0f}%) "
          f"bernilai TEPAT {cut} mm -> diduga artefak pembulatan.")
    print(f"  [i] ISIC: {int((d['breslow_mm']==0).sum())} kasus in situ "
          f"(diharapkan 0 — arsip ISIC tidak menyimpan ketebalan untuk Mis).")

    d["ambiguous_cutoff"] = (d["breslow_mm"] == cut)

    if cfg.isic_mode == "subsample":
        rng = np.random.RandomState(42)
        thin = d[d["breslow_mm"] < cut]
        thick = d[d["breslow_mm"] >= cut]
        if cfg.isic_deprioritize_exact_cut:
            # utamakan yang BUKAN tepat di ambang
            pref = thick[thick["breslow_mm"] > cut]
            rest = thick[thick["breslow_mm"] == cut]
            order = pd.concat([pref.sample(frac=1, random_state=rng),
                               rest.sample(frac=1, random_state=rng)])
        else:
            order = thick.sample(frac=1, random_state=rng)
        thin = thin.head(cfg.isic_n_thin)
        thick = order.head(cfg.isic_n_thick)
        d = pd.concat([thin, thick], ignore_index=True)
        print(f"  [i] ISIC disubsampel -> {len(thin)} tipis + {len(thick)} tebal "
              f"= {len(d)} gambar (cegah source-label shortcut).")

    d["img_path"] = d["isic_id"].apply(P.resolve_isic)
    P.report_missing(d, label="ISIC")
    d = d[d["img_path"].notna()].copy()

    d["image_id"] = d["isic_id"]
    d["patient_id"] = "ISIC_" + d["isic_id"].astype(str)   # 1 gambar = 1 kasus
    d["source"] = "ISIC"
    d["breslow_is_range"] = False
    d["is_invasive"] = (d["breslow_mm"] > 0.0).astype(int)
    d = d[["img_path", "image_id", "patient_id", "source", "breslow_mm",
           "breslow_is_range", "ambiguous_cutoff", "is_invasive"]]
    return _derive_labels(d, cfg.breslow_cutoff_mm)


# ----------------------------- KAWAHARA (EKSTERNAL) ------------------------ #
KAWA_MAP = {
    "melanoma (in situ)":          (0.0, 0, False),
    "melanoma (less than 0.76 mm)": (0.5, 1, True),
    "melanoma (0.76 to 1.5 mm)":   (1.1, 1, True),
    "melanoma (more than 1.5 mm)": (2.5, 1, True),
}


def load_kawahara(cfg) -> pd.DataFrame:
    d = pd.read_csv(C.KAWAHARA_CSV)
    diag = d["diagnosis"].astype(str).str.strip()
    keep = diag.isin(KAWA_MAP)          # buang non-melanoma, metastasis, unspecified
    print(f"  [i] Kawahara: {int(keep.sum())} kasus melanoma dari {len(d)} baris "
          f"(buang non-melanoma, 'melanoma metastasis', 'melanoma' tanpa keterangan).")
    d, diag = d[keep].copy(), diag[keep]

    d["breslow_mm"] = diag.map(lambda x: KAWA_MAP[x][0]).astype(float)
    d["is_invasive"] = diag.map(lambda x: KAWA_MAP[x][1]).astype(int)
    d["breslow_is_range"] = diag.map(lambda x: KAWA_MAP[x][2])
    # proxy AJCC-7: pita 0.76-0.80 mm tidak terpisahkan -> tandai ambigu
    d["ambiguous_cutoff"] = diag.eq("melanoma (0.76 to 1.5 mm)")

    # depth memakai PROXY 0.76 mm, bukan 0.8 mm
    proxy = cfg.kawahara_proxy_mm
    depth = np.where(d["is_invasive"] == 1,
                     (d["breslow_mm"] >= proxy).astype(float), np.nan)
    d["depth_ge_cut"] = pd.array(depth, dtype="Int64")
    d["multiclass"] = np.where(d["is_invasive"] == 0, 0,
                               np.where(d["breslow_mm"] >= proxy, 2, 1)).astype(int)

    d["img_path"] = d["derm"].apply(P.resolve_kawahara)   # citra dermoskopi saja
    P.report_missing(d, label="Kawahara")
    d = d[d["img_path"].notna()].copy()

    d["image_id"] = d["derm"].astype(str)
    d["patient_id"] = "KAW_" + d["case_num"].astype(str)
    d["source"] = "Kawahara"
    return d[["img_path", "image_id", "patient_id", "source", "breslow_mm",
              "breslow_is_range", "ambiguous_cutoff", "is_invasive",
              "depth_ge_cut", "multiclass"]]


# --------------------------------------------------------------------------- #
def assign_folds(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Fold pada data internal saja; eksternal = -1. Stratifikasi 3 kelas,
    grouping patient_id."""
    df = df.copy()
    df["fold"] = -1
    internal = df[~df["is_external"]].copy()

    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=42)
    y = internal["multiclass"].astype(int).values
    groups = internal["patient_id"].astype(str).values

    for k, (_tr, va) in enumerate(sgkf.split(np.zeros(len(internal)), y, groups)):
        df.loc[internal.index[va], "fold"] = k

    # ---- assertion anti-kebocoran ----
    for k in range(cfg.n_folds):
        tr_p = set(df[(~df.is_external) & (df.fold != k)]["patient_id"])
        va_p = set(df[(~df.is_external) & (df.fold == k)]["patient_id"])
        assert not (tr_p & va_p), f"KEBOCORAN patient-level pada fold {k}!"
    assert (df.loc[~df.is_external, "fold"] >= 0).all(), "ada baris internal tanpa fold"
    print(f"  [OK] Tidak ada pasien yang muncul di train & val pada {cfg.n_folds} fold.")
    return df


def build(cfg=None, save=True) -> pd.DataFrame:
    cfg = cfg or C.CFG
    print("=" * 70)
    print("MEMBANGUN MANIFEST")
    print("=" * 70)

    parts = []
    for name, fn in [("VRUH", load_vruh), ("Polesie", load_polesie), ("ISIC", load_isic)]:
        print(f"\n-- {name} --")
        d = fn(cfg)
        if len(d):
            d["is_external"] = False
            parts.append(d)

    print("\n-- Kawahara (EKSTERNAL, dikunci) --")
    kaw = load_kawahara(cfg)
    kaw["is_external"] = True
    parts.append(kaw)

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["img_path"]).reset_index(drop=True)

    print("\n-- Pembagian fold (StratifiedGroupKFold, group=patient_id) --")
    df = assign_folds(df, cfg)
    df = df[COLS]

    if save:
        df.to_csv(C.MANIFEST, index=False)
        print(f"\n[SAVED] {C.MANIFEST}")
    summarize(df)
    return df


def summarize(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("RINGKASAN MANIFEST")
    print("=" * 70)
    names = {0: "Mis", 1: "Miv<cut", 2: "Miv>=cut"}
    piv = (df.assign(kelas=df["multiclass"].map(names))
             .pivot_table(index="source", columns="kelas",
                          values="img_path", aggfunc="count", fill_value=0))
    print(piv.to_string())
    print(f"\nTotal gambar        : {len(df)}")
    print(f"Total pasien/kasus  : {df['patient_id'].nunique()}")
    intern = df[~df["is_external"]]
    print(f"Internal (latih/CV) : {len(intern)} gambar / {intern['patient_id'].nunique()} kasus")
    print(f"Eksternal (Kawahara): {int(df['is_external'].sum())} gambar")
    print(f"Ambigu di ambang    : {int(df['ambiguous_cutoff'].sum())} gambar")
    print("\nDistribusi per fold (internal):")
    print(pd.crosstab(intern["fold"], intern["multiclass"]).to_string())
