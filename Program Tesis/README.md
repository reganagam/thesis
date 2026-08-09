# Dual-Stage Hierarchical Breslow Classification dengan Bayesian Gatekeeper

**Regan Agam — NIM 24/PTK/552177/16439** · Tesis M.Sc. Teknik Elektro

Pipeline lengkap: manifest → preprocessing → training → dump prediksi →
analisis offline → figur → Grad-CAM++.

---

## Prinsip desain

1. **Satu manifest, fold dibekukan.** `manifest_frozen.csv` ditulis sekali,
   di-commit ke git. Tidak ada skrip lain yang boleh memanggil KFold lagi.
   Ini menjamin Skenario A/B/C benar-benar *apple-to-apple*.
2. **Semua inferensi menghasilkan CSV per-sampel.** Setelah CSV ada, seluruh
   Bab 4 bisa dikerjakan dengan pandas **tanpa GPU** — analisis bisa diulang
   puluhan kali semalam tanpa training ulang.
3. **Checkpoint bermetadata.** `history` (kurva loss per epoch) adalah
   satu-satunya artefak yang hilang permanen bila tidak disimpan.
4. **Ambang gatekeeper dikalibrasi dari inner-val, bukan test.** Anti kebocoran.

---

## URUTAN MENJALANKAN

```bash
# ---------- SEKALI DI AWAL ----------
python run_check_setup.py            # verifikasi GPU, library, path gambar
python run_00_build_manifest.py      # -> data/manifest_frozen.csv  (COMMIT KE GIT!)
python run_01_preprocess_cache.py    # -> cache/raw_224 & cache/proc_224

# ---------- TRAINING ----------
python run_02_train_stage1.py             # Stage 1 + dump MC-Dropout
python run_03_train_stage2.py --mode both # Stage 2 (gt & oof)
python run_04_train_flat.py               # Skenario A (flat multiclass)

# ---------- INFERENSI EKSTERNAL ----------
python run_05_external_kawahara.py        # zero-shot, dikunci sampai tahap ini

# ---------- ANALISIS & FIGUR (TANPA GPU) ----------
python run_06_analysis.py                 # tabel A/A-rej/B/C, AURC, Wilcoxon
python run_06_analysis.py --split external
python run_07_visualize.py                # semua figur
python run_07_visualize.py --split external
python run_08_gradcam.py                  # Grad-CAM++

# ---------- ABLASI ----------
python run_09_ablations.py --what plan      # daftar perintah ablasi
python run_09_ablations.py --what shortcut  # diagnostik source shortcut
```

> **Penting:** `run_03` mode `oof` membutuhkan checkpoint dari `run_02`.
> `run_05` membutuhkan checkpoint dari `run_02`–`run_04`.

---

## Peta file

| File | Peran |
|---|---|
| `config.py` | **Satu-satunya** tempat mengedit path & hyperparameter |
| `src/paths.py` | Resolver path gambar 4 sumber (dengan indeks fallback) |
| `src/manifest.py` | Bangun manifest, label, patient_id, subsample ISIC, fold beku |
| `src/preprocessing.py` | Hair removal, Shades of Gray (p=6), border crop |
| `src/data.py` | Dataset, transform, split fold, cache |
| `src/models.py` | Backbone + head Dense-128, `enable_mc_dropout()` |
| `src/uncertainty.py` | MC-Dropout, dekomposisi epistemik/aleatorik, BALD, MSP |
| `src/metrics.py` | Metrik, risk-coverage, AURC, Wilcoxon, bootstrap CI |
| `src/selective.py` | Kalibrasi ambang bersarang, gatekeeper, perakitan 3-kelas |
| `src/engine.py` | Loop training: AMP, class weight, early stopping |
| `src/checkpoint.py` | Simpan/muat checkpoint **bermetadata + history** |
| `src/plots.py` | Semua figur Bab 4 |
| `src/gradcam.py` | Grad-CAM++ |
| `src/wb.py` | Wrapper wandb yang tidak pernah menggagalkan run |

---

## Keluaran (di `PROJECT_ROOT`)

```
data/manifest_frozen.csv              <- sumber kebenaran, COMMIT
cache/{raw,proc}_224/                 <- citra terproses
checkpoints/*.pt                      <- bobot + config + history + val paths
predictions/*.csv                     <- prediksi per-sampel + skor ketidakpastian
results/*.csv                         <- tabel Bab 4
figures/*.png|pdf                     <- figur Bab 4
```

### Skema CSV prediksi

`img_path, patient_id, source, fold, seed, split, stage, y_true_*,
p_miv_mean / p_bt_mean / p_c0..p_c2, var_epistemic, bald, entropy_total,
entropy_aleatoric, msp, mc_samples`

`mc_samples` menyimpan **30 nilai mentah** MC-Dropout per gambar — sehingga
semua skor ketidakpastian bisa diturunkan ulang tanpa inferensi ulang.

---

## Skenario yang dievaluasi

| Kode | Sistem | Menguji |
|---|---|---|
| `A_flat` | Satu CNN, 3 kelas | Baseline (jalur *supervised* paper rujukan) |
| `A_flat_reject` | Flat + penolakan | **Kontrol wajib** — memisahkan efek hierarki dari efek penolakan |
| `B_hier` | Stage 1 → Stage 2 | Apakah dekomposisi hierarkis membantu? |
| `C_hier_gate` | Stage 1 → Gatekeeper → Stage 2 | Apakah penolakan menekan perambatan eror? |

Semua dinilai pada **confusion matrix 3×3 yang sama**; C melaporkan tambahan
**coverage aktual**.

### Pembanding skor ketidakpastian

Membuang 10% sampel paling tidak pasti hampir selalu menaikkan akurasi — untuk
metrik apa pun. Karena itu `run_06` otomatis membandingkan pada coverage sama:
penolakan acak (lantai), **MSP** (1 forward pass — pembanding terkuat),
predictive entropy, MC variance, dan BALD. Metriknya **AURC** dan
**AUROC deteksi eror**.

Jika MC variance tidak mengungguli MSP, itu **tetap temuan yang layak
dilaporkan**: 30× biaya komputasi tidak sepadan pada skala data ini.

---

## Catatan pertahanan sidang

- `enable_mc_dropout()` membekukan BatchNorm dan hanya mengaktifkan Dropout.
  `model.train()` naif akan merusak statistik BN dan membuat estimasi
  ketidakpastian tidak valid.
- Split memakai `StratifiedGroupKFold` pada `patient_id` — VRUH punya 860
  gambar dari hanya 314 pasien. Ada assertion yang gagal keras bila bocor.
- ISIC tidak menyumbang kasus in situ dan 58% bernilai tepat 0.8 mm; karena
  itu disubsampel. Jalankan `--isic-mode full/none` sebagai analisis sensitivitas.
- Kawahara memakai proxy AJCC-7 (0.76 mm) dan **dikunci** sampai `run_05`.
- Ambang gatekeeper dikalibrasi dari inner-val lalu dibekukan; **coverage
  aktual di test tidak akan persis 90%** — angka itu yang dilaporkan.

---

## Lisensi

Kode ini orisinal. Bila Anda menyalin bagian dari repo
`jpdominguez/Breslow_Melanoma_DeepLearning` (**GPL-3.0**), letakkan di folder
terpisah dengan header atribusi dan patuhi ketentuan GPL saat mendistribusikan.
