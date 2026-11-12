# Bean leaf disease classification with a from-scratch CNN

[![tests](https://github.com/alisimsek55/bean-leaf-disease-cnn/actions/workflows/tests.yml/badge.svg)](https://github.com/alisimsek55/bean-leaf-disease-cnn/actions/workflows/tests.yml)

[Türkçe](README.tr.md)

A 16-layer convolutional neural network, designed and trained from scratch in PyTorch, that classifies bean leaf photos as common bacterial blight (CBB), halo blight, or healthy. It was built as a university "AI in Agriculture" course project. It reaches 97.6% validation accuracy, an upper bound because the committed split leaks burst frames between train and validation (see [Results](#results)). The more interesting part of the repository is what happened after that number came out: I tested how much of it reflects disease recognition and how much reflects dataset artifacts, and the answer is "less than the headline suggests".

![Leaves before and after preprocessing: raw image, leaf mask, background painted black and cropped](ciktilar/sekiller/on_isleme_oncesi_sonrasi.png)

*One example from each source/class combination: raw image, leaf mask, and the cropped, background-blackened image the model sees.*

## Results

Validation set, n = 738 images (70/30 split grouped by dHash near-duplicate groups). Source: [`ciktilar/metrikler.json`](ciktilar/metrikler.json).

> **Known leakage in this split.** The train and validation sets share no near-duplicate group, but a later audit ([`scripts/bolme_denetimi.py`](scripts/bolme_denetimi.py)) found that burst frames from source A were split across them: 145 capture-seconds appear on both sides, and 299 of the 314 source-A validation images (40.5% of the 738-image validation set) have a frame from the same second in training. The numbers below, 97.56% accuracy and 95.5% macro F1, are therefore optimistic and should be read as an upper bound until the split is rebuilt with burst-aware grouping and the models are retrained (planned). See [Limitations](#limitations).

| Metric | Value |
|---|---|
| Accuracy | 97.56% (720/738) |
| Balanced accuracy | 98.68% |
| Macro F1 | 95.48% |
| Majority-class baseline (always "healthy") | 69.0% accuracy, 27.2% macro F1 |

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Common bacterial blight (CBB) | 1.000 | 0.994 | 0.997 | 160 |
| Halo blight | 0.793 | 1.000 | 0.885 | 69 |
| Healthy | 1.000 | 0.967 | 0.983 | 509 |

Confusion matrix (rows = true, columns = predicted: CBB, halo, healthy): CBB `[159, 1, 0]`, halo `[0, 69, 0]`, healthy `[0, 17, 492]`.

There is no separate test set: the validation set was also used for early stopping and model selection, so these numbers are slightly optimistic even before the checks below.

## Is the accuracy real?

Almost every class comes from a different data source with a different background (details under [Data](#data)): all halo blight images come from source B, 95% of CBB images from source A. A network can score well by recognising the source instead of the disease. I ran four checks.

| Check | What was done | Result | What it shows |
|---|---|---|---|
| Silhouette only | Same architecture, split and settings, but the input is only the leaf outline (flat gray inside, black outside; no color or texture) | 52.3% accuracy, 47.9% macro F1 (majority baseline: 69.0%) | Leaf shape alone does not explain the result. |
| Source prediction | Same CNN on the masked healthy leaves only, target = data source (A vs B) | 100% (509/509) | Masking removed the background but not the source signature (lighting, camera, leaf side, edge artifacts). Because halo = source B and CBB is about 95% source A, the CBB vs halo distinction cannot be separated from source identity. |
| Unseen source (Embrapa) | Source C removed from training and validation, model retrained, then tested on the 27 Embrapa images (expert-labelled *Phaseolus vulgaris*) | 5/25 CBB correct (20%): 13 called halo, 7 called healthy. 2/2 healthy correct. The same model scored 99.59% accuracy on the A+B validation set. | Generalization to a source the model has not seen is weak. n is small (25 CBB leaves). |
| Grad-CAM | Last conv layer (8x8), predicted class, over all 738 validation images | Median CAM-weight enrichment on lesion pixels: 1.44x (CBB), 2.65x (halo); about 1x on the leaf edge. 32-45% of CAM weight still lands on the black background. | The model does look at lesions, but this cannot tell whether it uses disease-specific features or source-specific lesion appearance. The 8x8 map is also coarse. |

![Main model vs silhouette control vs majority baseline vs unseen-source test](ciktilar/sekiller/kontrol_deneyleri.png)

![Grad-CAM examples per class](ciktilar/sekiller/gradcam_ornekleri.png)

**Conclusion.** Distinguishing diseased from healthy leaves within the same source looks plausible: the background is removed, the silhouette is not enough, and the model attends to lesions. But the split is not leak-free: in source A, burst frames of the same leaf ended up on both sides (299 of 314 source-A validation images have a same-second sibling in training), so the source-A results are inflated by an unknown amount. Sources B and C have no capture timestamps in their file names, so for them only the dHash grouping applies and burst leakage can be neither detected nor ruled out by the audit. For healthy leaves, 100% of source A and 95.1% of source B were classified correctly, and the errors (17, all source B) went toward halo blight. The CBB vs halo blight distinction cannot be shown to come from the disease, because class and source overlap almost completely, and the one-in-five result on unseen-source CBB leaves says the 97.6% is an in-distribution (same sources) figure. Treat it as an upper bound.

An optional ImageNet-pretrained EfficientNet-B0 (fine-tuned, 4,011,391 parameters) reached 100% validation accuracy. The same source overlap applies, so this is not evidence of disease recognition either. The unseen-source test was not run for it.

## Pipeline

### Data

Three public sources, 2,460 images in total (CBB 533, halo blight 231, healthy 1,696). Labels are the dataset authors' labels and were not changed. Full details: [docs/DATASET.tr.md](docs/DATASET.tr.md) (Turkish).

| Source | Content used | Background |
|---|---|---|
| A: Mendeley Data, Rashid et al. | CBB 508, healthy 549 | Phone photos on black/dark cloth |
| B: Mendeley Data, Mojumdar et al. | Halo blight 231, healthy 1,145 | Background removed (white), PNG |
| C: Embrapa PDDB, Barbedo et al. | CBB 25, healthy 2 (one leaf) | Color calibration card, expert-labelled *P. vulgaris* |

Nine images with failed masks (all source A, healthy) were excluded, leaving 2,451.

| Class | Train | Validation | Total |
|---|---|---|---|
| CBB | 373 | 160 | 533 |
| Halo blight | 162 | 69 | 231 |
| Healthy | 1,178 | 509 | 1,687 |
| Total | 1,713 | 738 | 2,451 |

Many images are near-identical frames of the same leaf; the 231 halo blight images come from only 146 distinct leaf groups.

### Preprocessing

[`src/01_on_isleme.py`](src/01_on_isleme.py). The goal is to stop the model from learning the background (white PNG vs black cloth vs calibration card). A leaf mask is computed per source (HSV thresholding for the white background; GrabCut seeded by Otsu and Lab chroma for the cloth; a hue seed plus GrabCut for the calibration card), cleaned with morphology and the largest connected component, and everything outside the leaf is painted black. Images are cropped to the leaf bounding box, padded to a square and saved as 256x256 PNG; the model reads them at 128x128. The first two masking attempts failed on the cloth images and were replaced; the final one was checked by eye on contact sheets. Known defects remain (small calibration-card fragments in some source C images, a little cloth residue in a few source A images).

### Split

[`src/02_veri_bolme.py`](src/02_veri_bolme.py). 70% train / 30% validation, grouped by near-duplicate group (256-bit dHash, Hamming distance <= 20) and stratified by (class, source). No samples were discarded to balance classes. The committed split has 0 shared groups, but that did not prevent leakage: consecutive burst frames of the same leaf in source A often differ by more than the dHash threshold and got groups of their own. The audit [`scripts/bolme_denetimi.py`](scripts/bolme_denetimi.py) found 145 capture-seconds (file names `IMG_<date>_<time>_<ms>`) on both sides, covering 299 of the 314 source-A validation images (40.5% of the validation set); with a 2-second burst window, 312 of 314 are affected. [`scripts/yakin_kopya.py`](scripts/yakin_kopya.py) now also merges source-A frames whose timestamps are at most 2 seconds apart (same source and class) into one group; the committed split and the reported models predate this fix and have not been rebuilt yet.

### Model

[`src/03_model.py`](src/03_model.py). No pretrained weights; He initialization. 1,772,995 trainable parameters.

| Layers | Description | Output |
|---|---|---|
| 1-3 | Conv 4x4, 32, BatchNorm, ReLU; MaxPool 2x2; Dropout 0.2 | 64x64x32 |
| 4-6 | Conv 4x4, 64, BN, ReLU; MaxPool; Dropout 0.2 | 32x32x64 |
| 7-9 | Conv 4x4, 128, BN, ReLU; MaxPool; Dropout 0.2 | 16x16x128 |
| 10-12 | Conv 4x4, 256, BN, ReLU; MaxPool; Dropout 0.2 | 8x8x256 |
| 13 | Conv 4x4, 256, BN, ReLU (Grad-CAM layer) | 8x8x256 |
| 14 | Global average pooling | 256 |
| 15 | Dense 128, ReLU, Dropout 0.3 | 128 |
| 16 | Dense 3 (softmax in the loss) | 3 |

Input is 128x128x3. Layers are counted Keras-style, with BatchNorm folded into the convolution layer.

### Training

[`src/04_egitim.py`](src/04_egitim.py).

| Setting | Value |
|---|---|
| Loss | Class-weighted cross-entropy, w = N/(K*n_c): 1.531 / 3.525 / 0.485 (CBB / halo / healthy) |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4, batch 32 |
| Schedule | ReduceLROnPlateau (x0.5 after 5 epochs without validation-loss improvement) |
| Stopping | Max 100 epochs; early stopping after 15 epochs without improvement; best validation loss kept |
| Augmentation (train only) | Flips, 0-360 degree rotation, random resized crop, brightness/contrast/saturation/hue jitter; background re-blackened after augmentation |
| Seed | 42, deterministic cuDNN |

It ran 50 epochs; the best epoch was 35 (validation loss 0.0627). Class-weighted loss and augmentation were preferred over undersampling the healthy class because the data is small and healthy comes from two sources.

![Training curves](ciktilar/sekiller/egitim_egrileri.png)

### Severity estimate

[`src/06_siddet_tahmini.py`](src/06_siddet_tahmini.py). **This is a rule-based estimate, not an expert label**; the datasets have no severity annotation. Within the eroded leaf mask, a pixel counts as lesion if it is reddish/brown (Lab a* - 128 > 4), yellow-orange (H < 27 and S > 60), or very dark (V < 45). The lesion ratio is split by two thresholds chosen by looking at the data: T1 = 0.5% (about the 99th percentile of healthy leaves) and T2 = 13% (median of the CBB and halo classes). Below T1 is "healthy", from T1 up to (not including) T2 "mildly diseased", at or above T2 "diseased".

Counts over the 2,451 images: healthy 1,716, mildly diseased 347, diseased 388. The measure depends on source (median lesion ratio: CBB source A 19.1%, CBB source C 1.9%), misses the diffuse chlorosis in Embrapa leaves, and finds no clear lesion in 41 of the 231 halo images (18%). Reflections, stems and edge shadows can produce false positives.

![Severity level examples](ciktilar/sekiller/siddet_duzeyi_ornekleri.png)

## Limitations

1. **Source-class confound (the main one).** Halo = 100% source B, CBB = 95% source A. Source is predictable at 100% even after masking, so the CBB/halo split cannot be attributed to disease. Unseen-source CBB accuracy is 20%.
2. **Burst-frame leakage in the committed split.** The dHash grouping kept every near-duplicate group on one side, but it did not catch burst frames of the same leaf: in source A, 145 capture-seconds are shared between train and validation, and 299 of the 314 source-A validation images (40.5% of all 738) have a same-second sibling in training (312 of 314 with a 2-second burst window). The reported 97.56% accuracy and 95.5% macro F1 are therefore optimistic, especially for source A, and should be read as upper bounds; the control runs and EfficientNet-B0 used the same split and are affected the same way. The grouping is now burst-aware and [`scripts/bolme_denetimi.py`](scripts/bolme_denetimi.py) checks for this; rebuilding the split and retraining is planned but not done. Sources B and C have no timestamps in their file names, so burst leakage there can be neither detected nor ruled out.
3. **Low confidence in the halo blight labels.** In a visual spot check during data preparation (a sample of images, not a measured rate), only about 20-30% of the halo folder looked like typical halo blight; most resembled CBB (large brown edge necrosis). No pathogen verification or methods paper was found for source B. Labels were left unchanged.
4. **Species uncertainty.** The "bean" leaves in sources A and B may not be *Phaseolus vulgaris*; only source C is confirmed.
5. **No separate test set**, a single seed and a single split; the validation accuracy fluctuated between 85% and 99% across epochs and variance was not measured.
6. **Few independent samples.** Halo blight has 231 images but 146 distinct leaves (44 groups in validation). Source A has 1,057 images, but combining the dHash groups with the 2-second burst window leaves only 110 groups (100 of them are bursts of 10 frames).
7. **Imperfect masking.** 9 images excluded; residual card fragments and cloth remain.
8. **The severity measure is rule-based**, source-dependent and not validated against experts.
9. **Grad-CAM is coarse** (8x8), and 32-45% of its weight falls outside the leaf.
10. **The Embrapa test is tiny:** 25 CBB images plus 2 healthy frames of the same leaf.

Not done: expert severity labels, multiple seeds or cross-validation, the unseen-source test for EfficientNet-B0, and a full end-to-end run of `calistir.ps1` (the scripts were run one by one in the same order).

## What I would do next

- Rebuild the split with the burst-aware groups, check it with `scripts/bolme_denetimi.py`, retrain all models and replace the reported numbers.
- Collect CBB and halo blight images from several sources per class, so that source and disease are no longer the same variable, and evaluate with leave-one-source-out.
- Get the halo blight labels reviewed by a plant pathologist, or replace that class with a verified one.
- Run the unseen-source test for every model, including the pretrained one, and report variance over several seeds.
- Hold out a real test set; use the validation set only for model selection.
- Use a higher-resolution CAM method and calibrate the severity rule against expert scores.

## Reproduce

### Changes since the reported run

Code fixes were made after the run that produced the reported numbers, so a fresh run will not reproduce them bit-for-bit:

1. Evaluation now re-applies the leaf mask after resizing, the same way training does.
2. The best checkpoint is now the true minimum validation loss (previously a new best required an improvement greater than 1e-4).
3. Grouping in [`scripts/yakin_kopya.py`](scripts/yakin_kopya.py) is now burst-aware (source-A frames at most 2 seconds apart share a group), so re-running it and `src/02_veri_bolme.py` produces a different split from the committed `data/bolme_*.csv`.

The numbers in [`ciktilar/metrikler.json`](ciktilar/metrikler.json) and in this README are from the original run and are kept unchanged.

### Setup

Python 3.13. The pinned [`requirements.txt`](requirements.txt) is a `pip freeze` of the environment used (CUDA 12.8 wheels of PyTorch). For a CPU-only setup, install `torch`/`torchvision` from `https://download.pytorch.org/whl/cpu` and the remaining packages by hand; [`requirements-ci.txt`](requirements-ci.txt) lists what the tests need.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The data is not distributed with this repository. The scripts in [`scripts/`](scripts) download and organize it (about 1.6 GB, no account needed); the argument is the project root:

```powershell
python -I scripts/indir.py . mendeley_ykvcrjffzd
python -I scripts/indir.py . mendeley_3km9d246z2
python -I scripts/indir.py . embrapa_pddb
python -I scripts/duzenle.py .        # verify, de-duplicate, build data/raw and data/etiketler.csv
python -I scripts/yakin_kopya.py .    # near-duplicate + burst groups (the `grup` column)
```

Pipeline order (or run `calistir.ps1`, which does the same; add `-EkModel` for EfficientNet-B0):

```powershell
python src/01_on_isleme.py
python src/02_veri_bolme.py
python -I scripts/bolme_denetimi.py .                     # leakage audit of the split (exit code 1 if leakage)
python src/04_egitim.py                                   # main model
python src/04_egitim.py --girdi siluet --ad siluet        # control: silhouette only
python src/04_egitim.py --gorev kaynak --ad kaynak        # control: source prediction
python src/04_egitim.py --haric_kaynak embrapa_pddb --ad embrapa_haric   # control: unseen source
python src/06_siddet_tahmini.py
python src/07_ek_efficientnet.py                          # optional
python src/05_degerlendirme.py                            # metrics, Grad-CAM, figures, metrikler.json
```

If the project path is very long, Windows may fail to load some DLLs ("file name too long"); move the project to a short path.

Hardware used: NVIDIA RTX 4070 Laptop GPU (8 GB), PyTorch 2.11.0+cu128. The main model trained in 1,009 s (while two control runs shared the GPU). Preprocessing time was not recorded. The seed is 42 and cuDNN is deterministic, so results should reproduce on the same hardware and library versions; small differences are possible elsewhere. The trained weights are not included.

Tests (no data or GPU needed): `pip install -r requirements-ci.txt` plus PyTorch, then `pytest -q`. They check the model's output shape and parameter count (1,772,995), the severity function and its thresholds, the grouped split, burst-aware grouping, the leakage audit, and background masking on synthetic images. One test runs the audit on the committed split and is marked as an expected failure (`xfail`) because of the known burst leakage; it is to be un-marked once the split is rebuilt.

## Data sources and licenses

- **A.** Rashid M. R. A., Hasan M., Gani R., Kamara R., Tarin T. K., Rabbi S. F. *Image Dataset for Bean and Cowpea Plant Leaf Disease Detection and Freshness Assessment from Bangladesh.* Mendeley Data, v1 (2024). doi:[10.17632/ykvcrjffzd.1](https://doi.org/10.17632/ykvcrjffzd.1). Paper: *Data in Brief* (2024), doi:[10.1016/j.dib.2024.111023](https://doi.org/10.1016/j.dib.2024.111023). License: CC BY 4.0.
- **B.** Mojumdar M. U., Dip S. D., Alahi T. E., Tonmoy S. *Bean Health and Disease Dataset: A Resource for Agricultural Research.* Mendeley Data, v3 (2025). doi:[10.17632/3km9d246z2.3](https://doi.org/10.17632/3km9d246z2.3). License: CC BY 4.0.
- **C.** Barbedo J. G. A. et al. *Embrapa PDDB: Base de Imagens de Sintomas de Doenças de Plantas.* Redape Dataverse. doi:[10.48432/XA1OVL](https://doi.org/10.48432/XA1OVL). Paper: Barbedo et al., *IEEE Latin America Transactions* 16(6), 2018, doi:[10.1109/TLA.2018.8444395](https://doi.org/10.1109/TLA.2018.8444395). License: CC BY-NC 4.0.

The figures in `ciktilar/sekiller/` contain images derived from these datasets (masked and cropped leaf photos). They are shared here with attribution, under the same license terms, for non-commercial purposes only. The CSV files in `data/` contain only metadata (file names, classes, sources, licenses, hashes, group IDs), no images.

The code is released under the [MIT License](LICENSE). That license does not extend to the dataset-derived images.

## How it was built

I built this project with AI-assisted development, using Claude Code as a pair programmer. I set the scope and the requirements from the course brief: the two diseases to separate from healthy leaves, a network designed from scratch rather than a pretrained one, a 70/30 train/validation split, and a way to show healthy, mildly diseased and diseased leaves. The AI did most of the implementation under that direction. The results, including the limitations above, are reported as they came out.