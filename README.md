# glaucoma-screen

A convolutional neural network (CNN) with 2.79 million parameters that scores color fundus photographs (photos of the back of the eye) for referable glaucoma, as defined by the JustRAIGS dataset: eyes graded as having signs that warrant referral to a specialist.

> **This software is a research prototype and has not been clinically validated.**
> Model predictions are intended for research purposes only and must not be used for clinical diagnosis, patient screening, or treatment decisions.

## How it works

- **Input:** one RGB fundus photograph, resized to 128 × 128 pixels and normalized with ImageNet mean and standard deviation.
- **Output:** one number per image, passed through a sigmoid to give a score between 0 and 1.
- **Decision rule:** scores at or above 0.25 are flagged as referable glaucoma. The threshold was picked from a sweep of values on the validation split (see Results).
- **Training label:** the JustRAIGS classification, RG (referable glaucoma) or NRG (no referable glaucoma). The model predicts only this label. It does not detect or localize individual signs.
- **Network:** `CompactGlaucomaNet`, a ResNet-style CNN trained from scratch.

## Model and training

| | |
|---|---|
| Parameters | 2,793,953 |
| Training data | JustRAIGS: 101,368 images (64.7 GB) |
| Positive class | 3,269 images (3.22%) |
| Split | 85% / 15%, stratified, seed 42: 86,164 train (2,779 positive), 15,204 validation (490 positive) |
| Augmentation | Random horizontal flip, rotation up to 10°, brightness and contrast jitter of 0.1 |
| Training | 30 epochs, batch 32, AdamW (lr 0.001), cosine schedule |
| Loss | Weighted binary cross-entropy (positive weight 30.01) |
| Hardware | Cloud AMD Instinct MI300X, PyTorch with ROCm |

The JustRAIGS label file lists 101,423 labelled images. 55 were missing from the download.

## Results

### Internal: JustRAIGS validation split

Best validation AUC 0.9396, at epoch 27. The 0.25 threshold was chosen using this same split, so the figures at 0.25 are not an independent estimate.

| Threshold | Sensitivity | Specificity |
|---:|---:|---:|
| 0.10 | 96.73% | 62.20% |
| 0.20 | 94.49% | 74.92% |
| **0.25** | **93.27%** | **78.56%** |
| 0.30 | 90.00% | 81.28% |
| 0.50 | 84.08% | 88.60% |

### External: REFUGE

The 400-image REFUGE training subset (40 glaucoma, 360 non-glaucoma), evaluated with the model and the 0.25 threshold frozen, on CPU.

| Metric | Result | 95% confidence interval |
|---|---|---|
| AUC | 0.8973 | 0.8255–0.9591 (bootstrap) |
| Sensitivity | 82.5% (33 / 40) | 67.2–92.7% (Clopper-Pearson) |
| Specificity | 88.6% (319 / 360) | 84.9–91.7% (Clopper-Pearson) |

| | Predicted glaucoma | Predicted non-glaucoma |
|---|---:|---:|
| **Glaucoma** | 33 | 7 |
| **Non-glaucoma** | 41 | 319 |

Compared with the internal split, external AUC and sensitivity are lower and specificity is higher. With only 40 positive cases the intervals are wide, and the reasons for the differences have not been established.

### Missed cases

The 7 false negatives score 0.0058, 0.0093, 0.0245, 0.0517, 0.0579, 0.0623 and 0.0827. The lowest threshold tested, 0.10, would recover none of them, and recovering all of them would need a threshold of 0.005796 or lower. The cause of these misses has not been established.

### Calibration

On REFUGE: Brier score 0.0585, expected calibration error 0.0615, with overconfidence in every probability bin. The outputs are uncalibrated scores, not probabilities of disease. No recalibration was applied.

Per-image REFUGE predictions are in `models/checkpoints/external_val_refuge_predictions.csv`.

## Limitations

- One external test set: the REFUGE training split (400 images, 40 glaucoma cases), not its official test split.
- REFUGE has 10% glaucoma and JustRAIGS has 3.22%. The share of flagged images that are truly glaucoma depends on this rate, so that figure is not reported.
- AUC here is not comparable to JustRAIGS challenge results, which use sensitivity at 95% specificity on a private test set.

## Planned work

Further analysis of the seven missed glaucoma cases.

## Weights

Trained weights are not distributed. This model has not been clinically validated: external sensitivity is 82.5%, seven glaucoma cases were missed at very low scores, and the scores are overconfident. The repository is for transparency and reproducibility. To discuss the work or the results, open an issue or discussion on this repository.

## Reproducing

Obtain JustRAIGS and REFUGE from their owners; neither is included here. Install a PyTorch build for your hardware, then the packages in `requirements.txt`. Training code is under `src/` and the evaluation and analysis scripts are in `scripts/`. REFUGE labels come from the `g` / `n` filename prefix.

## Built with

[PyTorch](https://pytorch.org/) and torchvision (trained with the ROCm build on AMD), [scikit-learn](https://scikit-learn.org/), [SciPy](https://scipy.org/), [NumPy](https://numpy.org/) and [Pillow](https://python-pillow.org/).

## License

Code: MIT. The datasets remain under their own terms.

## Sources

- Madadi Y, et al. JustRAIGS: Justified Referral in AI Glaucoma Screening Challenge. IEEE Trans Med Imaging 2026;45(1):320–335. doi:[10.1109/TMI.2025.3596874](https://doi.org/10.1109/TMI.2025.3596874). Data: [Zenodo 10035093](https://zenodo.org/records/10035093).
- Orlando JI, Fu H, Breda JB, et al. REFUGE Challenge: A unified framework for evaluating automated methods for glaucoma assessment from fundus photographs. Medical Image Analysis 2020;59:101570. doi:[10.1016/j.media.2019.101570](https://doi.org/10.1016/j.media.2019.101570).
