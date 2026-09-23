# PCINet

Reproduction and improvement of **PCINet** — an EfficientNetV2-S model for ordinal classification of building facade condition (Premise Condition Index, PCI 1–5) from street-level photos, used to identify high-risk urban areas for *Aedes aegypti* (Mosquito) infestation.

Based on: *Automatic mapping of high-risk urban areas for Aedes aegypti infestation based on building facade image analysis* (Laranjeira et al., PLOS One 2024).

---

## Results summary

| Model | MAE ↓ | Soft Acc ±1 ↑ | Balanced Acc ↑ |
|---|---|---|---|
| Paper target (Laranjeira et al.) | 0.66 | 0.8973 | — |
| Baseline (this repo) | 0.7109 | 0.8722 | 0.4270 |
| Run 5 — Attentive CutMix | 0.6365 | 0.9069 | 0.4474 |
| **Ensemble (Run 7 × 0.7 + GAIN AG × 0.3)** | **0.6228** | **0.9243** | 0.4287 |

The ensemble is the best result: **13% MAE improvement over the baseline**, exceeding both paper targets.

---

## Repository structure

```
notebooks/
  gradcam-pointing-games.ipynb   # Baseline training + GradCAM + Pointing Game analysis
  attentive-cutmix.ipynb         # Attentive CutMix augmentation experiments (Runs 1–7)
  GAIN.ipynb                     # GAIN Attention Mining + Attention Grounding experiments
  ensemble.ipynb                 # Checkpoint ensemble + TTA
  README.md                      # Notebook-level documentation
  requirements.txt               # Python dependencies

scripts/
  annotate.py                    # Two-pass annotation pipeline (Grounding DINO + YOLOv8m)

results/
  all_experiments_summary.csv    # All 11 experiments, all metrics
  Cutmix_runs_summary.csv        # CutMix runs (Baseline → Run 7)
  gain_results.csv               # GAIN AM and AG experiments
  ensemble_tta_results.csv       # Ensemble and TTA results
  Pointing_metrics_results.txt   # GradCAM Pointing Game hit rates
  raw_metrics/                   # Per-run raw CSV files from Kaggle
```

---

## Methodology

### Baseline
Re-implementation of PCINet with several improvements over the original: fixed train/val/test split (3755/804/806), `WeightedRandomSampler` for balanced mini-batches, class-weighted Focal Loss, `ReduceLROnPlateau` scheduler, and stronger regularisation (dropout 0.4, weight decay 1e-4).

### GradCAM + Pointing Game
GradCAM applied to all prediction errors. Relevant regions (building, wall, gate) annotated automatically with Grounding DINO; irrelevant objects (vehicles, people) with YOLOv8m. The Pointing Game revealed the baseline frequently attends to background regions — motivating the subsequent improvements.

### Attentive CutMix
Same-class CutMix augmentation where the patch is centred on the source image's attention peak (`model.features[-1]`). Patch size sampled from Beta(α, α) with α=2.0. At `CUTMIX_PROB=0.7`, both paper targets are surpassed for the first time (Run 5: MAE=0.6365, Soft Acc=0.9069).

### GAIN (Guided Attention Inference Network)
Two branches from Li et al. (CVPR 2018) were evaluated on top of Run 5:
- **Attention Mining (AM):** masks the most-attended region and adds a classification penalty. Caused class collapse at w=0.3 and degraded Balanced Accuracy at w=0.1. Closed.
- **Attention Grounding (AG):** supervises the attention map directly against ground-truth bounding boxes using `L_AG = -mean(log(A[Ω] + ε))`. Met paper targets at w=0.02 but did not improve over Run 5. Closed.

### Checkpoint Ensemble
Softmax probability averaging of Run 7 (×0.7) and GAIN AG w=0.02 (×0.3). Best result across all experiments: MAE=0.6228, Soft Acc=0.9243.

---

## Setup

```bash
pip install -r notebooks/requirements.txt
```

Notebooks were developed and run on [Kaggle](https://www.kaggle.com) (GPU T4 × 2). Dataset and model checkpoints are not included in this repository.

---

## Dataset

5,365 street-level building facade photos labelled with PCI 1–5 (1 = worst condition / highest mosquito risk, 5 = best). Collected by public health agents in Brazilian municipalities.

Split: Train = 3,755 | Val = 804 | Test = 806.

---

## References

- **Laranjeira et al. (2024)** — *Automatic mapping of high-risk urban areas for Aedes aegypti infestation based on building facade image analysis.* [https://doi.org/10.1371/journal.pntd.0011811]
