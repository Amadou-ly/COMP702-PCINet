# Bridging the Gap between AI and Monitoring Mosquito-borne Diseases
Source Code & Supplementary Material

All experiments throughout this project were implemented entirely using Jupyter notebooks; 
Each notebook is self-contained, parameterized with configurable paths, and can be executed independently.


## 1. Project Structure & Notebooks Overview

| Notebook Filename | Description | Key Outputs |
| `gradcam-pointing-games.ipynb` | Baseline model with GradCAM analysis, and Pointing Game validation. | `test_predictions.csv`, `test_metrics.csv`, `pointing_game_results.csv`, `debug_overlays.zip` |
| `attentive-cutmix.ipynb` | Implementation of CutMix. TTA can optionally be ran using the TTA_ENABLED variable. | `test_predictions.csv`, `training_curves.pdf`, `cutmix_A_examples.zip`, `cutmix_B_examples.zip` |
| `GAIN.ipynb` | Implementation of the 2 GAIN branches (AG and AM) using the variables AM_WEIGHT and AG_WEIGHT. TTA can optionnally be ran using the TTA_ENABLED variable. | `test_predictions.csv`, `test_metrics.csv`, `confusion_matrices.txt`, `training_curves.pdf` |
| `ensemble.ipynb` | Ensemble combining softmax distributions from Run 7 and GAIN AG checkpoints via weight sweeping. | `ensemble_sweep.csv`, `ensemble_best_metrics.csv`, `ensemble_confusion_matrix.txt` |


## 2. Requirements

All requirements for libraries are listed in `requirements.txt`.


## 3. Configuration & Path Mapping

All notebooks contain a parameterized configuration cell near the top. These variables can be adjusted for file paths and hyperparameters setting:

| Variable | Description |
| `DATASET_DIR` | Root directory containing image subfolders and CSV split files. |
| `IMAGES_DIR` | Directory containing facade images. |
| `TRAIN_CSV`, `VAL_CSV`, `TEST_CSV` | Split index CSVs containing columns: `name`, `label`. |
| `OUTPUT_DIR` | Directory where all outputs are written. |
| `SAVE_PATH` | Checkpoint Path|
| `ANNOT_PATH` | CSV with object bounding boxes (`image_name`, `box_type`, coordinates). |
| `TRAIN_ANNOT_PATH` | CSV containing bounding boxes used for GAIN AG training loss. |
| `MODEL_A`, `MODEL_B` | Checkpoints Paths for Ensembles. |
| `EPOCHS` | Maximum number of epochs. |
| `PATIENCE` | Number of epochs with no validation improvement before early stopping triggers. |
| `BATCH_SIZE` | Number of image samples processed per optimization step. |
| `LR` | Initial learning rate for the Adam optimizer. |
| `WEIGHT_DECAY` | L2 regularization weight penalty. |
| `FOCAL_GAMMA` | Focusing parameter in Focal Loss. |
| `DROPOUT` | Dropout probability applied before the linear classification layer. |
| `CUTMIX_PROB` | Probability of applying Attentive CutMix (set to 0 to disable). |
| `CUTMIX_ALPHA` | Beta distribution parameter ($\alpha$) used to sample the CutMix patch dimensions. |
| `ORDINAL_WEIGHT` | Penalty weight for the expected-class MAE loss added to the Focal Loss objective. |
| `AM_WEIGHT` | Loss weight for GAIN Attention Mining. |
| `AG_WEIGHT` | Loss weight for GAIN Attention Grounding. |


All the notebooks expect the `DATASET_DIR` directory to contain the images alongside the CSV split files like shown below:
```
DATASET_DIR/
├── images/                  # Directory containing all raw facade images referenced by CSVs
│   ├── image_001.jpg
│   └── ...
├── train.csv                # Training split: columns ['name', 'label']
├── val.csv                  # Validation split: columns ['name', 'label']
└── test.csv                 # Evaluation split: columns ['name', 'label']```

## 4. Execution Flags & Workflow

Each training notebook includes these execution flags

SKIP_TRAINING : Bypasses model training and loads the checkpoint defined in `SAVE_PATH` when set to True. Retrains the model when set to False

TTA_ENABLED : When enabled, the model evaluates both the original test image and its horizontal reflection, averaging their softmax probability.
