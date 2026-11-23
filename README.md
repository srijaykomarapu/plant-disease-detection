# Plant Disease Detection Using EfficientNetB0

An end-to-end computer vision and deep learning project for automated plant leaf disease diagnosis and classification using **EfficientNetB0**, transfer learning, two-stage fine-tuning, and robust evaluation metrics.

---

## 1. Project Detailed Description

### 1.1 Overview & Objective
This project automates the identification and classification of plant leaf diseases from digital imagery using deep convolutional neural networks. Crop diseases present major risks to agricultural yield, food security, and economic sustainability. By leveraging transfer learning on ImageNet-pretrained weights and fine-tuning higher-level representations, the model achieves high discriminative power across various crop-disease pairs without requiring training a large vision backbone from scratch.

### 1.2 Pipeline & Technical Workflow

```
Raw PlantVillage Images (Train / Val / Test)
                     │
                     ▼
  [ Data Augmentation Layer (Flip, Rotation, Zoom) ]
                     │
                     ▼
   [ EfficientNetB0 Backbone (Pretrained ImageNet) ]
                     │
                     ▼
  [ Custom Classification Head (GAP -> BN -> Dropout -> Softmax) ]
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
[ Stage 1: Feature Extraction ] [ Stage 2: Fine-Tuning ]
(Base Frozen, lr=0.001, 10 ep)   (Top 30 Layers Unfrozen, lr=1e-5, 10 ep)
       │                           │
       └─────────────┬─────────────┘
                     ▼
           [ Saved Final Model ]
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
 [ Test Set Evaluation ]   [ Single Leaf Inference ]
 - Accuracy, Precision,     - Confidence Threshold (80%)
   Recall, Macro F1         - Health Status (Healthy / Diseased)
 - Classification Report
 - Confusion Matrix Heatmap
```

The pipeline operates in clearly delineated phases:

1. **Reproducibility & Hardware Initialization:**
   - Global deterministic seeds (`SEED = 42`) set for TensorFlow, NumPy, and Python's random module.
   - GPU memory growth dynamically enabled across physical GPUs to prevent Out-Of-Memory (OOM) allocations.
   - Strict dataset directory verification for `train`, `val`, and `test` subdirectories.

2. **Data Ingestion & Augmentation:**
   - Datasets are loaded in batches (`BATCH_SIZE = 32`) resized to standard dimensions (`IMAGE_SIZE = (224, 224)`).
   - Labels are one-hot encoded (`label_mode="categorical"`).
   - Dynamic data augmentation applied on input tensors:
     - Horizontal and vertical flips: `RandomFlip("horizontal_and_vertical")`
     - Rotation: `RandomRotation(0.15)`
     - Zoom: `RandomZoom(0.15)`
   - Datasets prefetch batches with `tf.data.AUTOTUNE` to eliminate I/O bottlenecks.

3. **Model Architecture & Two-Stage Training:**
   - **Backbone**: EfficientNetB0 with ImageNet pretrained weights (top layers excluded). Input rescaling is natively handled inside the EfficientNetB0 architecture.
   - **Custom Head**:
     - `GlobalAveragePooling2D`: Flattens spatial feature maps into a 1D feature vector.
     - `BatchNormalization`: Normalizes activations to stabilize gradient flow.
     - `Dropout(0.3)`: Regularization to minimize overfitting.
     - `Dense(num_classes, activation="softmax")`: Outputs class probability distributions.
   - **Stage 1 (Feature Extraction)**:
     - Pretrained backbone weights are frozen (`base_model.trainable = False`).
     - Only the custom head is trained using Adam optimizer with `learning_rate = 0.001` for 10 epochs.
     - Checkpointed locally as `efficientnetb0_stage1.keras`.
   - **Stage 2 (Fine-Tuning)**:
     - Base model is unfrozen, and all except the final 30 layers are locked (`base_model.layers[:-30].trainable = False`).
     - All `BatchNormalization` layers inside the backbone remain explicitly frozen to preserve learned mean/variance statistics.
     - Model is re-compiled using Adam optimizer with a lower learning rate (`learning_rate = 1e-5`) for 10 epochs.
     - Checkpointed as `efficientnetb0_plant_disease.keras`.

4. **Evaluation & Visualization:**
   - Comprehensive model evaluation on the independent `test` dataset.
   - Computes:
     - Test Accuracy
     - Macro Precision, Macro Recall, and Macro F1 Score (`zero_division=0`).
     - Per-class precision, recall, and f1-score printed and saved to `classification_report.txt`.
     - Multiclass Confusion Matrix visualized via `seaborn.heatmap` and saved to `confusion_matrix.png`.
     - Concatenated training history curves (accuracy and loss across both stages) saved to `training_history.png`.

5. **Inference & Decision Logic:**
   - `predict_single_leaf`: Preprocesses a single image, evaluates Softmax probabilities, and checks against a confidence threshold (`threshold = 80.0%`).
   - If confidence is below 80.0%, flags as `"Low Confidence / Uncertain"`.
   - If confidence is $\ge 80.0\%$, parses category label and outputs:
     - `predicted_disease`: Class label
     - `health_status`: `"Healthy"` if `"healthy"` is present in class name, otherwise `"Diseased"`
     - `confidence`: Formatted percentage string.

---

## 2. Project Directory Structure

```text
plant-disease-detection/
│
├── src/                                  # Core source code package
│   ├── __init__.py                       # Python package marker
│   ├── utils.py                          # Global configuration, constants, seeds, GPU setup, JSON & plotting
│   ├── data_preprocessing.py             # Data augmentation sequential model & dataset batch loader
│   ├── model.py                          # EfficientNetB0 model builder with custom classification head
│   ├── fine_tune.py                      # Stage 2 fine-tuning logic (layer freezing & low lr optimizer)
│   ├── train.py                          # Two-stage training orchestrator & model saving
│   ├── evaluate.py                       # Test set evaluation, metrics calculation & confusion matrix
│   └── predict.py                        # Single image inference & automated test set sample prediction
│
├── main.py                               # Master script to run Train -> Evaluate -> Predict sequentially
├── requirements.txt                      # Project library dependencies
├── .gitignore                            # Rules for excluding datasets, models, artifacts, and cache
└── README.md                             # Comprehensive project documentation
```

### File-by-File Breakdown:

| File | Purpose / Role | Key Functions & Components |
| :--- | :--- | :--- |
| `src/utils.py` | Central configuration & helper routines | `DATASET_DIR`, `OUTPUT_DIR`, `IMAGE_SIZE`, `BATCH_SIZE`, `SEED`, `set_seeds()`, `setup_gpu()`, `verify_dataset_dirs()`, `save_class_names()`, `load_class_names()`, `plot_training_history()` |
| `src/data_preprocessing.py` | Data preparation & augmentation | `data_augmentation` (flip, rotation, zoom), `load_datasets()` (creates `train_ds`, `val_ds`, `test_ds` with prefetching) |
| `src/model.py` | Neural network architecture definition | `build_model()` (constructs EfficientNetB0 + GlobalAveragePooling2D + BatchNorm + Dropout + Softmax head) |
| `src/fine_tune.py` | Transfer learning fine-tuning | `fine_tune_model()` (unfreezes top 30 layers, locks BatchNorm, trains with Adam `lr=1e-5`) |
| `src/train.py` | Model training orchestrator | `train()` (manages Stage 1 training, Stage 2 fine-tuning, model checkpointing, history plotting) |
| `src/evaluate.py` | Performance assessment & metrics | `evaluate_model()` (predicts test set, computes accuracy, macro precision/recall/F1, outputs report & confusion matrix) |
| `src/predict.py` | Inference on new or sample images | `predict_single_leaf()` (confidence thresholding, health status tag), `predict_sample()` (automated test sample lookup) |
| `main.py` | Root pipeline execution entrypoint | Runs `train()`, `evaluate_model()`, and `predict_sample()` in sequence |
| `requirements.txt` | Package dependencies | Lists `tensorflow`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` |
| `.gitignore` | Git exclusion rules | Prevents committing heavy datasets (`PlantVillage/`), trained weights (`*.keras`), results, and caches |

---

## 3. Dataset Setup & Directory Layout

The dataset must be arranged in the standard classification directory structure with subfolders per class:

```text
PlantVillage/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── Apple___Cedar_apple_rust/
│   ├── Apple___healthy/
│   └── ...
├── val/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── Apple___Cedar_apple_rust/
│   ├── Apple___healthy/
│   └── ...
└── test/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    ├── Apple___Cedar_apple_rust/
    ├── Apple___healthy/
    └── ...
```

---

## 4. How to Execute

### Step 1: Clone the Repository & Navigate to Directory
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd plant-disease-detection
```

### Step 2: Set Up Virtual Environment (Recommended)
**On Linux / macOS / WSL:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Dataset Paths (Optional)
By default, the code looks for the dataset at:
- `DATASET_DIR`: `/mnt/c/Users/chait/Downloads/PlantVillage`
- `OUTPUT_DIR`: `~/PlantVillage_Results`

You can override these paths without modifying the code by setting environment variables:

**On Linux / macOS / WSL:**
```bash
export DATASET_DIR="/path/to/your/PlantVillage"
export OUTPUT_DIR="/path/to/your/results"
```

**On Windows (PowerShell):**
```powershell
$env:DATASET_DIR = "C:\path\to\your\PlantVillage"
$env:OUTPUT_DIR = "C:\path\to\your\results"
```

---

### Step 5: Run the Project

#### Option A: Run the Complete End-to-End Pipeline
Executes the full workflow: dataset verification, Stage 1 training, Stage 2 fine-tuning, training history plotting, test evaluation with confusion matrix, and sample image prediction.
```bash
python main.py
```

#### Option B: Run Modular Steps Independently

1. **Train Model Only (Stage 1 + Stage 2):**
   ```bash
   python -m src.train
   ```
   *Trains the model, generates `efficientnetb0_stage1.keras`, `efficientnetb0_plant_disease.keras`, `efficientnetb0_classes.json`, and `training_history.png`.*

2. **Evaluate the Trained Model:**
   ```bash
   python -m src.evaluate
   ```
   *Loads the saved model, runs evaluation over the test dataset, prints classification report, and exports `classification_report.txt` and `confusion_matrix.png`.*

3. **Run Leaf Prediction / Sample Inference:**
   ```bash
   python -m src.predict
   ```
   *Scans the test directory for sample leaf images and outputs the predicted disease, health status, and confidence percentage.*

---

## 5. Generated Outputs & Saved Artifacts

All outputs are saved directly to the configured `OUTPUT_DIR`:

- `efficientnetb0_stage1.keras`: Checkpoint after Stage 1 feature extraction.
- `efficientnetb0_plant_disease.keras`: Final fine-tuned model checkpoint after Stage 2.
- `efficientnetb0_classes.json`: List of class names sorted by class index.
- `training_history.png`: Plots showing training vs. validation accuracy and loss over all 20 epochs.
- `classification_report.txt`: Macro and per-class precision, recall, and F1-score.
- `confusion_matrix.png`: High-resolution Seaborn heatmap showing true vs. predicted classes.
