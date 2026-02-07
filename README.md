# Road Anomaly Detection

This repository contains code, data and notebooks for detecting road anomalies (potholes, cracks, etc.) using YOLO-style models and associated data preparation and inference tooling.

---

## Data Understanding

This project relies on multiple road-damage datasets collected under different conditions and label standards. Each dataset serves a specific role in the pipeline (training, fine-tuning, or inference-only evaluation). Understanding their differences is critical for reproducibility and correct interpretation of results.

### Datasets Used
### 1. Road Damage Dataset (RDD – multiple versions)

Purpose: Primary dataset for model training and baseline experiments.

Usage in this repo:

- Core training for YOLO-based models

- Class-wise detection of road anomalies such as cracks and potholes

Datasets used:

- [RDD Dataset](https://github.com/sekilab/RoadDamageDetector): This repository contains all the required datasets.

- [RDD2022_India.zip](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip): Please click on this link to download the data or manually go to the [RDD](https://github.com/sekilab/RoadDamageDetector)  repository and download `RDD2022_India.zip`

- [Japan/India/Czech Training Data](https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/IEEE_bigdata_RDD2020/train.tar.gz): Please click on this link and download the dataset which contains `Japan/India/Czech` countries' training data or manually go to the [RDD](https://github.com/sekilab/RoadDamageDetector) repository and download `train.tar.gz` under `Dataset for GRDDC 2020`

- [Japan/India/Czech Test Data](https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/IEEE_bigdata_RDD2020/test1.tar.gz): Please click on this link and download the dataset which contains `Japan/India/Czech` countries' test data or manually go to the [RDD](https://github.com/sekilab/RoadDamageDetector) repository and download `test1.tar.gz` under `Dataset for GRDDC 2020`


#### Key characteristics:

- Collected from dashcam-style vehicle-mounted cameras

- Images captured across multiple countries and road conditions

- Annotations provided in bounding-box format

- Standard road damage classes (e.g., longitudinal crack, transverse crack, alligator crack, pothole)

#### Why RDD:

- Well-established benchmark for road anomaly detection

- Matches the target deployment scenario (dashcam → edge device)

- Suitable for YOLO-style object detection pipelines

### 2. RDD2022ES (Kaggle)

Purpose: Fine-tuning and domain adaptation

Usage in this repo:

- Fine-tuning YOLOv8 models

- Quantization and TFLite conversion experiments

[RDD2022ES Data](https://www.kaggle.com/datasets/juusos/rdd2022es?resource=download): Please download this data which has been used in this repository for fine-tuning YOLOv8s model which has been trained on `three country data - Japan/India/Czech`.

#### Key characteristics:

- Part of the RDD 2022 challenge ecosystem

- Improved annotation consistency compared to earlier RDD versions

- Higher-quality labels for certain damage classes

- Suitable for downstream optimization (INT8 quantization, edge inference)

#### Why RDD2022ES:

- It has better label quality than RDD2022 which has drastically improved model performance

- Ideal for refining a pre-trained detector

### 3. Indian Driving Dataset (IDD)

Purpose: Inference-only testing and generalization analysis

Usage in this repo:

- Evaluating trained models on unseen, real-world Indian road scenes

- Stress-testing robustness under different traffic and road conditions

[IDD Data](https://idd.insaan.iiit.ac.in/): Please visit this link and create an account to download the data. Once you are logged in, please go to the `Download` section from the navigation bar and download the following manually:

-  IDD Temporal Test - Part I (10.8 GB)

#### Key characteristics:

- Captured in complex Indian traffic environments

- High scene diversity (lighting, road quality, occlusion)

- Not originally designed specifically for road-damage detection

- Used without fine-tuning to assess generalization

### Why IDD:

- Represents real-world deployment conditions more accurately

- Helps evaluate model robustness beyond training distributions

- Useful for identifying failure modes before edge deployment

## Annotation Formats

- All training datasets are converted into YOLO-compatible format:

- One .txt file per image

- Normalized bounding box coordinates

- Class mappings are kept consistent across datasets where possible

- Any incompatible or missing classes are either:

- Mapped to a reduced label set, or Ignored during training/inference (documented per notebook)

---

## Quick Start

### Install Python dependencies

This project uses a standard Python virtual environment and a pinned `requirements.txt`.

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Alternative: using `uv` (faster, more robust)

You may optionally use the **`uv` package manager** for faster and more reliable dependency resolution, especially on larger environments:

```bash
uv venv .venv-uv
source .venv-uv/bin/activate
uv pip install -r requirements.txt
```

**Important notes when using `uv`:**

* A **separate virtual environment is recommended** (e.g., `.venv-uv`)
* Some toolchains (e.g., **ONNX / ONNX Runtime**, TensorRT-related tooling) may require a **different venv** due to binary compatibility and system-level dependencies
* Avoid mixing `pip` and `uv` installs inside the same virtual environment


### Common tips

* Use **Python 3.10+**
* GPU support requires a compatible **PyTorch + CUDA** build
  (install Torch and TorchVision according to your CUDA version)
* Verify CUDA availability with:

  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```

---


## Repository top-level (short) directory structure


```
├── data
│   ├── combined_annotatedv2/              # Extracted RDD2022ES dataset
│   ├── idd_temporal_test_1/               # Extracted IDD temporal test set
│   └── videos/                            # Place test videos for inference here
│
│   ├── India/                             # Extracted RDD2022_India.zip
│   │   ├── train/
│   │   └── test/
│
│   ├── train/                             # Extracted train.tar.gz (Japa/India/Czech)
│   │   ├── Czech
│   │   └── India
│   │   └── Japan
│   │
│   ├── test1/                             # Extracted test1.tar.gz (Japa/India/Czech)
│   │   ├── Czech
│   │   └── India
│   │   └── Japan
│
├── logs/                                  # Created at runtime (training/inference logs)
├── runs/                                  # Created at runtime (YOLO outputs)
│
├── src
│   ├── fine_tuning
│   │   └── rd22                           # RDD2022ES fine-tuning experiments
│   │       ├── inspect_data.ipynb
│   │       ├── train_model.ipynb
│   │       ├── inference.ipynb
│   │       ├── inference_tflite_int8.ipynb
│   │       ├── yolov8s_fine_tuned_tflite.ipynb
│   │       ├── yolov8s_fine_tuned_quantized_tflite.ipynb
│   │
│   ├── India_data_only                    # India-only dataset experiments
│   │   ├── india_data_understanding.ipynb
│   │   ├── standby_yolo_models.ipynb
│   │   ├── yolo_model.ipynb
│   │   ├── yolov8s_binary_model.ipynb
│   │
│   ├── three_countries                    # Multi-country training & inference
│   │   ├── data_preparation.ipynb
│   │   ├── inference_nano.ipynb
│   │   ├── inference_small.ipynb
│   │   ├── inference-medium.ipynb
│   │   ├── inference_idd_yolov8s.ipynb
│   │   ├── inferencee_idd_time_based.ipynb
│   │   ├── inference_yolov26s.ipynb
│   │   ├── inference_on_video.ipynb
│   │   │
│   │   ├── training_script.sh
│   │   ├── yolo_nano.py
│   │   ├── yolo_small.py
│   │   ├── yolo_small_26.py
│   │   ├── yolo_medium.py
│
├── README.md
└── requirements.txt
```


---

## High-level Repository Structure

> **Note:** This is a high-level view. See subfolders and notebooks for implementation details.

### Top-level folders

* **`data/`**
  Central location for all datasets used in this project.
  After downloading datasets, **extract them directly into this folder**.

  Expected contents:

  * `combined_annotatedv2/` — extracted **RDD2022ES** dataset (used for fine-tuning)
  * `idd_temporal_test_1/` — extracted **IDD** dataset (used for inference/testing)
  * `videos/` — place any custom test videos here for video-based inference

* **`India/`**
  Extracted **RDD2022_India** dataset, structured into:

  * `train/`
  * `test/`
    Used for India-only training experiments.

* **`src/`**
  Main source folder containing all notebooks and scripts.
  Experiments are organized by dataset scope and project phase:

  * `India_data_only/` — India-only experiments
  * `three_countries/` — multi-country training and inference
  * `fine_tuning/rd22/` — RDD2022ES fine-tuning and model optimization

* **`logs/`**
  Automatically created during runtime. Stores training and inference logs.

* **`runs/`**
  Automatically created YOLO output directory (checkpoints, metrics, predictions).

* **`requirements.txt`**
  Pinned Python dependencies for reproducibility.

---

## Where to Place Datasets

After downloading datasets, extract them as follows:

```text
data/
├── combined_annotatedv2/        # RDD2022ES (fine-tuning dataset)
├── idd_temporal_test_1/         # IDD (inference/testing only)
├── videos/                      # Optional video inputs
│
India/
├── train/                       # RDD2022 India training split
└── test/                        # RDD2022 India test split
│                    
train/
├── Czech/                       # train.tar.gz Czech training split
└── India/                       # train.tar.gz India training split
└── Japan/                       # train.tar.gz Japan training split
│
test1/
├── Czech/                       # test1.tar.gz Czech training split
└── India/                       # test1.tar.gz India training split
└── Japan/                       # test1.tar.gz Japan training split


```

No dataset files should be placed directly inside `src/`.

---

## Notebook Execution Order and Purpose

This project was executed **iteratively**, improving model performance step by step as limitations were identified. Below is the **recommended execution order**, along with an explanation of what each notebook does and *why it exists*.


### Phase 1: India-only experiments

**Goal:** Train a road anomaly detector using only India data.

📁 `src/India_data_only/`

1. **`india_data_understanding.ipynb`**

   * Exploratory data analysis on the India dataset
   * Class distribution analysis
   * Identification of severe class imbalance

2. **`yolo_model.ipynb`**

   * Multi-class (5-class) YOLO training on India dataset
   * Result: **Poor performance** due to limited data and class imbalance

3. **`yolov8s_binary_model.ipynb`**

   * Reformulated the task as **binary classification** (defect vs non-defect)
   * Reduced label complexity to improve learning stability
   * Result: Performance improved slightly but still below desired accuracy

4. **`standby_yolo_models.ipynb`**

   * Trained three **standby binary models**:

     * Nano
     * Small
     * Medium
   * These models served as fallbacks and baselines for later comparison

➡️ **Outcome:**
India-only data was insufficient for robust performance. This motivated the move to a larger, more diverse dataset with binary classification.

---

### Phase 2: Three-country training and inference

**Goal:** Improve generalization by increasing data diversity.

📁 `src/three_countries/`

1. **`data_preparation.ipynb`**

   * Merged datasets from **India, Japan, and Czech Republic**
   * Unified label schema and YOLO-compatible formatting

2. **Model training scripts**

   * `training_script.sh`
   * `yolo_nano.py`
   * `yolo_small.py`
   * `yolo_medium.py`
     Trained three models (nano, small, medium) on the combined dataset.

3. **Inference notebooks**

   * `inference_nano.ipynb`
   * `inference_small.ipynb`
   * `inference-medium.ipynb`
   * `inference_yolov26s.ipynb`
   * `inference_idd_yolov8s.ipynb`
   * `inferencee_idd_time_based.ipynb`
   * `inference_on_video.ipynb`

   These notebooks:

   * Evaluate trained models on IDD
   * Compare performance across model sizes
   * Test inference on both images and videos

➡️ **Outcome:**
Performance improved to **slightly above average**. Among the models, the **small model** consistently performed best.

---

### Phase 3: Fine-tuning on RDD2022ES (final optimization)

**Goal:** Improve precision, recall, and robustness using higher-quality labels.

📁 `src/fine_tuning/rd22/`

1. **`inspect_data.ipynb`**

   * Analyzed RDD2022ES label quality
   * Identified improved annotations compared to original RDD2022

2. **`train_model.ipynb`**

   * Fine-tuned the **best-performing small model**
   * Trained on ~48,000 images from RDD2022ES

   **Why fine-tuning was necessary:**

   * Original RDD2022 labels caused false positives
   * Road lane markings were frequently detected as defects
   * RDD2022ES significantly reduced this noise

3. **`inference.ipynb`**

   * Post-fine-tuning evaluation
   * Achieved ~ **>70% precision, recall, and mAP**

4. **Deployment preparation**

   * `yolov8s_fine_tuned_tflite.ipynb`
     → Exported YOLOv8s model to **TFLite** (standby for Raspberry Pi)
   * `inference_tflite_int8.ipynb`
   * `yolov8s_fine_tuned_quantized_tflite.ipynb`
     → Quantized INT8 TFLite model for efficient edge inference

➡️ **Final Outcome:**
A high-quality, fine-tuned YOLOv8s model with strong accuracy and an optimized **INT8 TFLite version** suitable for deployment on Raspberry Pi–class devices.

---

## How to run a notebook (local)

1. Activate your virtual environment (see above).
2. From repo root, start jupyter lab or notebook:

   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

3. Open the notebook file in the UI and run cells in the suggested order once you fill the execution-order placeholders.

---
## Contributing

- If you add notebooks, please add them to this README's list with a short description and suggested order.

---
## License

This project is under MIT License


