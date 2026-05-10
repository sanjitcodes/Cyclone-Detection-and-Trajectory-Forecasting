# 🌀 Cyclone Detection & Trajectory Forecasting using Deep Learning

> An end-to-end deep learning system for detecting cyclones in satellite imagery and forecasting their paths.

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![YOLOv7](https://img.shields.io/badge/YOLOv7-Object%20Detection-00B4D8?style=flat-square)](https://github.com/WongKinYiu/yolov7)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🎬 Demo

<!-- Replace YOUR_VIDEO_ID with the actual YouTube video ID after uploading -->
[![Cyclone Prediction Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v)

> *Click the thumbnail to watch the full demo video*

---

## 📌 Overview

India's 7,516 km coastline is struck by 5–6 tropical cyclones annually. Traditional detection relies on ground stations with limited spatial coverage. This project uses satellite imagery and sequential deep learning to address two distinct problems:

| Module | Approach | Key Result |
|---|---|---|
| **Cyclone Detection** | Fine-tuned YOLOv7 on satellite imagery | Test Precision: **0.699**, mAP: **0.584** |
| **Path Forecasting** | Bi-LSTM / GRU / RNN on IBTrACS dataset | Best 6-hour error: **40.87 km** |

---

## 📸 Results

### Cyclone Detection — YOLOv7 Output

<p align="center">
  <img src="assets\images\yolo-results\1.jpg" width="600" alt="YOLOv7 cyclone detection with bounding box and confidence score"/>
</p>

### Path Forecasting — Model Comparison

<p align="center">
  <img src="assets\images\results-visualized-summary\Km-data.png" width="600" alt="Distance wise model comparison (km) — RNN vs BiLSTM vs GRU"/>
</p>

### Predicted Path on India Map

<p align="center">
  <img src="assets\images\model-results-comparison\GRU212.png" width="600" alt="Cyclone path prediction plotted on India map using GeoPandas"/>
</p>

---

## 🏗️ System Architecture

```
                    DETECTION PIPELINE
    ┌─────────────┐    ┌──────────────────────────┐     ┌─────────────┐
    │  Satellite  │───▶│   Preprocessing          │───▶│   YOLOv7    │
    │   Images    │    │ Grayscale + Thresholding │     │  Detection  │
    └─────────────┘    └──────────────────────────┘     └─────────────┘
                              │                       │
                         3x Augment             Bounding Box
                         (LabelImg)             + Confidence

                    FORECASTING PIPELINE
    ┌─────────────┐    ┌──────────────────┐     ┌─────────────┐
    │  IBTrACS   │───▶│  Feature Extract  │───▶│ RNN/BiLSTM  │
    │  Dataset    │    │  + Normalisation │     │    / GRU    │
    └─────────────┘    └──────────────────┘     └─────────────┘
                       Lat, Lon (3-hr intervals)   36-hr forecast
```

---

## 📂 Repository Structure

```
Cyclone-Prediction/
│
├── 📁 code-notebooks/
│   ├── 01_data_preprocessing.ipynb      # Image preprocessing pipeline
│   ├── 02_bilstm_training.ipynb         # Bi-LSTM model training (500 epochs)
│   ├── 03_gru_training.ipynb            # GRU model training (500 epochs)
│   ├── 04_rnn_training.ipynb            # Simple RNN model training (500 epochs)
│   └── 05_inference_comparison.ipynb    # Inference and all-model comparison
│
├── 📁 datasets/
│   └──                                  # All of the data used as input
|
├── 📁 detection/
│   └── testing.py                       # YOLOv7 inference script
│
├── 📁 assets/
│   ├── 📁 images/                       # Result images
│   └── 📁 vidoes/                       # Demo video
│
├── 📁 docs/
│   ├── FY_Project.pdf
│   └── G22_Presentation_22-23.pptx
│
├── requirements.txt
└── README.md
```

---

## 🔬 Module 1 — Cyclone Detection (YOLOv7)

### Dataset
- **Source:** Kaggle satellite imagery dataset
- **Size:** 1,005 images — 804 train / 201 test (80:20 split)
- **Preprocessing:** Grayscale conversion → binary thresholding → manual bounding box annotation (LabelImg) → 3x augmentation
- **Format:** YOLO annotation, single class: `cyclone`

### Model & Training
- **Architecture:** YOLOv7 with E-ELAN backbone
- **Base:** Pre-trained on COCO (80 classes), fine-tuned for single-class cyclone detection
- **Hyperparameters:** 100 epochs · batch size 4 · learning rate 0.01 · momentum 0.937

### Results

| Metric | Training | Testing |
|---|---|---|
| Precision | 0.595 | **0.699** |
| Recall | 0.614 | 0.425 |
| mAP@0.5 | 0.612 | **0.584** |

---

## 📡 Module 2 — Path Forecasting (RNN / Bi-LSTM / GRU)

### Dataset
- **Source:** [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) — International Best Track Archive for Climate Stewardship (NOAA)
- **Scope:** North Indian Ocean · 60,458 rows · 1,776 unique cyclones
- **Filtered:** 1,439 cyclones with ≥18 timestamps used for training
- **Features:** Latitude, Longitude at 3-hour intervals (normalised ÷ 90)
- **Sequence split:** Input 23 timesteps → Output 12 timesteps (RNN & Bi-LSTM) · Input 20 → Output 15 (GRU)

### Models Compared

| Model | Layers | Activation | Parameters | Optimizer |
|---|---|---|---|---|
| Simple RNN | 4 | ReLU | 547,778 | Adam |
| Bi-LSTM | 3 | Tanh | 1,589,878 | Adam |
| GRU | 3 | Tanh | 598,574 | Adam |

All models trained for **500 epochs** · batch size 16 · learning rate 0.001 · MSE loss

### Distance Error Results (km) — lower is better

| Model | 6 hr | 12 hr | 18 hr | 24 hr | 36 hr |
|---|---|---|---|---|---|
| **Simple RNN** | **40.87** | **104.56** | 181.33 | 190.56 | 202.32 |
| Bi-LSTM | 138.54 | 143.33 | 165.44 | 170.39 | 212.43 |
| **GRU** | 119.90 | 127.05 | **162.27** | **165.32** | **197.10** |

**Key finding:** Simple RNN achieves the lowest error for short-horizon forecasts (6–12 hr). GRU generalises better for longer horizons (24–36 hr).

---

## ⚙️ Getting Started

### Prerequisites

```
Python 3.8+
CUDA-enabled GPU recommended (project trained on Kaggle T4 GPU)
```

### Installation

```bash
git clone https://github.com/sanjitcodes/Cyclone-Prediction.git
cd Cyclone-Prediction
pip install -r requirements.txt
```

### Running Detection (YOLOv7)

```bash
# Clone YOLOv7
git clone https://github.com/WongKinYiu/yolov7
cd yolov7

# Run inference on an image
python detect.py --weights best.pt --source <path_to_image>

# Evaluate on test set
python test.py --weights best.pt --data data.yaml --batch-size 4
```

### Running Path Forecasting

```bash
# Run notebooks in order
jupyter notebook notebooks/01_data_preprocessing.ipynb
jupyter notebook notebooks/02_bilstm_training.ipynb   # or 03 / 04 for GRU / RNN
jupyter notebook notebooks/05_inference_comparison.ipynb
```

> **Note:** Notebook 05 requires the three pre-trained `.h5` model files.
> Model weights are not included in the repo due to file size.
> Open an issue or reach out directly to request them.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Object Detection | YOLOv7, PyTorch, OpenCV |
| Sequence Modelling | Keras-TensorFlow (Bi-LSTM, GRU, SimpleRNN) |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Annotation | LabelImg |
| Geospatial Plotting | GeoPandas, Shapely |
| Visualisation | Matplotlib |
| Training Environment | Kaggle (T4 GPU), Google Colab |
| Datasets | Kaggle (satellite imagery), IBTrACS / NOAA |

---

## 📈 Key Engineering Challenges Solved

- **No annotated dataset existed** — manually annotated 1,005 satellite images using LabelImg, then applied 3x augmentation to address the dataset size constraint
- **Variable-length cyclone sequences** — handled with a custom mean-interpolation padding strategy, filtering out cyclones with fewer than 18 timestamps
- **Overfitting on small dataset** — addressed with dropout layers (rates 0.2–0.5) across all three forecasting models
- **GPU memory limits on free-tier Kaggle** — batch size tuned to 4 (detection) and 16 (forecasting) to stay within hardware constraints

---

## 📄 Documentation

| Resource | Link |
|---|---|
| Full Project Report | [`docs/FY_Project_Report.pdf`](docs/FY_Project_Report.pdf) |
| Presentation Slides | [`docs/Cyclone_Prediction_Presentation.pptx`](docs/Cyclone_Prediction_Presentation.pptx) |
| Demo Video | [YouTube](https://www.youtube.com/watch?v=YOUR_VIDEO_ID) |

---

## 👥 Authors

**Sanjit Anand** · **Siddharth Gautam** · **Somya Gupta**

B.Tech, Electronics & Communication Engineering
National Institute of Technology (NIT), Surat — 2023

*Guided by Dr. Prashant K. Shah, Associate Professor, DoECE, SVNIT*

---

## 📚 References

- Wang, C. et al. — [YOLOv7: Trainable Bag-of-Freebies](https://arxiv.org/abs/2207.02696)
- Knapp, K. et al. — [IBTrACS: Unifying Tropical Cyclone Best Track Data](https://doi.org/10.1175/2009BAMS2755.1)
- Redmon, J. & Farhadi, A. — [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)

---

⭐ If this project was useful, consider starring the repo!