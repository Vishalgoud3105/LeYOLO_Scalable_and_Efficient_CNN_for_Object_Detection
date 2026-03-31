# 🔍 LeYOLO — Lightweight & Efficient Object Detection

> A scalable CNN architecture for real-time object detection, optimized for edge devices and resource-constrained environments.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Published](https://img.shields.io/badge/Published-IJETRM%20Mar%202025-purple.svg)](https://ijetrm.com/issues/files/Mar-2025-24-1742838603-MAR68.pdf)

---

## 📄 Publication

This project was published in the **International Journal of Engineering Technology Research & Management (IJETRM)**, Volume 09, Issue 03, March 2025.

**Authors:** Novera Habeeb, C. Vishal Goud, D. Kishore Reddy, M. Rathna Teja, M. Varshith

📎 [Read the Paper](https://ijetrm.com/issues/files/Mar-2025-24-1742838603-MAR68.pdf)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Applications](#applications)
- [Project Structure](#project-structure)
- [Team](#team)

---

## 🧠 Overview

LeYOLO (**Le**ightweight **YOLO**) is an advanced object detection model built on top of the YOLO framework, designed to overcome limitations of traditional detection models in resource-constrained environments such as IoT devices, mobile platforms, and embedded systems.

LeYOLO achieves a **42% reduction in FLOPs** compared to YOLOv9-Tiny while maintaining competitive — and in most metrics, superior — detection performance.

**Core innovations:**
- **Fast Pyramidal Architecture Network (FPAN)** for multi-scale feature extraction
- **Decoupled Network-in-Network (DNiN)** detection head for lightweight inference
- **Information Bottleneck-inspired backbone** for efficient feature compression

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🏗️ Lightweight Architecture | Depthwise separable convolutions reduce parameters and FLOPs significantly |
| 🔭 Multi-Scale Detection | Feature Pyramid Network (FPN) for detecting small, medium, and large objects |
| 🌙 Low-Light Support | CLAHE + infrared/thermal image fusion for night-time detection |
| ⚡ Real-Time Inference | Optimized for live webcam and video stream processing |
| 📱 Edge Compatibility | Deployable on Jetson Nano, Raspberry Pi, and mobile devices |
| 🎯 Adaptive Loss Function | Dynamically adjusts to object complexity and detection confidence |
| 🔁 Hybrid Activation | Leaky ReLU + Mish for improved gradient flow and convergence |

---

## 🏛️ Architecture

LeYOLO is composed of three main architectural blocks:

```
Input Image
    │
    ▼
┌─────────────────────────────────────┐
│  FPAN Block (Fast Pyramidal Arch.)  │  ← Multi-scale feature extraction
│  1×1, 3×3, 5×5 parallel convs      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  DNiN Block (Decoupled NiN)         │  ← Lightweight depthwise conv
│  DepthwiseConv → BN → ReLU → 1×1   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  FPN Block (Feature Pyramid Net)    │  ← Small & far object detection
│  3×3, 5×5, 7×7 parallel convs      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Residual Connections               │  ← Gradient stability
│  Global Average Pooling → Dense     │
└─────────────────────────────────────┘
    │
    ▼
  Output: Bounding Boxes + Labels + Confidence Scores
```

---

## 📊 Performance

### Improved vs. Baseline LeYOLO

| Metric | Baseline LeYOLO | Improved LeYOLO |
|---|---|---|
| mAP (%) | 45.2 | **54.6** |
| IoU (%) | 65.3 | **76.8** |
| Precision | 0.78 | **0.87** |
| Recall | 0.74 | **0.84** |
| F1-Score | 0.76 | **0.85** |
| FPS | 50 | **64** |

### LeYOLO vs. YOLOv9-Tiny

| Metric | YOLOv9-Tiny | LeYOLO |
|---|---|---|
| Accuracy (%) | 80.5 | **89.3** |
| mAP (%) | 49.3 | **69.6** |
| IoU (%) | 72.1 | **76.8** |
| Precision | 0.81 | **0.87** |
| Recall | 0.79 | **0.84** |
| F1-Score | 0.80 | **0.85** |
| FPS | 55 | **64** |
| FLOPs Reduction | — | **↓ 42%** |

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (NVIDIA RTX 2080 or higher recommended for training)

### Clone the Repository

```bash
git clone https://github.com/<your-username>/LeYOLO.git
cd LeYOLO
```

### Install Dependencies

```bash
pip install torch torchvision tensorflow opencv-python numpy requests pillow
pip install ultralytics scikit-learn matplotlib seaborn pycocotools kagglehub
```

---

## 🚀 Usage

### Run Object Detection

When you run the detection script, you'll be prompted to choose an input mode:

```bash
python detect.py
```

**Detection modes:**

**1. From Image URL**
```
Enter 'url' for image URL, 'webcam' for live detection, or 'image' for a static image file: url
Enter image URL: https://example.com/street.jpg
```

**2. From Local Image File**
```
Enter 'url' for image URL, 'webcam' for live detection, or 'image' for a static image file: image
Enter image file path: /path/to/your/image.jpg
```

**3. Live Webcam Detection**
```
Enter 'url' for image URL, 'webcam' for live detection, or 'image' for a static image file: webcam
```
> Press `q` to quit the live feed.

### Load Pre-trained Model

```python
from ultralytics import YOLO

model = YOLO("weights/LeYOLOMedium.pt")
results = model("your_image.jpg")
```

---

## 📦 Dataset

LeYOLO was trained and evaluated on the **COCO 2017** dataset (7,000 image subset for training, full validation set).

To download the dataset via KaggleHub:

```python
import kagglehub
path = kagglehub.dataset_download("snikhilrao/coco-2017")
```

**Data augmentation techniques applied:**
- Random rotation, flipping, and cropping
- Brightness & contrast adjustments
- MixUp and CutMix strategies
- Mosaic augmentation

---

## 🖼️ Results

The model was tested across diverse real-world scenarios:

| Scenario | Detection Mode |
|---|---|
| City street with pedestrians and vehicles | Static & Live |
| Flock of birds in flight | Static & Live |
| People with tabletop objects (wine, glasses) | Static |
| Food items (cake, coffee, cutlery) | Static |

Training metrics (over 50 epochs) show consistent improvement in both accuracy and loss convergence, with IoU and mAP rising steadily across training epochs.

---

## 🌍 Applications

LeYOLO is designed for broad deployment across multiple domains:

- 🚗 **Autonomous Vehicles** — pedestrian detection, traffic sign recognition, collision avoidance
- 🏥 **Healthcare** — tumor detection in MRI/CT scans, medical equipment tracking
- 🔒 **Security & Surveillance** — intruder detection, crowd monitoring, weapon recognition
- 🏭 **Manufacturing** — defect detection, quality control on assembly lines
- 🌾 **Agriculture** — crop disease monitoring, pest detection, livestock tracking
- 🛍️ **Retail** — automated checkout, inventory management, customer behavior analysis
- 🛰️ **Satellite Imagery** — urban planning, disaster response, environmental monitoring

---

## 📁 Project Structure

```
LeYOLO/
│
├── detect.py                  # Main detection script (URL / static / webcam)
├── train.py                   # Model training pipeline
├── model.py                   # LeYOLO architecture definition
├── evaluate.py                # Metrics: IoU, mAP, Precision, Recall, F1
│
├── weights/
│   └── LeYOLOMedium.pt        # Pre-trained model weights
│
├── dataset_7000/
│   ├── images/                # Training images (COCO subset)
│   └── annotations.json       # COCO-format annotations
│
├── outputs/                   # Detection output images
├── requirements.txt
└── README.md
```

---

## 🧑‍💻 Team

**B.Tech in Artificial Intelligence and Machine Learning**
J.B. Institute of Engineering & Technology, Hyderabad (2021–2025)

| Name | Roll Number |
|---|---|
| C. Vishal | 21671A7311 |
| D. Kishore | 21671A7313 |
| M. Rathna Teja | 21671A7342 |
| M. Varshith | 22675A7306 |

**Guide:** Ms. Novera Habeeb, Assistant Professor  
**Head of Department:** Dr. G. Arun Sampaul Thomas, Associate Professor

---

## 📚 References

- Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016
- Bochkovskiy et al., *YOLOv4: Optimal Speed and Accuracy of Object Detection*, arXiv 2020
- Lin et al., *Focal Loss for Dense Object Detection*, TPAMI 2020
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
- Howard et al., *MobileNets: Efficient CNNs for Mobile Vision Applications*, arXiv 2017

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
