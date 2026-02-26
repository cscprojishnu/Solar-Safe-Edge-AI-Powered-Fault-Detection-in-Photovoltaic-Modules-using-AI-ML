# 🌞 SOLAR-SAFE: Edge AI-Powered Fault Detection in Photovoltaic Modules

**Authors:**  
Jishnu Teja Dandamudi (IEEE Member), Yogendra Chhetri, Prajeesh C. B, Abhishek S     
**Institution:** 
Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India 

Department of Continuing Education, Indian Institue of Science, Bengaluru, India

> **Solar-Safe: An Edge AI - Powered Fault Detection in PhotoVoltaic Modules Using Deep Learning Models**  
> Published in: *IEEE Xplore* (2026)  
> DOI: [10.1109/ICITEICS64870.2025.11341086](https://doi.org/10.1109/ICITEICS64870.2025.11341086)]

---

## 📘 Project Overview

**SOLAR-SAFE** is a deep learning–based system for **fault detection in photovoltaic (PV) modules** using **infrared (IR) imagery**.  
It employs **Convolutional Neural Networks (CNN)** and **Involutional Neural Networks (INN)** to detect anomalies such as **hotspots, cracks, shading**, and **diode faults**.  
The trained models are deployed on the **Google Coral Edge TPU** for **real-time, low-latency inference** at the edge.

This project supports the transition to clean energy by enabling **automated, safe, and scalable fault detection** in solar farms.

---

## ⚙️ Features

- 🧠 Deep Learning–based CNN and INN architectures  
- 🌡️ Trained on *InfraredSolarModules* dataset (20,000 IR images)  
- ⚡ Real-time on-device inference using Google Coral Dev Board  
- 📊 Evaluated on Accuracy, Precision, Recall, and F1-score  
- 🔍 Supports automated anomaly classification for 12 fault classes  

---

## 🗂 Dataset – *InfraredSolarModules*

| Class | Description | Images |
|-------|--------------|--------|
| Cell | Hot spot on single cell | 1,877 |
| Cell-Multi | Multiple cell hotspots | 1,288 |
| Cracking | Surface cracks on module | 941 |
| Hot-Spot | Thin film hotspot | 251 |
| Hot-Spot-Multi | Multiple thin film hotspots | 247 |
| Shadowing | Obstruction due to shading | 1,056 |
| Diode | Activated bypass diode (⅓ module) | 1,499 |
| Diode-Multi | Multiple bypass diodes (⅔ module) | 175 |
| Vegetation | Vegetation blocking panel | 1,639 |
| Soiling | Dust or debris | 205 |
| Offline-Module | Entire module heated | 828 |
| No-Anomaly | Normal module | 10,000 |

---

## 🧩 Methodology

1. **Preprocessing:**  
   - Resizing, normalization, and label encoding  
   - Dataset split: 70% Train / 15% Validation / 15% Test  

2. **Model Training:**  
   - Optimizer: Adam (`lr=1e-3`)  
   - Loss: CrossEntropyLoss  
   - Epochs: 100–200 | Batch size: 64–128  

3. **Architectures:**  
   - **CNN:** Captures spatial patterns and textures  
   - **INN:** Uses position-dependent kernels for localized defect detection  

4. **Edge Deployment:**  
   - Quantized TensorFlow Lite models  
   - Compiled using Edge TPU Compiler  
   - Deployed for on-device inference  

---

## 📈 Results Summary

| Model | Accuracy (%) | F1-Score | Inference Time (ms/frame) | Device |
|--------|---------------|----------|---------------------------|--------|
| CNN | 95.8 | 0.79 | 66.4 | Coral Edge TPU |
| INN | 93.6 | 0.78 | 25.5 | Coral Edge TPU |

✅ **CNN:** Best performance at 98.9% training accuracy, 84.9% validation, 82.9% test  
⚡ **INN:** 2.5× faster inference than CNN, ideal for real-time inspection  

---

## 🧰 Hardware – Google Coral Dev Board

| Feature | Specification |
|----------|----------------|
| Processor | NXP i.MX 8M (Quad Cortex-A53 + Cortex-M4) |
| AI Accelerator | Edge TPU (4 TOPS @ <2W) |
| Memory | 1 GB LPDDR4 |
| Storage | 8 GB eMMC + microSD |
| OS | Mendel Linux |
| Connectivity | Wi-Fi, Bluetooth, Gigabit Ethernet |
| Use Case | Low-power, real-time AI inference |

---

## 🚀 Installation & Usage

### 🔧 Requirements
- Python 3.9+
- TensorFlow / PyTorch
- OpenCV
- NumPy, Pandas, Matplotlib
- Edge TPU Compiler (for deployment)

### ▶️ Steps
```bash
# Clone repository
git clone https://github.com/cscprojishnu/Solar-Safe-Edge-AI-Powered-Fault-Detection-in-Photovoltaic-Modules-using-AI-ML.git
cd Solar-Safe-Edge-AI-Powered-Fault-Detection-in-Photovoltaic-Modules-using-AI-ML

# Install dependencies
pip install -r requirements.txt

# Train CNN / INN models
python train_cnn.py
python train_inn.py

# Convert to Edge TPU compatible model
edgetpu_compiler model.tflite

# Run real-time inference
python inference_edge.py
