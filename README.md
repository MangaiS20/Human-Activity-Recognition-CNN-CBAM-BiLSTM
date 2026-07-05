# Human Activity Recognition using CNN, CBAM and BiLSTM

A deep learning framework for **Human Activity Recognition (HAR)** that combines **Convolutional Neural Networks (CNN)**, the **Convolutional Block Attention Module (CBAM)**, and **Bidirectional Long Short-Term Memory (BiLSTM)** networks to classify human activities from wearable sensor data.

---

## Project Overview

Human Activity Recognition (HAR) aims to identify human activities using data collected from wearable sensors such as accelerometers and gyroscopes.

This project enhances a CNN-CBAM architecture by integrating **BiLSTM** to better capture temporal dependencies in multivariate sensor signals.

The model was evaluated on two benchmark datasets:

- **KU-HAR**
- **UCI-HAR**

---

## Motivation

While CNNs are effective at extracting spatial features, they often struggle to model long-term temporal dependencies in sequential sensor data.

To address this, this project combines:

- CNN for spatial feature extraction
- CBAM for channel and spatial attention
- BiLSTM for temporal sequence learning

The resulting hybrid architecture aims to improve overall HAR performance.

---

# Model Architecture

<p align="center">
<img src="images/architecture.png" width="750">
</p>

Pipeline:

```
Input Sensor Data
        ↓
CNN Feature Extraction
        ↓
CBAM Attention Module
        ↓
Residual Learning
        ↓
BiLSTM
        ↓
Dense Layers
        ↓
Softmax Classification
```

---

# Datasets

## KU-HAR

- Smartphone sensor dataset
- 18 activity classes
- Accelerometer and gyroscope signals

## UCI-HAR

- Public benchmark HAR dataset
- 6 activity classes
- Accelerometer and gyroscope signals

---

# Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- hdf5storage

---

# Experimental Results

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Performance

| Dataset | Model | Metrics |
|----------|-------|----------|
| KU-HAR | CNN + CBAM + BiLSTM | Accuracy, Precision, Recall, F1 |
| UCI-HAR | CNN + CBAM + BiLSTM | Accuracy, Precision, Recall, F1 |

---

# Training Accuracy

<p align="center">
<img src="images/accuracy.png" width="700">
</p>

---

# Training Loss

<p align="center">
<img src="images/loss.png" width="700">
</p>

---

# F1 Score

<p align="center">
<img src="images/f1_score.png" width="700">
</p>

---

# Confusion Matrix

<p align="center">
<img src="images/confusion_matrix.png" width="700">
</p>

---

# Repository Structure

```
Human-Activity-Recognition-CNN-CBAM-BiLSTM
│
├── images/
├── notebooks/
├── dataset/
├── results/
├── HAR_CNN_CBAM_BiLSTM.ipynb
├── HAR_CNN_CBAM_BiLSTM_Updated.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

# Installation

```bash
git clone https://github.com/MangaiS20/Human-Activity-Recognition-CNN-CBAM-BiLSTM.git

cd Human-Activity-Recognition-CNN-CBAM-BiLSTM

pip install -r requirements.txt
```

---

# Usage

Open and execute:

```
HAR_CNN_CBAM_BiLSTM.ipynb
```

or run the Python implementation after configuring the dataset paths.

---

# Future Improvements

- Evaluate additional HAR datasets
- Hyperparameter optimization
- Model compression for edge devices
- Real-time deployment
- Explainable AI (XAI) analysis

---

# References

1. Woo S, Park J, Lee JY, Kweon IS. **CBAM: Convolutional Block Attention Module.**
2. Human Activity Recognition Using Attention-Mechanism-Based Deep Learning Feature Combination.

---

# Author

**Mangai S**

**M.Sc. Data Science**

Python | SQL | Machine Learning | Data Analytics

Open to Data Analyst, Machine Learning and AI opportunities.
