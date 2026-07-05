# Human Activity Recognition using CNN, CBAM and BiLSTM

## Overview

Human Activity Recognition (HAR) is a classification task that identifies human activities using data collected from wearable sensors such as accelerometers and gyroscopes.

This project presents a deep learning framework that combines **Convolutional Neural Networks (CNN)**, the **Convolutional Block Attention Module (CBAM)**, and **Bidirectional Long Short-Term Memory (BiLSTM)** networks to improve feature extraction and temporal sequence learning.

The proposed model was evaluated on two benchmark datasets:

- KU-HAR
- UCI-HAR

---

## Motivation

Traditional CNN models are effective at extracting spatial features from sensor data but have limited capability in capturing long-term temporal dependencies.

To overcome this limitation, this project extends a CNN-CBAM architecture by integrating **BiLSTM** layers, enabling the model to learn both spatial and temporal representations from multivariate sensor signals.

---

## Model Architecture

The proposed architecture follows the pipeline:

Input Sensor Data

↓

CNN Feature Extraction

↓

CBAM Attention Module

↓

Residual Feature Learning

↓

BiLSTM

↓

Dense Layers

↓

Softmax Classification

---

## Datasets

### KU-HAR

- Smartphone sensor dataset
- 18 activity classes
- Accelerometer and gyroscope signals

### UCI-HAR

- Public Human Activity Recognition dataset
- 6 activity classes
- Accelerometer and gyroscope data

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Results

The proposed model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Training and validation performance are included in this repository.

---

## Repository Structure

```
Human-Activity-Recognition-CNN-CBAM-BiLSTM

│── notebooks/
│── images/
│── dataset/
│── results/
│── README.md
│── requirements.txt
```

---

## Future Improvements

- Evaluate on additional HAR datasets
- Hyperparameter optimization
- Model compression for edge devices
- Real-time HAR deployment

---

## Author

**Mangai S**

M.Sc. Data Science

Open to Data Analytics and Machine Learning opportunities.
