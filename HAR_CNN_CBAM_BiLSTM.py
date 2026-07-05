# ============================================================
# Human Activity Recognition using CNN + CBAM + BiLSTM
# Author: Mangai S
#
# Refactored version for GitHub portfolio.
# Improvements:
# - Professional header
# - Clear section separators
# - Ready for modularization into src/model.py, src/train.py, etc.
# - Original implementation preserved (logic unchanged)
# ============================================================

# -*- coding: utf-8 -*-
# ==================== KU-HAR ====================

import tensorflow as tf
print(tf.__version__)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tensorflow.keras.utils import plot_model
import pickle
from sklearn.utils import shuffle

# Import necessary items from Keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Activation, Dropout, UpSampling2D,
                                      Conv2DTranspose, Conv2D, MaxPooling2D,
                                      Input, BatchNormalization, Add,
                                      AveragePooling2D, Flatten, ReLU, Dense,
                                      Reshape, Multiply, concatenate, add,
                                      GlobalAveragePooling2D, GlobalMaxPool2D,
                                      Bidirectional, LSTM, TimeDistributed, Permute)
from tensorflow.keras import regularizers

# ==================== Data Loading ====================
!pip install hdf5storage
import hdf5storage
mat = hdf5storage.loadmat('/content/drive/MyDrive/datasets-mini/KU-HAR/KU-HAR/KU_har_time_freq_spectrogram_SP.mat')
data = mat['SP']
del mat

img_rows, img_cols = 8, 129
Channels = 12

# Load labels
df = pd.read_csv("/content/drive/MyDrive/datasets-mini/kuhar_labels.csv", header=None)
dff = df.values
y = dff[:, 0]
num_classes = len(set(y))
print(f"Number of classes: {num_classes}")

# Prepare data - use first 6 channels
x = np.array(data[:, :, :, 0:6], dtype=np.float32)
print(f"Data shape: {x.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=9, stratify=y
)
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Max label: {max(y_train)}")

# Convert to categorical
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

del x

# ==================== Metrics ====================
def dsc(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dsc(y_true, y_pred)

def IOU(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    thresh = 0.5
    y_true = tf.cast(tf.greater_equal(y_true, thresh), tf.float32)
    y_pred = tf.cast(tf.greater_equal(y_pred, thresh), tf.float32)
    union = tf.reduce_sum(tf.maximum(y_true, y_pred)) + tf.keras.backend.epsilon()
    intersection = tf.reduce_sum(tf.minimum(y_true, y_pred)) + tf.keras.backend.epsilon()
    iou = intersection/union
    return iou

def recall_m(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))

# ==================== CBAM Module ====================
def CBAM(x, r):
    """Convolutional Block Attention Module"""
    # Channel Attention
    f = x.shape[-1]
    a_pool = GlobalAveragePooling2D()(x)
    m_pool = GlobalMaxPool2D()(x)
    Dense_layer1 = Dense(f // r, activation='relu')
    Dense_layer2 = Dense(f, activation='relu')
    avg_out = Dense_layer2(Dense_layer1(a_pool))
    max_out = Dense_layer2(Dense_layer1(m_pool))
    channel = add([avg_out, max_out])
    channel = Activation('sigmoid')(channel)
    channel = Reshape((1, 1, f))(channel)
    C_out = Multiply()([x, channel])

    # Spatial Attention
    av_pooling = AveragePooling2D(pool_size=(1, 1))(C_out)
    ma_pooling = MaxPooling2D(pool_size=(1, 1))(C_out)
    spatial = concatenate([av_pooling, ma_pooling], axis=3)
    spatial = Conv2D(1, (3, 3), strides=1, padding='same')(spatial)
    spatial = Activation('sigmoid')(spatial)
    S_out = Multiply()([C_out, spatial])

    return S_out

def conv_block(input, num_filters):
    """Convolution block"""
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = Activation("relu")(x)
    return x

# ==================== BiLSTM+CNN+CBAM Model ====================
def build_bilstm_cnn_cbam_model(input_shape, num_classes):
    """
    Build BiLSTM + CNN + CBAM model for HAR

    Architecture:
    1. CNN layers with CBAM for spatial feature extraction
    2. Reshape for sequence modeling
    3. BiLSTM layers for temporal dependencies
    4. Dense layers for classification
    """
    inputA = Input(shape=input_shape)

    # ===== CNN Feature Extraction with CBAM =====
    # Block 1
    s1 = conv_block(inputA, 16)
    a1 = CBAM(s1, 8)
    pool1 = MaxPooling2D(pool_size=(2, 2))(s1)

    # Block 2
    s2 = conv_block(pool1, 32)
    s2 = Dropout(0.1)(s2)
    a2 = CBAM(s2, 8)

    # Block 3
    s3 = conv_block(a2, 32)
    s3 = Dropout(0.1)(s3)
    a3 = CBAM(s3, 8)

    # Residual connection
    a33 = Add()([a2, a3])
    pool3 = MaxPooling2D(pool_size=(2, 2))(a33)

    # Block 4
    s4 = conv_block(pool3, 64)
    s4 = Dropout(0.15)(s4)
    a4 = CBAM(s4, 8)

    # Block 5
    s5 = conv_block(a4, 64)
    s5 = Dropout(0.15)(s5)
    a5 = CBAM(s5, 8)

    # Residual connection
    a55 = Add()([a4, a5])
    pool5 = MaxPooling2D(pool_size=(2, 2))(a55)

    # Block 6
    s6 = conv_block(pool5, 128)
    s7 = Dropout(0.15)(s6)

    # ===== Prepare for BiLSTM =====
    # Reshape: (batch, height, width, channels) -> (batch, height, width*channels)
    # This treats each height position as a time step
    shape = s7.shape
    reshaped = Reshape((shape[1], shape[2] * shape[3]))(s7)

    # ===== BiLSTM Layers =====
    # First BiLSTM layer
    bilstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(reshaped)

    # Second BiLSTM layer
    bilstm2 = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(bilstm1)

    # ===== Dense Layers =====
    x = Dense(128, activation="relu")(bilstm2)
    x = Dropout(0.15)(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.15)(x)

    x = Dense(48, activation="relu")(x)
    x = Dropout(0.1)(x)

    x = Dense(32, activation="relu")(x)

    # Output layer
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputA, outputs=output)

    return model

# ==================== Build and Compile Model ====================
print("\nBuilding BiLSTM+CNN+CBAM model...")
model = build_bilstm_cnn_cbam_model(X_train[0].shape, num_classes)

opt = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',  # Changed from 'mse' for classification
    metrics=['accuracy', precision_m, recall_m, f1_m]
)

model.summary()

# ==================== Save Model Architecture ====================
model.save('/content/drive/MyDrive/datasets-mini/BiLSTM_CNN_CBAM_Signal.h5')
print("\nModel architecture saved!")

# ==================== Training ====================
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ==================== Save Training History ====================
train_loss = history.history['loss']
valid_loss = history.history['val_loss']
train_acc = history.history['accuracy']
valid_acc = history.history['val_accuracy']
train_f1 = history.history['f1_m']
valid_f1 = history.history['val_f1_m']

from numpy import savetxt
savetxt('train_loss_BiLSTM_CBAM.csv', train_loss, delimiter=',')
savetxt('valid_loss_BiLSTM_CBAM.csv', valid_loss, delimiter=',')
savetxt('train_acc_BiLSTM_CBAM.csv', train_acc, delimiter=',')
savetxt('valid_acc_BiLSTM_CBAM.csv', valid_acc, delimiter=',')
savetxt('train_f1_BiLSTM_CBAM.csv', train_f1, delimiter=',')
savetxt('valid_f1_BiLSTM_CBAM.csv', valid_f1, delimiter=',')

# ==================== Plot Training Results ====================
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Accuracy
axes[0].plot(train_acc, label='training')
axes[0].plot(valid_acc, label='validation')
axes[0].set_title('Accuracy Curve')
axes[0].set_xlabel('epochs')
axes[0].set_ylabel('accuracy')
axes[0].legend()

# Loss
axes[1].plot(train_loss, label='training')
axes[1].plot(valid_loss, label='validation')
axes[1].set_title('Loss Curve')
axes[1].set_xlabel('epochs')
axes[1].set_ylabel('loss')
axes[1].legend()

# F1 Score
axes[2].plot(train_f1, label='training')
axes[2].plot(valid_f1, label='validation')
axes[2].set_title('F1 Score Curve')
axes[2].set_xlabel('epochs')
axes[2].set_ylabel('f1_score')
axes[2].legend()

plt.tight_layout()
plt.savefig('BiLSTM_CNN_CBAM_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== Evaluation ====================
print("\nEvaluating on test set...")
y_predict = model.predict(X_test)
y_pred = np.argmax(y_predict, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Confusion Matrix
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test_labels, y_pred)
print("\nConfusion Matrix:")
print(confusion_matrix)
savetxt('confusion_matrix_BiLSTM_CBAM.csv', confusion_matrix, delimiter=',')

# Classification Report
print("\nClassification Report:")
print(metrics.classification_report(y_test_labels, y_pred))

# Overall Accuracy
accuracy = metrics.accuracy_score(y_test_labels, y_pred)
print(f"\nOverall Test Accuracy: {accuracy:.4f}")

# Per-class Accuracy
print("\nPer-class Accuracy:")
for i in range(num_classes):
    class_mask = y_test_labels == i
    if class_mask.sum() > 0:
        class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
        print(f"Class {i}: {class_acc:.4f}")

print("\n" + "="*50)
print("Training and Evaluation Complete!")
print("="*50)


# ==================== UCI-HAR ====================


# =====================================================
# UCI-HAR Spectrogram Dataset (SP.mat)
# BiLSTM + CNN + CBAM (150 epochs, LR scheduler)
# =====================================================

import tensorflow as tf
print("TensorFlow:", tf.__version__)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn import metrics
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Conv2D, MaxPooling2D,
                                     Add, Activation, Multiply, Reshape,
                                     concatenate, GlobalAveragePooling2D, GlobalMaxPool2D,
                                     Bidirectional, LSTM, Lambda)
!pip install hdf5storage
import hdf5storage

# ==================== Load Spectrogram Data ====================
mat_path = '/content/drive/MyDrive/datasets-mini/UCI_har_time_spectrogram_SP.mat'
mat = hdf5storage.loadmat(mat_path)
data = mat['SP']
print(" Loaded UCI-HAR spectrogram data! Shape:", data.shape)

# ==================== Load Labels ====================
label_path = "/content/drive/MyDrive/datasets-mini/ucihar_labels.csv"
df = pd.read_csv(label_path, header=None)
y = df.values[:, 0]
num_classes = len(set(y))
print(f"Number of classes: {num_classes}")

# ==================== Prepare Data ====================
x = np.array(data, dtype=np.float32)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=9, stratify=y
)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ==================== Metrics ====================
def recall_m(y_true, y_pred):
    tp = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    pos = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    return tp / (pos + tf.keras.backend.epsilon())

def precision_m(y_true, y_pred):
    tp = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    pred_pos = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    return tp / (pred_pos + tf.keras.backend.epsilon())

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

# ==================== Fixed CBAM Module ====================
def CBAM(x, r):
    f = int(x.shape[-1])
    # Channel Attention
    avg = GlobalAveragePooling2D()(x)
    maxp = GlobalMaxPool2D()(x)
    fc1 = Dense(f // r, activation='relu')
    fc2 = Dense(f, activation='relu')
    avg_out = fc2(fc1(avg))
    max_out = fc2(fc1(maxp))
    scale = Add()([avg_out, max_out])
    scale = Activation('sigmoid')(scale)
    scale = Reshape((1, 1, f))(scale)
    x = Multiply()([x, scale])
    # Spatial Attention
    avg_pool = Lambda(lambda z: tf.reduce_mean(z, axis=3, keepdims=True))(x)
    max_pool = Lambda(lambda z: tf.reduce_max(z, axis=3, keepdims=True))(x)
    concat = concatenate([avg_pool, max_pool], axis=3)
    attn = Conv2D(f, (3,3), padding='same', activation='sigmoid')(concat)
    return Multiply()([x, attn])

# ==================== Conv Block ====================
def conv_block(x, f):
    x = Conv2D(f, 3, padding='same', activation='relu')(x)
    return x

# ==================== BiLSTM + CNN + CBAM Model ====================
def build_bilstm_cnn_cbam(input_shape, num_classes):
    inp = Input(shape=input_shape)
    c1 = conv_block(inp, 16)
    c1 = CBAM(c1, 8)
    p1 = MaxPooling2D((2,2))(c1)
    c2 = conv_block(p1, 32)
    c2 = CBAM(c2, 8)
    p2 = MaxPooling2D((2,2))(c2)
    c3 = conv_block(p2, 64)
    c3 = Dropout(0.2)(c3)  # slightly reduced dropout
    # Reshape for BiLSTM
    s = c3.shape
    reshaped = Reshape((s[1], s[2]*s[3]))(c3)
    lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(reshaped)
    lstm2 = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(lstm1)
    x = Dense(128, activation='relu')(lstm2)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inp, out)

# ==================== Build & Compile ====================
model = build_bilstm_cnn_cbam(X_train[0].shape, num_classes)
model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy', precision_m, recall_m, f1_m])
model.summary()

# ==================== Learning Rate Scheduler ====================
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.5, patience=10, min_lr=1e-6, verbose=1
)

# ==================== Train ====================
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[lr_callback],
    verbose=1
)

# ==================== Plot Results ====================
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy'); plt.legend()
plt.subplot(1,3,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss'); plt.legend()
plt.subplot(1,3,3)
plt.plot(history.history['f1_m'], label='Train')
plt.plot(history.history['val_f1_m'], label='Val')
plt.title('F1 Score'); plt.legend()
plt.tight_layout()
plt.show()

# ==================== Evaluate ====================
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nClassification Report:\n", metrics.classification_report(y_true, y_pred))
print("\nConfusion Matrix:\n", metrics.confusion_matrix(y_true, y_pred))
print(f"\n Test Accuracy: {metrics.accuracy_score(y_true, y_pred):.4f}")