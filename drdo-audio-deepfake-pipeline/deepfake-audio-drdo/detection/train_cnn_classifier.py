# Train a CNN classifier using MFCC features
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

X, y = [], []
for label, folder in enumerate(["real", "fake"]):
    for file in os.listdir(f"../data/{folder}"):
        features = extract_mfcc(f"../data/{folder}/{file}")
        X.append(features)
        y.append(label)

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(13,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save("../models/audio_cnn.h5")
