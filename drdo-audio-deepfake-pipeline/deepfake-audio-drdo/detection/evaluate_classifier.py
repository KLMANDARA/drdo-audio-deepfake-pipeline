# Evaluate the trained model on test data
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("../models/audio_cnn.h5")
# Dummy test example for placeholder
example = np.random.rand(1, 13)
prediction = model.predict(example)

print("Prediction score (0: real, 1: fake):", prediction)
