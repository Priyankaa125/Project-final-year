import pandas as pd
import numpy as np
from keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('fer2013.csv')  

# Preview the dataset
print(data.head())

# Separate features (pixel values) and labels (emotion categories)
X = []
y = []
for i in range(len(data)):
    pixels = data['pixels'][i]
    X.append(np.array(pixels.split(), dtype='float32').reshape(48, 48, 1))  # Reshape to (48, 48, 1) for grayscale
    y.append(data['emotion'][i])

# Convert to NumPy arrays
X = np.array(X) / 255.0  # Normalize pixel values to the range [0, 1]
y = to_categorical(np.array(y), num_classes=7)  # One-hot encode the labels (7 emotion categories)

print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")
