import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Define paths to training and validation datasets
train_dir = r'C:\Users\ASUS\OneDrive\Desktop\project\real_time_facial_emotion_detection\train_images'
val_dir = r'C:\Users\ASUS\OneDrive\Desktop\project\real_time_facial_emotion_detection\val_images'

# Image data generator for augmenting and normalizing images
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2, rotation_range=10)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),  # Resize images to 48x48
    color_mode='grayscale', # Use grayscale images
    batch_size=64,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),  # Resize images to 48x48
    color_mode='grayscale', # Use grayscale images
    batch_size=64,
    class_mode='categorical'
)

# Define the CNN model
model = Sequential()

#This layer applies 32 filters (or kernels) of size 3x3 to the input image.
# Each filter will learn to detect specific features (like edges, textures) in the images.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

#2nd convolutional layer with 64 filters
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
#fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

#sum of probability of all 7 emotions should be 1
model.add(Dense(7, activation='softmax'))  # 7 emotion classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the generators
history = model.fit(
    train_generator,
    epochs=35,
    validation_data=val_generator
)

# Save the trained model
model.save('model.h5')

# Plot training and validation loss/accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Save the plot
plt.savefig('training_validation_results.png')
plt.show()
