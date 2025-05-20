import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os

# Load your model and emotion labels
model = load_model('C:/Users/ASUS/OneDrive/Desktop/project/real_time_facial_emotion_detection/model.h5')

emotion_labels = ['happy', 'sad', 'angry', 'surprised', 'disgusted', 'fearful', 'neutral']

def find_image_path(image_name):
    base_directory = "C:/Users/ASUS/OneDrive/Desktop/project/real_time_facial_emotion_detection/train_images"
    
    # Loop through each emotion folder
    for emotion_folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, emotion_folder)
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if os.path.isfile(image_path):
                return image_path
    
    return None

def predict_emotion(image_name_input):
    image_path = find_image_path(image_name_input)
    
    if image_path is None:
        print(f"Image '{image_name_input}' not found in any emotion folder.")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to load image from path: {image_path}")
        return

    # Resize the image to display it smaller (e.g., 400x400 pixels)
    image_resized = cv2.resize(image, (400, 400))
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # Preprocess the image for the model
    gray = cv2.resize(gray, (48, 48))  # Resize to the model's input size
    gray = gray.astype('float32') / 255.0  # Scale pixel values
    gray = img_to_array(gray)
    gray = np.expand_dims(gray, axis=0)  # Expand dimensions to match model input shape
    
    # Predict the emotion
    prediction = model.predict(gray)[0]
    
    # Get the predicted emotion
    emotion_index = np.argmax(prediction)
    emotion_label = emotion_labels[emotion_index]
    
    # Get the confidence of the prediction
    confidence = prediction[emotion_index]
    
    # Display the resized image with the predicted emotion and confidence
    cv2.putText(image_resized, f'Emotion: {emotion_label} ({confidence:.2f})', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Image", image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Input the image name
image_name_input = input("Enter the image name: ")
predict_emotion(image_name_input)
