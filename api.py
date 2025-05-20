from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Update these paths to the correct locations on your system
model_path = 'C:/Users/ASUS/OneDrive/Desktop/project/real_time_facial_emotion_detection/model.h5'
face_classifier_path = 'C:/Users/ASUS/OneDrive/Desktop/project/real_time_facial_emotion_detection/haarcascade_frontalface_default.xml'

# Load and compile the model (if necessary for training or evaluation)
classifier = load_model(model_path)
classifier.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Load the Haar Cascade
face_classifier = cv2.CascadeClassifier(face_classifier_path)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read and process the image
    img = file.read()
    np_img = np.frombuffer(img, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    faces = face_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({'error': 'No face detected'}), 400

    for (x, y, w, h) in faces:
        roi_gray = image[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float32') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        return jsonify({'emotion': label})

    return jsonify({'error': 'No face detected'}), 400

if __name__ == '__main__':
    app.run(debug=True)
