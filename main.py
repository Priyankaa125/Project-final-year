import tkinter as tk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageTk
from datetime import datetime

# Load pre-trained emotion detection model
def load_models():
    try:
        emotion_model = load_model(r'C:\Users\ASUS\OneDrive\Desktop\project\real_time_facial_emotion_detection\model.h5')
        return emotion_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Detect faces using a custom Haar Cascade
def detect_faces(frame):
    # Load Haar Cascade for face detection
    face_classifier = cv2.CascadeClassifier(r'C:\Users\ASUS\OneDrive\Desktop\project\real_time_facial_emotion_detection\haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces

# Initialize previous emotion to track changes
previous_emotion = None

# Detect emotions in the faces using CNN model
def detect_emotions(frame, emotion_model):
    global previous_emotion
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # Detect faces in the frame
    faces = detect_faces(frame)

    # Loop through detected faces and process each one
    for (x, y, w, h) in faces:
        # Draw a rectangle around each face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Region of interest (ROI) for emotion detection
        roi = frame[y:y + h, x:x + w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Check if ROI is not empty
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float32') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict the emotion of the face
            prediction = emotion_model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]

            # Check if the detected emotion has changed
            if label != previous_emotion:
                # Log the emotion change with timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                emotion_text.set(f"Emotion: {label}\nTime: {timestamp}")  # Update the label text
                previous_emotion = label

            # Display the emotion label on the video frame
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame

# Open video stream and update Tkinter window
def update_frame():
    ret, frame = cap.read()

    if ret:
        # Detect and predict emotions
        frame = detect_emotions(frame, emotion_model)

        # Convert frame to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img)

        # Update image on the label
        label_video.img_tk = img_tk  # Keep a reference to avoid garbage collection
        label_video.config(image=img_tk)

    # Call the update_frame function every 10ms
    window.after(10, update_frame)

# Start emotion detection by opening the camera feed
def start_detection():
    global cap
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    # Start updating frames in Tkinter window
    update_frame()

# Stop emotion detection and close the window
def stop_detection():
    global cap
    if cap.isOpened():
        cap.release()  # Stop the video feed
    window.quit()  # Quit the Tkinter main loop
    cv2.destroyAllWindows()  # Close OpenCV windows

# Setup for Tkinter GUI
def start_button_clicked():
    start_detection()

def stop_button_clicked():
    stop_detection()

# Create the main window using Tkinter
window = tk.Tk()
window.title("Emotion Detection System")

# Set window size and background color
window.geometry("900x700")
window.configure(bg="#f2f2f2")

# Add a title label at the top
title_label = tk.Label(window, text="Emotion Detection System", font=("Helvetica", 24, "bold"), bg="#f2f2f2", fg="darkblue")
title_label.pack(pady=20)

# Create a Label widget to display video feed
label_video = tk.Label(window)
label_video.pack()

# Create a StringVar to update emotion text
emotion_text = tk.StringVar()
emotion_text.set("Emotion: \nTime: ")

# Create a Label widget to display the emotion and timestamp
emotion_label = tk.Label(window, textvariable=emotion_text, font=("Helvetica", 16), bg="#f2f2f2", fg="darkblue")
emotion_label.pack(pady=20)

# Create Start button to begin emotion detection
start_button = tk.Button(window, text="Start Emotion Detection", font=("Helvetica", 16), bg="green", fg="white", command=start_button_clicked)
start_button.pack(pady=20)

# Create Stop button to stop emotion detection and close the window
stop_button = tk.Button(window, text="Stop Emotion Detection", font=("Helvetica", 16), bg="red", fg="white", command=stop_button_clicked)
stop_button.pack(pady=20)

# Load the pre-trained emotion detection model
emotion_model = load_models()

# Start the Tkinter main loop
window.mainloop()