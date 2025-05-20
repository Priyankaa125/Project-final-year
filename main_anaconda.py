import tkinter as tk
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import threading
from PIL import Image, ImageTk

def load_models():
    try:
        emotion_model = load_model(r'C:\Users\ASUS\OneDrive\Desktop\project\real_time_facial_emotion_detection\model.h5')
        return emotion_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Computing the integral image
def compute_integral_image(image):
    rows = len(image)
    cols = len(image[0])
    integral_image = [[0] * (cols + 1) for _ in range(rows + 1)]

    for y in range(1, rows + 1):
        for x in range(1, cols + 1):
            integral_image[y][x] = (image[y - 1][x - 1] +
                                     integral_image[y - 1][x] +
                                     integral_image[y][x - 1] -
                                     integral_image[y - 1][x - 1])
    return integral_image

# Custom Haar feature calculation function
def haar_feature_value(integral_image, top_left, width, height, feature_type):
    x, y = top_left
    if feature_type == 'edge':
        white = integral_image[y + height][x + width] - integral_image[y][x + width] - integral_image[y + height][x] + integral_image[y][x]
        black = integral_image[y + height][x] - integral_image[y][x] - integral_image[y + height][x + width] + integral_image[y][x + width]
        return white - black
    return 0

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    integral_image = compute_integral_image(gray)
    
    face_cascade_params = {
        'width': 24, 
        'height': 24, 
        'min_size': (30, 30) 
    }

    faces = []
    for y in range(0, len(integral_image) - face_cascade_params['height'], 3):
        for x in range(0, len(integral_image[0]) - face_cascade_params['width'], 3):
            feature_value = haar_feature_value(integral_image, (x, y), face_cascade_params['width'], face_cascade_params['height'], 'edge')
            
            if feature_value > 100: 
                faces.append((x, y, face_cascade_params['width'], face_cascade_params['height']))
    
    return faces

def detect_emotions(frame, emotion_model):
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    faces = detect_faces(frame)
    print(f"Detected faces: {len(faces)}")

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi = frame[y:y + h, x:x + w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float32') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame

def start_detection():
    emotion_model = load_models()
    if not emotion_model:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    print("Camera opened successfully!")

    def capture_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            frame = detect_emotions(frame, emotion_model)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(img)

            label_video.img_tk = img_tk  
            label_video.config(image=img_tk)

            window.update_idletasks()
            window.update()

        cap.release()
        cv2.destroyAllWindows()
        print("Emotion detection stopped.")

    window = tk.Toplevel()
    window.title("Emotion Detection Feed")

    label_video = tk.Label(window)
    label_video.pack(padx=10, pady=10)

    threading.Thread(target=capture_frames, daemon=True).start()

def start_button_clicked(root):
    root.destroy()  
    start_detection()

def stop_detection():
    global cap
    if cap.isOpened():

        cap.release()
    cv2.destroyAllWindows()
    window.quit()


root = tk.Tk()
root.title("Sentiment Analysis System")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}+0+0")

root.configure(bg='#f0f0f0')

main_frame = tk.Frame(root, bg='#f0f0f0')
main_frame.pack(expand=True, fill='both')

title_font = ("Arial", 60, "bold")


title_label = tk.Label(main_frame, text="Sentiment Analysis System", font=title_font, bg='#f0f0f0', fg='#333333')
title_label.pack(pady=(30, 20))


instruction_label = tk.Label(main_frame, text="Click the button below to start emotion detection.", font=("Helvetica", 24), bg='#f0f0f0')
instruction_label.pack(pady=20)


button_frame = tk.Frame(main_frame, bg='#f0f0f0')
button_frame.pack(expand=True)


start_button = tk.Button(button_frame, text="Start Emotion Detection", command=lambda: start_button_clicked(root), font=("Helvetica", 20), bg='#4CAF50', fg='white', padx=20, pady=10)
start_button.pack(pady=(20, 10))


stop_button = tk.Button(button_frame, text="Stop Emotion Detection", command=stop_detection, font=("Helvetica", 20), bg='#FF6347', fg='white', padx=20, pady=10)
stop_button.pack(pady=(10, 20))


root.mainloop()
