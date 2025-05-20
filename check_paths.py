import os

# Paths
haar_cascade_path = r'C:\Users\ASUS\OneDrive\Desktop\project\real_time_facial_emotion_detection\haarcascade_frontalface_default.xml'
model_path = r'C:\Users\ASUS\OneDrive\Desktop\project\real_time_facial_emotion_detection\model.h5'

# Check if the files exist
print("Checking Haar Cascade Path:")
print(f"Exists: {os.path.isfile(haar_cascade_path)}")

print("Checking Model Path:")
print(f"Exists: {os.path.isfile(model_path)}")
