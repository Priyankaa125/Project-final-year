import tkinter as tk
from tkinter import font
import threading
import main  # Assuming main.py contains the emotion detection logic

# Function to start emotion detection
def start_emotion_detection():
    threading.Thread(target=main.start_detection, daemon=True).start()

# Set up the main window
root = tk.Tk()
root.title("Sentiment Analysis System")

# Set the window size to full screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}+0+0")

# Set background color
root.configure(bg='#f0f0f0')

# Create a frame to hold the widgets
main_frame = tk.Frame(root, bg='#f0f0f0')
main_frame.pack(expand=True, fill='both')

# Custom font for the title
title_font = font.Font(family="Arial", size=60, weight="bold")

# Title label
title_label = tk.Label(main_frame, text="Sentiment Analysis System", font=title_font, bg='#f0f0f0', fg='#333333')
title_label.pack(pady=(100, 40))  # Title with padding at the top

# Instruction label
instruction_label = tk.Label(main_frame, text="Click the button below to start sentiment analysis.", font=("Helvetica", 24), bg='#f0f0f0', fg="#555555")
instruction_label.pack(pady=20)

# Button frame for better positioning
button_frame = tk.Frame(main_frame, bg='#f0f0f0')
button_frame.pack(pady=50)

# Start Button with styling
start_button = tk.Button(button_frame, text="Start Sentiment Analysis", command=start_emotion_detection, font=("Helvetica", 20), bg='#4CAF50', fg='white', padx=30, pady=15, relief="raised", bd=5)
start_button.pack()

# Allow the user to resize the window
root.resizable(True, True)

# Start the Tkinter main loop
root.mainloop()
