# 😴 Real-Time Drowsiness Detection System

A Python-based application that uses computer vision to detect signs of drowsiness in real-time from a webcam feed and triggers an alert.

---

## 🚀 Project Description

This project is a safety-oriented application designed to prevent accidents caused by driver fatigue. It uses **OpenCV** to capture video, **Dlib** for highly accurate facial landmark detection, and **Pygame** for sounding an alarm. The core of the system is the calculation of the **Eye Aspect Ratio (EAR)**, a metric that determines the level of eye openness. By monitoring this value, the program can reliably detect when a person's eyes are closed for too long and alert them immediately.

---

## ✨ Key Features

* **Real-Time Video Processing**: Analyzes a live webcam feed to monitor the user continuously.
* **Facial Landmark Detection**: Utilizes Dlib's pre-trained model to accurately identify 68 key points on the face.
* **Eye Aspect Ratio (EAR) Monitoring**: Implements the EAR algorithm to precisely track eye closure.
* **Head Pose Estimation**: Provides basic data on head orientation as a potential secondary indicator of drowsiness.
* **Audible & Visual Alerts**: Triggers a loud alarm sound and displays a "DROWSINESS ALERT!" message on the screen to warn the user.

---

## 💻 Technology Stack

* **Language**: **Python**
* **Computer Vision**: **OpenCV** – For video capture, processing, and displaying frames.
* **Face & Landmark Detection**: **Dlib** – For detecting faces and predicting facial landmark locations.
* **Audio Playback**: **Pygame** – For playing the alarm sound.
* **Array Manipulation**: **NumPy** – (Used within helper functions for numerical operations).
* **Utilities**: **imutils** – For convenience functions related to facial landmarks.

---

## 🔧 How It Works

1.  The script initializes the webcam and loads the Dlib facial landmark predictor model.
2.  It captures video frame-by-frame.
3.  For each frame, Dlib's face detector locates a face, and the landmark predictor maps the 68 facial points.
4.  The application isolates the landmark coordinates for both eyes.
5.  It calculates the **Eye Aspect Ratio (EAR)**. A smaller EAR value indicates that the eyes are more closed.
6.  If the EAR value falls below a specific threshold (`0.25`) for a set number of consecutive frames (`20`), the system determines that the user is drowsy.
7.  Once drowsiness is detected, an alert is triggered, playing an alarm sound and overlaying a warning text on the video feed until the user's eyes open again.

---

## ⚙️ Setup and Installation

To get this project running locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    * It is recommended to use a virtual environment.
    * Create a `requirements.txt` file with the following content:
        ```
        opencv-python
        dlib
        imutils
        pygame
        numpy
        ```
    * Install the packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Required Assets:**
    * Download the Dlib facial landmark predictor model: `shape_predictor_68_face_landmarks.dat`. You can get it from the [Dlib website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). Unzip it and place the `.dat` file in the project's root directory.
    * Ensure you have an `alarm.wav` sound file in the same directory.
    * Make sure the `utils.py` file containing the helper functions (`eye_aspect_ratio`, `get_landmarks`, etc.) is also in the directory.

4.  **Run the application:**
    ```bash
    python your_script_name.py
    ```
