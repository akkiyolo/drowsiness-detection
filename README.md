﻿# drowsiness-detection

 # 🧠 Head Pose and Drowsiness Detection using OpenCV & Dlib

This project detects head pose orientation and eye-blink-based drowsiness using OpenCV, Dlib, and facial landmarks. It uses your webcam to monitor and alert when signs of drowsiness are detected.

---

## 🔧 Features

- ✅ Real-time drowsiness detection using Eye Aspect Ratio (EAR)
- ✅ Head pose estimation using 3D facial landmarks
- ✅ Alarm sound for drowsiness alert
- ✅ Simple and lightweight implementation using OpenCV & Dlib

---

## 📁 Project Structure

head-pose-drowsiness/
│
├── main.py # Main script to run detection
├── utils.py # Helper functions for EAR and pose
├── shape_predictor_68_face_landmarks.dat # Dlib’s landmark model (download separately)
├── alarm.wav # Alarm sound triggered on drowsiness
└── requirements.txt # Python dependencies
