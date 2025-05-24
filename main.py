import cv2
import dlib
import time
from utils import eye_aspect_ratio, get_landmarks, shape_to_np, get_head_pose
from imutils import face_utils
import pygame

EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20

COUNTER = 0

# Load assets
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Sound
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

# Start video stream
cap = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    shape = get_landmarks(gray, detector, predictor)
    if shape:
        shape_np = shape_to_np(shape)
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape_np[lStart:lEnd]
        rightEye = shape_np[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
        else:
            COUNTER = 0
            pygame.mixer.music.stop()

        # Draw eye landmarks
        for (x, y) in shape_np:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Head pose
        rvec, tvec = get_head_pose(shape_np)
        cv2.putText(frame, f"Head Pose Vec: {rvec.flatten()[0]:.2f}", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
