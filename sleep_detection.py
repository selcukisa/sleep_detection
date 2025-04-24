import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
import pyttsx3

SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
EAR_THRESHOLD = 0.22  # Göz kapanma eşiği 
MOUTH_DIST_THRESHOLD = 25 # Esneme eşiği 
ALERT_COOLDOWN_SECONDS = 5 # Sesli uyarılar arası bekleme

EYE_CLOSED_DURATION_SECONDS = 0.5
TARGET_FPS = 30 # Kodun çalışması için varsayılan FPS 
EYE_CLOSED_TARGET_FRAMES = int(EYE_CLOSED_DURATION_SECONDS * TARGET_FPS) 

LEFT_EYE_SLICE = slice(36, 42)
RIGHT_EYE_SLICE = slice(42, 48)
MOUTH_SLICE = slice(60, 68)
MOUTH_TOP_LIP_RELATIVE_IDX = 2
MOUTH_BOTTOM_LIP_RELATIVE_IDX = 6

def calculate_ear(eye_landmarks):
    """Bir göz için EAR değerini hesaplar."""
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C) if C > 0 else 0.0 
    return ear

def calculate_vertical_mouth_dist(mouth_landmarks):
    """İç ağız dikey açıklığını hesaplar."""
    distance = dist.euclidean(mouth_landmarks[MOUTH_TOP_LIP_RELATIVE_IDX],
                               mouth_landmarks[MOUTH_BOTTOM_LIP_RELATIVE_IDX])
    return distance

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

engine = pyttsx3.init()
last_alert_time = 0
eye_closed_frame_counter = 0 


while True:
    ret, frame = cap.read()
    landmark_frame = frame.copy()
    frame = cv2.flip(frame, 1)
    landmark_frame = cv2.flip(landmark_frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    alert_text = "" 

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        shape = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        cv2.rectangle(landmark_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        for (x, y) in landmarks:
            cv2.circle(landmark_frame, (x, y), 1, (0, 0, 255), -1)

        left_eye = landmarks[LEFT_EYE_SLICE]
        right_eye = landmarks[RIGHT_EYE_SLICE]
        mouth = landmarks[MOUTH_SLICE]

        average_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
        mouth_dist = calculate_vertical_mouth_dist(mouth)

        
        if average_ear < EAR_THRESHOLD:
            
            eye_closed_frame_counter += 1

            
            if eye_closed_frame_counter >= EYE_CLOSED_TARGET_FRAMES:
                alert_text = f"GOZ {EYE_CLOSED_DURATION_SECONDS} SN KAPALI!"

        else:
            eye_closed_frame_counter = 0


        if not alert_text and mouth_dist > MOUTH_DIST_THRESHOLD:
             alert_text = "ESNEME dikkatli olun!"

        cv2.putText(frame, f"EAR: {average_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Mouth: {mouth_dist:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Kapali Kare: {eye_closed_frame_counter}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


    current_time = time.time()
    if alert_text:
        cv2.putText(frame, alert_text, (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if current_time - last_alert_time > ALERT_COOLDOWN_SECONDS:
            print(f"[SESLI UYARI] {alert_text}")
            engine.say(alert_text)
            engine.runAndWait()
            last_alert_time = current_time

    # Kareyi Göster
    cv2.imshow("Yorgunluk Tespiti", frame)
    cv2.imshow("Landmarklar", landmark_frame)
    # Çıkış
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()