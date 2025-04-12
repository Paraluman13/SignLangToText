import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

DATA_FILE = "gesture_data.csv"

def collect_gesture(gesture_name, num_samples=100):
    cap = cv2.VideoCapture(0)
    collected_samples = 0
    data = []

    print(f"Show gesture '{gesture_name}' to the camera...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract 21 landmark points (x, y, z)
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                if len(landmarks) == 63:
                    data.append([gesture_name] + landmarks)
                    collected_samples += 1
                    print(f"Collected {collected_samples}/{num_samples}")

                    if collected_samples >= num_samples:
                        cap.release()
                        cv2.destroyAllWindows()

                        df = pd.DataFrame(data)
                        if os.path.exists(DATA_FILE):
                            df.to_csv(DATA_FILE, mode='a', header=False, index=False)
                        else:
                            df.to_csv(DATA_FILE, mode='w', header=['gesture'] + [f"x{i},y{i},z{i}" for i in range(21)], index=False)

                        print(f"Gesture '{gesture_name}' saved successfully!")
                        return

        cv2.putText(frame, f"Show '{gesture_name}' ({collected_samples}/{num_samples})", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture Gesture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

gesture_name = input("Enter gesture: ")
collect_gesture(gesture_name)
