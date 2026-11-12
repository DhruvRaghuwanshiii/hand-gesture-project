import cv2
import os
import sys
import pandas as pd
import numpy as np

# Ensure Python locates detector.py inside src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detector import HandDetector

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "gestures.csv")
os.makedirs(DATA_DIR, exist_ok=True)

GESTURE_NAME = input("Enter gesture label to record (e.g., pinky_up, fist, open_palm, peace, thumbs_up): ").strip()
SAMPLES_TO_COLLECT = 200

cap = cv2.VideoCapture(0)
detector = HandDetector()

samples = []
print(f"\n[INFO] Hold your hand up in the '{GESTURE_NAME}' pose.")
print("[INFO] Tap or hold SPACE in the video window to record frames.")
print("[INFO] Press 'q' in the video window to stop and save.")

while cap.isOpened() and len(samples) < SAMPLES_TO_COLLECT:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detector.detect_async(rgb_frame)

    raw_lm, features = detector.extract_features()

    if raw_lm:
        h, w, _ = frame.shape
        for lm in raw_lm:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    cv2.putText(frame, f"Label: {GESTURE_NAME} | Progress: {len(samples)}/{SAMPLES_TO_COLLECT}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and features is not None:
        samples.append(np.append(features, GESTURE_NAME))
        print(f"Captured: {len(samples)}/{SAMPLES_TO_COLLECT}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()

if samples:
    columns = [f"f_{i}" for i in range(63)] + ["label"]
    new_df = pd.DataFrame(samples, columns=columns)
    
    if os.path.exists(CSV_PATH):
        existing_df = pd.read_csv(CSV_PATH)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
        
    combined_df.to_csv(CSV_PATH, index=False)
    print(f"\n[SUCCESS] Saved {len(samples)} samples to {CSV_PATH}")