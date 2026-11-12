import os
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:
    def __init__(self, model_path="models/hand_landmarker.task", num_hands=1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=num_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            result_callback=self._result_callback
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.latest_result = None

    def _result_callback(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.latest_result = result

    def detect_async(self, frame_bgr):
        """Sends frame to MediaPipe Landmarker asynchronously."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_bgr)
        timestamp_ms = int(time.time() * 1000)
        self.landmarker.detect_async(mp_image, timestamp_ms)

    def extract_features(self):
        """
        Returns:
            raw_landmarks: 21 landmark objects (or None)
            feature_vector: 63-element normalized numpy array (or None)
        """
        if not self.latest_result or not self.latest_result.hand_landmarks:
            return None, None

        landmarks = self.latest_result.hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        # 1. Translation Invariance: Center coordinates relative to wrist (point 0)
        wrist = coords[0]
        centered = coords - wrist

        # 2. Scale Invariance: Divide by distance between wrist (0) and middle knuckle (9)
        palm_scale = np.linalg.norm(centered[9])
        if palm_scale > 0:
            normalized = centered / palm_scale
        else:
            normalized = centered

        return landmarks, normalized.flatten()

    def close(self):
        self.landmarker.close()