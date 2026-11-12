import os
import sys
import time
import argparse
import logging
from collections import deque

import cv2
import yaml
import joblib
import numpy as np
import pyautogui

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detector import HandDetector
from mouse_controller import AirMouseController

# Configure structured SDE-grade logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("GestureEngine")

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def execute_action(action_name, steps=1):
    if action_name == "screenshot":
        pyautogui.hotkey("win", "prtsc")
    else:
        for _ in range(steps):
            pyautogui.press(action_name)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Hand Gesture HCI Engine")
    parser.add_argument(
        "--profile",
        type=str,
        default="media",
        help="Active profile from config.yaml (e.g., media, presentation)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sys_cfg = cfg["system"]
    mouse_cfg = cfg["mouse"]

    if args.profile not in cfg["profiles"]:
        logger.error(f"Profile '{args.profile}' not found in {args.config}. Available: {list(cfg['profiles'].keys())}")
        sys.exit(1)

    profile = cfg["profiles"][args.profile]
    bindings = profile["bindings"]

    clf = joblib.load("models/gesture_classifier.pkl")
    detector = HandDetector()
    mouse = AirMouseController(
        smoothing=mouse_cfg["smoothing_factor"],
        margin=mouse_cfg["frame_margin"],
        pinch_threshold=mouse_cfg["pinch_threshold"],
        click_cooldown=mouse_cfg["click_cooldown"]
    )

    cap = cv2.VideoCapture(sys_cfg["camera_index"])
    prediction_buffer = deque(maxlen=sys_cfg["smoothing_window"])

    last_continuous_time = 0.0
    last_discrete_time = 0.0
    exit_start_time = None
    prev_frame_time = time.time()

    logger.info(f"Started Gesture Engine | Active Profile: '{args.profile.upper()}'")
    logger.info(f"Hold '{sys_cfg['exit_gesture']}' for {sys_cfg['exit_hold_seconds']}s or press 'q' to exit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame from webcam.")
                break

            now = time.time()
            fps = 1.0 / max(now - prev_frame_time, 1e-5)
            prev_frame_time = now

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detector.detect_async(rgb_frame)

            raw_lm, features = detector.extract_features()
            status_text = "No Hand Detected"
            feedback_text = f"Profile: {args.profile.upper()} | FPS: {int(fps)}"

            if features is not None:
                probs = clf.predict_proba([features])[0]
                max_idx = np.argmax(probs)
                confidence = probs[max_idx]
                predicted = clf.classes_[max_idx]

                if confidence >= sys_cfg["confidence_threshold"]:
                    prediction_buffer.append(predicted)
                else:
                    prediction_buffer.append("Uncertain")

                stable_gesture = max(set(prediction_buffer), key=prediction_buffer.count)
                status_text = f"{stable_gesture} ({confidence * 100:.1f}%)"

                # 1. Hold-to-Quit Check
                if stable_gesture == sys_cfg["exit_gesture"]:
                    if exit_start_time is None:
                        exit_start_time = now
                    elapsed = now - exit_start_time
                    progress = min(elapsed / sys_cfg["exit_hold_seconds"], 1.0)

                    cv2.rectangle(frame, (20, 95), (240, 115), (50, 50, 50), -1)
                    cv2.rectangle(frame, (20, 95), (20 + int(220 * progress), 115), (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"Closing in {max(0.0, sys_cfg['exit_hold_seconds'] - elapsed):.1f}s",
                        (250, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1
                    )

                    if elapsed >= sys_cfg["exit_hold_seconds"]:
                        logger.info("Exit gesture threshold reached. Shutting down cleanly.")
                        break
                else:
                    exit_start_time = None

                # 2. Air Mouse Mode
                if stable_gesture == mouse_cfg["enabled_gesture"]:
                    feedback_text = mouse.update(raw_lm)

                # 3. Profile Action Bindings
                elif stable_gesture in bindings:
                    rule = bindings[stable_gesture]
                    if rule["type"] == "continuous" and (now - last_continuous_time) > profile["continuous_cooldown"]:
                        execute_action(rule["action"], steps=profile["volume_step"])
                        feedback_text = f"Action: {rule['label']}"
                        last_continuous_time = now

                    elif rule["type"] == "discrete" and (now - last_discrete_time) > profile["discrete_cooldown"]:
                        execute_action(rule["action"], steps=1)
                        feedback_text = f"Action: {rule['label']}"
                        last_discrete_time = now

                # Render landmarks
                h, w, _ = frame.shape
                for lm in raw_lm:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 255), -1)
            else:
                exit_start_time = None

            # HUD Overlay
            cv2.rectangle(frame, (10, 10), (540, 85), (20, 20, 20), -1)
            cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, feedback_text, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

            cv2.imshow("Hand Gesture HCI Engine", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        logger.info("Resources released.")


if __name__ == "__main__":
    main()