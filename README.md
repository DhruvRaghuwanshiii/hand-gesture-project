# Real-Time Hand Gesture HCI & Air-Mouse Engine

A modular, low-latency Human-Computer Interaction (HCI) system in Python that translates real-time webcam hand landmarks into OS-level controls, media commands, presentation navigation, and smoothed virtual cursor tracking.

## Key Engineering Highlights
- **Translation & Scale Invariant Feature Engineering:** Extracts 21 3D hand landmarks via MediaPipe Tasks API, centering coordinates on the wrist and normalizing by palm scale to generate a 63-dimensional feature vector invariant to hand distance and screen position.
- **Zero-Latency Air-Mouse Kinematics:** Maps index fingertip coordinates within an active virtual bounding box using Exponential Moving Average (EMA) filtering and native Windows `ctypes` cursor calls, paired with Euclidean thumb-index pinch-to-click detection.
- **Configuration-Driven Profile Architecture:** Decouples gesture bindings, cooldowns, and sensitivity parameters into `config.yaml`, enabling instant CLI switching between `media` and `presentation` profiles without modifying source code.
- **Automated Verification:** Includes `pytest` unit tests validating coordinate normalization invariants and null-frame safety.

## Project Structure
```text
├── config.yaml                 # Profile definitions, keybindings, and smoothing parameters
├── data/gestures.csv           # Normalized 63-feature labeled dataset
├── models/                     # MediaPipe landmarker task & trained Random Forest model
├── src/
│   ├── detector.py             # Async MediaPipe hand tracking & feature normalization
│   ├── mouse_controller.py     # EMA-smoothed Air-Mouse & pinch-to-click engine
│   ├── collect_data.py         # Interactive gesture sample collector
│   ├── train.py                # Model training & classification report generator
│   └── app.py                  # CLI application runner, HUD overlay & action dispatcher
└── tests/
    └── test_detector.py        # Pytest unit tests for feature math