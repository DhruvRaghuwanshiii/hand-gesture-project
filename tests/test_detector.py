import numpy as np
from dataclasses import dataclass
from src.detector import HandDetector


@dataclass
class MockLandmark:
    x: float
    y: float
    z: float


class MockResult:
    def __init__(self, landmarks):
        self.hand_landmarks = [landmarks]


def test_translation_and_scale_invariance():
    """Verify that a small hand in the corner produces the exact same feature vector as a large hand in the center."""
    detector = HandDetector.__new__(HandDetector)

    hand_small = [MockLandmark(0.1 + i * 0.01, 0.1 + i * 0.01, 0.0) for i in range(21)]
    hand_large = [MockLandmark(0.5 + i * 0.05, 0.5 + i * 0.05, 0.0) for i in range(21)]

    detector.latest_result = MockResult(hand_small)
    _, vec_small = detector.extract_features()

    detector.latest_result = MockResult(hand_large)
    _, vec_large = detector.extract_features()

    assert vec_small.shape == (63,)
    assert np.allclose(vec_small, vec_large, atol=1e-6), "Features must be invariant to position and scale"


def test_empty_detection_returns_none():
    """Verify graceful handling when no hand is present in the frame."""
    detector = HandDetector.__new__(HandDetector)
    detector.latest_result = None

    raw_lm, features = detector.extract_features()
    assert raw_lm is None
    assert features is None