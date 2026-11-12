import time
import math
import ctypes
import numpy as np
import pyautogui

class AirMouseController:
    def __init__(self, smoothing=0.55, margin=0.18, pinch_threshold=0.045, click_cooldown=0.4):
        self.screen_w, self.screen_h = pyautogui.size()
        self.smoothing = smoothing
        self.margin = margin
        self.pinch_threshold = pinch_threshold
        self.click_cooldown = click_cooldown

        self.prev_x = self.screen_w / 2
        self.prev_y = self.screen_h / 2
        self.last_click_time = 0.0

    def update(self, landmarks):
        index_tip = landmarks[8]
        thumb_tip = landmarks[4]

        # Active Box Mapping
        norm_x = np.clip((index_tip.x - self.margin) / (1.0 - 2 * self.margin), 0.0, 1.0)
        norm_y = np.clip((index_tip.y - self.margin) / (1.0 - 2 * self.margin), 0.0, 1.0)

        target_x = norm_x * self.screen_w
        target_y = norm_y * self.screen_h

        # Snappier EMA smoothing
        curr_x = self.prev_x + self.smoothing * (target_x - self.prev_x)
        curr_y = self.prev_y + self.smoothing * (target_y - self.prev_y)

        # Zero-latency native Windows cursor movement
        ctypes.windll.user32.SetCursorPos(int(curr_x), int(curr_y))
        self.prev_x, self.prev_y = curr_x, curr_y

        # Pinch-to-click
        pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        now = time.time()

        if pinch_dist < self.pinch_threshold and (now - self.last_click_time) > self.click_cooldown:
            pyautogui.click(_pause=False)
            self.last_click_time = now
            return "Air Mouse: LEFT CLICK"

        return "Air Mouse: Tracking"