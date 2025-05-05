# autonomous/perception/app_observer.py
import platform
import subprocess
from PIL import ImageGrab

class ApplicationObserver:
    def __init__(self):
        self.os = platform.system()

    def capture_state(self):
        """Capture screenshots and active-window info across platforms."""
        screenshot = ImageGrab.grab()

        if self.os == "Windows":
            import pygetwindow as gw
            try:
                win = gw.getActiveWindow()
                active = {"title": win.title, "bbox": win.box}
            except Exception:
                active = {"title": None, "bbox": None}
        elif self.os == "Darwin":
            # macOS: use pygetwindow (supports macOS)
            import pygetwindow as gw
            win = gw.getActiveWindow()
            active = {"title": win.title, "bbox": win.box}
        else:
            # Linux fallback via xdotool
            try:
                win_id = subprocess.check_output(
                    ["xdotool", "getactivewindow"]
                ).decode().strip()
                title = subprocess.check_output(
                    ["xdotool", "getwindowname", win_id]
                ).decode().strip()
                active = {"id": win_id, "title": title}
            except Exception:
                active = {"id": None, "title": None}

        return {
            "screenshot": screenshot,
            "active_window": active
        }

    def detect_ui_components(self):
        """Basic edge‑based UI component detection on the screenshot."""
        import numpy as np
        import cv2

        img = np.array(self.capture_state()["screenshot"])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        components = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            components.append({"type": "ui_element", "bbox": (x, y, w, h)})

        return components
