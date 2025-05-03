import cv2
import pygetwindow as gw
import numpy as np

class ApplicationObserver:
    def __init__(self):
        self.window_handles = {}
        
    def capture_state(self):
        return {
            "screenshots": self.get_screenshots(),
            "active_window": self.get_active_window(),
            "ui_elements": self.detect_ui_components()
        }
    
    def get_screenshots(self):
        return {title: gw.getWindowsWithTitle(title)[0].screenshot() 
                for title in gw.getAllTitles() if gw.getWindowsWithTitle(title)}
    
    def get_active_window_screenshot(self):
        # Get active window and take its screenshot
        active_window = gw.getActiveWindow()
        if active_window:
            screenshot = active_window.screenshot()
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        return None
    
    def detect_ui_components(self):
        image = self.get_active_window_screenshot()
        if image is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection (adjust thresholds as needed)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours (UI component boundaries)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract bounding boxes as "components"
        components = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            components.append({
                "type": "ui_element",
                "bbox": (x, y, w, h)
            })
            
        return components  # Now properly defined