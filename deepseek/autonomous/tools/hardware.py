# tools/hardware.py

from gpiozero import LED, Button, Servo
import time

class HardwareTool:
    def run(self, *args, **kwargs):
        raise NotImplementedError

class LEDTool(HardwareTool):
    def __init__(self, pin):
        self.led = LED(pin)
    def run(self, state: str):
        if state == "on":   self.led.on()
        elif state == "off": self.led.off()
        return {"state": state}

class ServoTool(HardwareTool):
    def __init__(self, pin):
        self.servo = Servo(pin)
    def run(self, angle: float):
        self.servo.value = angle  # between -1 and +1
        return {"angle": angle}

class SensorTool(HardwareTool):
    def __init__(self, pin):
        self.sensor = Button(pin)  # generic digital sensor
    def run(self):
        return {"is_pressed": self.sensor.is_pressed}
