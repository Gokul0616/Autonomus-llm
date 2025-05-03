import gpiozero

class HardwareManager:
    def __init__(self):
        self.devices = {
            "robot_arm": gpiozero.Robot(left=(17,18), right=(22,23)),
            "sensors": gpiozero.DistanceSensor(echo=24, trigger=23)
        }
        
    def execute_physical_action(self, command):
        if command["device"] in self.devices:
            getattr(self.devices[command["device"]], command["action"])()