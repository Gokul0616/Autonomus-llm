# autonomous/self_model.py
import json
from pathlib import Path
import datetime

class SelfModel:
    def __init__(self, path="self_model.json"):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            # seed with basic identity
            self.data = {
                "name": "DeepAgent",
                "created_at": str(datetime.now()),
                "capabilities": ["code_generation", "planning", "self_improvement"],
                "goals": [],
                "limitations": []
            }

    def update(self, reflection_text: str):
        # Append reflection to log and optionally extract key-value beliefs
        self.data.setdefault("reflections", []).append({
            "time": str(datetime.now()),
            "text": reflection_text
        })
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))
