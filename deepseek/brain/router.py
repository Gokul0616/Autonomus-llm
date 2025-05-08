import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import yaml

class TaskRouter:
    def __init__(self, config_path="configs/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['router']['device'])
        self.model, self.tokenizer = self._init_router()
        
    def _load_config(self, path):
        with open(Path(__file__).parent.parent / path) as f:
            return yaml.safe_load(f)
            
    def _init_router(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config['router']['model_name']
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['router']['model_name']
        )
        return model, tokenizer
        
    def classify_task(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return torch.argmax(outputs.logits).item()
    
    def online_learn(self, text, label):
        # Implementation for online learning
        pass