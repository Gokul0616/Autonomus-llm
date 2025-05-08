import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
import yaml

class BaseModel(nn.Module, ABC):
    def __init__(self, tokenizer, config):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self._validate_config()
        
    @abstractmethod
    def forward(self, input_ids):
        pass
    
    @abstractmethod
    def generate(self, prompt, **kwargs):
        pass
    
    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'tokenizer_config': self.tokenizer.serialize()
        }, path)
        print(f"Model saved to {path}")
        
    @classmethod
    def load(cls, path, tokenizer_class):
        data = torch.load(path)
        tokenizer = tokenizer_class.deserialize(data['tokenizer_config'])
        model = cls(tokenizer, data['config'])
        model.load_state_dict(data['state_dict'])
        return model
        
    def _validate_config(self):
        required_keys = ['d_model', 'n_layers', 'n_heads']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

class ProductionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._init_quantization()
        
    def _init_quantization(self):
        if self.model.config.get('quantize', False):
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )