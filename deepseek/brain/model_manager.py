import torch
import gc
from pathlib import Path
from functools import lru_cache
from ..architectures import MoE, RetNet, RWKV

class ModelManager:
    def __init__(self, config_path="configs/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.loaded_models = {}
        self.memory_threshold = 0.8  # 80% GPU usage
        
    @lru_cache(maxsize=3)
    def load_model(self, model_type):
        if model_type not in ['moe', 'retnet', 'rwkv']:
            raise ValueError(f"Invalid model type: {model_type}")
            
        if self._check_memory() < self.memory_threshold:
            self._free_memory()
            
        model_class = globals()[model_type.upper()]
        model = model_class.load(
            self.config['model_paths'][model_type],
            self._get_tokenizer()
        )
        self.loaded_models[model_type] = model
        return model.to(self._get_device())
    
    def unload_model(self, model_type):
        if model_type in self.loaded_models:
            del self.loaded_models[model_type]
            gc.collect()
            torch.cuda.empty_cache()
            
    def _check_memory(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        return 0
    
    def _free_memory(self):
        # Advanced memory management
        pass