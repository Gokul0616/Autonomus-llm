# model_versioning.py
import json
import hashlib
from datetime import datetime
from pathlib import Path
import torch
class ModelVersioner:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
    def save(self, model, metadata=None):
        version = datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = self.model_dir / f"model_v{version}.pt"
        meta_path = self.model_dir / f"meta_v{version}.json"
        
        # Save model
        torch.save(model.state_dict(), model_path)
        
        # Create metadata
        checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
        config = {
            "created_at": datetime.now().isoformat(),
            "checksum": checksum,
            "parameters": sum(p.numel() for p in model.parameters()),
            "version": version,
            **metadata
        }
        
        with open(meta_path, "w") as f:
            json.dump(config, f)
            
        return version

    def load(self, version):
        model_path = self.model_dir / f"model_v{version}.pt"
        meta_path = self.model_dir / f"meta_v{version}.json"
        
        # Verify checksum
        with open(meta_path) as f:
            meta = json.load(f)
            
        current_checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if current_checksum != meta["checksum"]:
            raise ValueError("Model file corrupted!")
            
        return model_path, meta