import torch
import torch.nn as nn
from typing import Dict, List

class FusionHead(nn.Module):
    def __init__(self, model_dims: Dict[str, int], fusion_dim: int = 512):
        super().__init__()
        self.projectors = nn.ModuleDict({
            name: nn.Linear(dim, fusion_dim)
            for name, dim in model_dims.items()
        })
        self.attention = nn.MultiheadAttention(fusion_dim, num_heads=4)
        self.fc_out = nn.Linear(fusion_dim, fusion_dim)
        
    def forward(self, outputs: Dict[str, torch.Tensor]):
        projected = [proj(output) for output, proj in zip(outputs.values(), self.projectors.values())]
        stacked = torch.stack(projected, dim=0)
        
        attn_out, _ = self.attention(stacked, stacked, stacked)
        return self.fc_out(attn_out.mean(dim=0))

class HybridFusion:
    def __init__(self, models: List[str], config_path: str = "configs/model_config.yaml"):
        self.models = models
        self.config = self._load_config(config_path)
        self.fusion_head = FusionHead(self._get_model_dims())
        self.optimizer = torch.optim.AdamW(self.fusion_head.parameters(), lr=1e-4)
        
    def fuse_outputs(self, model_outputs: Dict[str, torch.Tensor]):
        with torch.inference_mode():
            return self.fusion_head(model_outputs)
    
    def adaptive_weighting(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor):
        self.optimizer.zero_grad()
        fused = self.fusion_head(outputs)
        loss = F.cross_entropy(fused, targets)
        loss.backward()
        self.optimizer.step()
        return fused
    
    def save_fusion(self, path: str):
        torch.save({
            'state_dict': self.fusion_head.state_dict(),
            'config': self.config
        }, path)
        
    @classmethod
    def load_fusion(cls, path: str, models: List[str]):
        data = torch.load(path)
        instance = cls(models)
        instance.fusion_head.load_state_dict(data['state_dict'])
        return instance