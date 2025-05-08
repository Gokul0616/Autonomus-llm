import torch
import torch.nn as nn
from .base_model import BaseModel, ProductionWrapper

class MoE(BaseModel):
    def __init__(self, tokenizer, config):
        super().__init__(tokenizer, config)
        self.embedding = nn.Embedding(tokenizer.vocab_size, config['d_model'])
        self.experts = nn.ModuleList([Expert(config) for _ in range(config['num_experts'])])
        self.gate = nn.Linear(config['d_model'], config['num_experts'])
        self.ln = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], tokenizer.vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        gates = torch.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        x = torch.einsum('btnd,btn->btd', expert_outputs, gates)
        return self.head(self.ln(x))
    
    def generate(self, prompt, **kwargs):
        return self._generation_engine(prompt, **kwargs)
        
    class Expert(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ffn = nn.Sequential(
                nn.Linear(config['d_model'], config['d_ffn']),
                nn.GELU(),
                nn.Dropout(config['dropout']),
                nn.Linear(config['d_ffn'], config['d_model'])
            )
            
        def forward(self, x):
            return self.ffn(x)

class ProductionMoE(ProductionWrapper):
    def generate(self, prompt, **kwargs):
        with torch.inference_mode():
            return super().generate(prompt, **kwargs)