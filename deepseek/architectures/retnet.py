import torch
import torch.nn as nn
from .base_model import BaseModel, ProductionWrapper
from einops import rearrange

class RetNet(BaseModel):
    def __init__(self, tokenizer, config):
        super().__init__(tokenizer, config)
        self.embedding = nn.Embedding(tokenizer.vocab_size, config['d_model'])
        self.layers = nn.ModuleList([
            RetNetLayer(config) for _ in range(config['n_layers'])
        ])
        self.norm = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], tokenizer.vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))
    
    def generate(self, prompt, **kwargs):
        return self._retentive_generate(prompt, **kwargs)
    
    class RetNetLayer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.retention = MultiScaleRetention(config)
            self.ffn = SwishGLU(config['d_model'], config['d_ffn'])
            
        def forward(self, x):
            x = x + self.retention(x)
            return x + self.ffn(x)

class MultiScaleRetention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.head_dim = self.d_model // self.n_heads
        
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.gate = nn.Linear(self.d_model, self.d_model)
        
        self.group_norm = nn.GroupNorm(self.n_heads, self.d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        
        # Retention mechanism
        retention = torch.einsum('bhqd,bhkd->bhqk', q, k) / (self.head_dim ** 0.5)
        retention = torch.softmax(retention, dim=-1)
        
        output = torch.einsum('bhqk,bhkd->bhqd', retention, v)
        output = rearrange(output, 'b h t d -> b t (h d)')
        
        return self.group_norm(self.gate(x) * output)

class SwishGLU(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn)
        self.w2 = nn.Linear(d_model, d_ffn)
        self.w3 = nn.Linear(d_ffn, d_model)
        
    def forward(self, x):
        return self.w3(torch.sigmoid(self.w1(x)) * self.w2(x))