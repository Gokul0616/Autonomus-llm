import torch
import torch.nn as nn
from .base_model import BaseModel, ProductionWrapper
from torch.nn import functional as F

class RWKV(BaseModel):
    def __init__(self, tokenizer, config):
        super().__init__(tokenizer, config)
        self.embedding = nn.Embedding(tokenizer.vocab_size, config['d_model'])
        self.blocks = nn.ModuleList([
            RWKVBlock(config) for _ in range(config['n_layers'])
        ])
        self.ln_out = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], tokenizer.vocab_size)
        
    def forward(self, input_ids, state=None):
        x = self.embedding(input_ids)
        new_state = []
        for i, block in enumerate(self.blocks):
            x, s = block(x, state[i] if state else None)
            new_state.append(s)
        return self.head(self.ln_out(x)), new_state
    
    def generate(self, prompt, **kwargs):
        return self._rwkv_generate(prompt, **kwargs)

class RWKVBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['d_model'])
        self.time_mix = TimeMix(config)
        self.ln2 = nn.LayerNorm(config['d_model'])
        self.channel_mix = ChannelMix(config)
        
    def forward(self, x, state=None):
        xln = self.ln1(x)
        dx, state = self.time_mix(xln, state)
        x = x + dx
        
        xln = self.ln2(x)
        dx = self.channel_mix(xln)
        return x + dx, state

class TimeMix(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        
        self.time_decay = nn.Parameter(torch.empty(config['d_model']))
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        
        self.key = nn.Linear(self.d_model, self.d_model, bias=False)
        self.value = nn.Linear(self.d_model, self.d_model, bias=False)
        self.receptance = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.orthogonal_(self.time_decay)
        
    def forward(self, x, state=None):
        shifted = self.time_shift(x)
        if state is not None:
            shifted = torch.cat([state.unsqueeze(1), shifted], dim=1)
            
        k = self.key(shifted)
        v = self.value(shifted)
        r = self.receptance(x)
        
        # WKV computation
        wkv = self._wkv(k, v)
        return r * wkv, shifted[:, -1]

    def _wkv(self, k, v):
        # Efficient WKV implementation
        pass  # Actual complex implementation here

class ChannelMix(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config['d_model'], config['d_ffn'], bias=False)
        self.value = nn.Linear(config['d_ffn'], config['d_model'], bias=False)
        self.receptance = nn.Linear(config['d_model'], config['d_model'], bias=False)
        
    def forward(self, x):
        k = torch.square(torch.relu(self.key(x)))
        v = self.value(k)
        r = torch.sigmoid(self.receptance(x))
        return r * v