# reward_model.py
import torch
from torch import nn

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Sequential(
            nn.Linear(base_model.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        features = self.base_model(x, output_hidden_states=True).hidden_states[-1]
        return self.reward_head(features.mean(dim=1))

    def compute_reward(self, code_validation, execution_time):
        inputs = self._format_inputs(code_validation, execution_time)
        with torch.no_grad():
            return self(inputs).item()

    def _format_inputs(self, validation, exec_time):
        return torch.tensor([
            validation['test_passed'],
            validation['valid_syntax'],
            exec_time / self.timeout,
            len(validation['output']),
            validation['error'] == ""
        ], dtype=torch.float32)