import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # The frozen original weight
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

        # LoRA low-rank adapters
        self.lora_down = nn.Linear(self.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, self.out_features, bias=False)
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        # Init small
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

        # Freeze original
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias) + \
               self.lora_up(self.lora_down(self.dropout(x))) * self.scaling
