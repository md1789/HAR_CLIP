import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModel
from .lora_utils import LoRALinear

class SGLIPWrapper(nn.Module):
    def __init__(self, num_classes: int, use_lora: bool = True):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        if use_lora:
            self._inject_lora(self.vision_model)

        hidden_size = self.vision_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Debug
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[SGLIP LoRA Check] Total params: {total_params:,}")
        print(f"[SGLIP LoRA Check] Trainable params: {trainable_params:,}")
        print(f"[SGLIP LoRA Check] Percentage trainable: {100*trainable_params/total_params:.2f}%\n")

    def _inject_lora(self, model):
        # Loop through vision transformer and patch q_proj/v_proj
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(x in name for x in ["q_proj", "v_proj"]):
                parent = model
                *path, last = name.split(".")
                for p in path:
                    parent = getattr(parent, p)
                setattr(parent, last, LoRALinear(module))  # replace layer with LoRA-wrapped

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        return self.classifier(pooled)
