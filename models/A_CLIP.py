import torch.nn as nn
from . import alpha_clip  # uses our self-contained alpha_clip.py
from peft import LoraConfig, get_peft_model


class A_CLIPWrapper(nn.Module):
    def __init__(self, num_classes: int, use_lora: bool = True, model_name: str = "ViT-B/16"):
        super().__init__()

        # Load AlphaCLIP model + preprocessing transform
        self.model, self.preprocess = alpha_clip.load(model_name, lora_adapt=use_lora)
        self.hidden_size = getattr(self.model, "hidden_size", 512)

        if use_lora:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],  # adjust once internals known
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_CLS"
            )
            self.model = get_peft_model(self.model, peft_config)

        # Classifier head
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, pixel_values):
        vision_outputs = self.model(pixel_values)
        pooled = vision_outputs["pooler_output"]
        logits = self.classifier(pooled)
        return logits
