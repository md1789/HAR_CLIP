import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model

class CLIPWrapper(nn.Module):
    def __init__(self, num_classes: int, use_lora: bool = True):
        super().__init__()
        base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model = base_model.vision_model
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        if use_lora:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],  # only in vision attention layers
                lora_dropout=0.1,
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            self.vision_model = get_peft_model(self.vision_model, peft_config)

        hidden_size = base_model.config.projection_dim
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        vision_outputs = self.vision_model(pixel_values)
        pooled = vision_outputs.pooler_output  # CLS token embedding
        logits = self.classifier(pooled)
        return logits
