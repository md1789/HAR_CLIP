import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model

class A_CLIPWrapper(nn.Module):
    def __init__(self, num_classes: int, use_lora: bool = True):
        super().__init__()
        base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Grab vision submodules
        self.vision_model = base_model.vision_model
        self.vision_encoder = self.vision_model.encoder  # <-- transformer block

        if use_lora:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)

        # Classifier head
        hidden_size = base_model.vision_model.config.hidden_size  # 768 for ViT-B/32
        self.classifier = nn.Linear(hidden_size, num_classes)


    def forward(self, pixel_values):
        # Forward through vision model
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output # shape [B, 768]
        return self.classifier(pooled)
