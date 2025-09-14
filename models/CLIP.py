import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from peft import LoraConfig, get_peft_model

class CLIPWrapper(nn.Module):
    def __init__(self, num_classes: int, use_lora: bool = True):
        super().__init__()
        # Use vision-only backbone
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Apply LoRA if requested
        if use_lora:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],  # LoRA hooks inside attention
                lora_dropout=0.1,
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            self.model = get_peft_model(self.model, peft_config)

        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        pooled = outputs.pooler_output  # CLS token embedding
        logits = self.classifier(pooled)
        return logits
