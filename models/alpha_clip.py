import os
import hashlib
import urllib
import warnings
from typing import Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from .model import build_model  # <- TODO: supply AlphaCLIP model builder here
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

# -------------------------------------------------------------------
# Available pretrained AlphaCLIP (extend with real URLs/checkpoints)
# -------------------------------------------------------------------
_MODELS = {
    "ViT-B/16": "https://example.com/alpha_clip_vit_b16.pt",  # placeholder
}

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        while True:
            buffer = source.read(8192)
            if not buffer:
                break
            output.write(buffer)
    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px: int):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def available_models() -> List[str]:
    return list(_MODELS.keys())


def load(name: str,
         device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         download_root: str = None,
         lora_adapt: bool = False,
         rank: int = 16):
    """
    Load an AlphaCLIP model.
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name],
                               download_root or os.path.expanduser("~/.cache/alpha_clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available = {available_models()}")

    # Load checkpoint
    state_dict = torch.load(model_path, map_location="cpu")

    # Build model using project-specific builder
    model = build_model(state_dict, lora_adapt=lora_adapt, rank=rank).to(device)
    if str(device) == "cpu":
        model.float()

    preprocess = _transform(model.visual.input_resolution)
    return model, preprocess


def tokenize(texts: Union[str, List[str]], context_length: int = 77,
             truncate: bool = True) -> torch.Tensor:
    """
    Tokenize input texts into AlphaCLIP-compatible tokens.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result
