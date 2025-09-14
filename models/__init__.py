# models/__init__.py

def get_model(name: str, num_classes: int, use_lora: bool = True):
    """
    Returns the requested model wrapper.

    Args:
        name (str): One of ["clip", "a_clip", "sglip"]
        num_classes (int): Number of output classes
        use_lora (bool): Whether to use LoRA adapters

    Returns:
        torch.nn.Module: The requested model
    """
    if name == "clip":
        from .CLIP import CLIPWrapper
        return CLIPWrapper(num_classes=num_classes, use_lora=use_lora)

    elif name == "a_clip":
        from .A_CLIP import A_CLIPWrapper
        return A_CLIPWrapper(num_classes=num_classes, use_lora=use_lora)

    elif name == "sglip":
        from .SGLIP import SGLIPWrapper
        return SGLIPWrapper(num_classes=num_classes, use_lora=use_lora)

    else:
        raise ValueError(f"Unknown model name: {name}. "
                         f"Choose from ['clip', 'a_clip', 'sglip'].")
