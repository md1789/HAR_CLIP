import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Default preprocessing (can be overridden by model-specific preprocess)
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])

def get_dataloader(root: str, split: str, batch_size: int, preprocess=None, shuffle=True):
    """
    Load the HAR dataset from data/Structured/{split}/

    Args:
        root (str): path to data/Structured
        split (str): 'train' or 'test'
        batch_size (int): batch size
        preprocess (callable): transform (from CLIP/A-CLIP/SGLIP or default)
        shuffle (bool): shuffle data (default True)

    Returns:
        dataloader, dataset
    """
    data_dir = os.path.join(root, split)
    transform = preprocess if preprocess is not None else default_transform

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader, dataset

def compute_metrics(preds, labels):
    """
    Compute accuracy given predictions and labels.
    Extend with F1, confusion matrix, etc. later.
    """
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return {"accuracy": correct / total}
