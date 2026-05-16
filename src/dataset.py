"""
TrashNet dataset loader for binary classification: recycle vs waste.

Label mapping:
  recycle: glass, paper, cardboard, metal, plastic (rigid)
  waste: trash

NOTE: Soft plastic is unresolved — TrashNet does not distinguish rigid from
soft plastic. All plastic is currently mapped to recycle. This needs a
separate data source or manual relabeling to handle properly.
"""

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# TrashNet original classes → binary label
LABEL_MAP = {
    "glass": 1,      # recycle
    "paper": 1,      # recycle
    "cardboard": 1,  # recycle
    "metal": 1,      # recycle
    "plastic": 1,    # recycle (rigid assumed — soft plastic unresolved)
    "trash": 0,      # waste
}

CLASS_NAMES = ["waste", "recycle"]


def get_transform(image_size: int, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


class TrashDataset(Dataset):
    """Expects data_dir to contain subdirectories named by TrashNet class."""

    def __init__(self, data_dir: str, image_size: int = 224, train: bool = True):
        self.data_dir = Path(data_dir)
        self.transform = get_transform(image_size, train)
        self.samples = []

        for class_name, label in LABEL_MAP.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


def get_dataloaders(data_dir: str, image_size: int = 224, batch_size: int = 32,
                    train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Split dataset into train/val/test and return DataLoaders."""
    dataset = TrashDataset(data_dir, image_size, train=True)
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Val/test use eval transforms (no augmentation)
    val_set.dataset = TrashDataset(data_dir, image_size, train=False)
    test_set.dataset = TrashDataset(data_dir, image_size, train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
