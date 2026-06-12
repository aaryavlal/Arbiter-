"""
TrashNet dataset loader for 3-class classification: recycle vs waste vs empty.

Label mapping:
  recycle: glass, paper, cardboard, metal, plastic (rigid)
  waste: trash
  empty: nothing present under the camera
"""

import os
from collections import defaultdict
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms

# Source class directory → integer label
LABEL_MAP = {
    "glass": 1,      # recycle
    "paper": 1,      # recycle
    "cardboard": 1,  # recycle
    "metal": 1,      # recycle
    "plastic": 1,    # recycle
    "trash": 0,      # waste
    "empty": 2,      # nothing present
}

# Index = integer label. Keep "empty" last so waste/recycle indices stay stable.
CLASS_NAMES = ["waste", "recycle", "empty"]


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

    def __init__(self, data_dir: str, image_size: int = 224, train: bool = True,
                 cfg: dict = None):
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

        # Per-sample weights based on filename prefix (Pi-captured vs TrashNet)
        self.sample_weights = self._compute_sample_weights(cfg)

    def _compute_sample_weights(self, cfg):
        """Assign sample weights: Pi-captured images (paper_*/trash_*) get
        higher weight than TrashNet originals."""
        if cfg is None:
            return [1.0] * len(self.samples)

        pi_weights_by_prefix = {
            "paper": cfg["training"]["pi_sample_weight_paper"],
            "trash": cfg["training"]["pi_sample_weight_trash"],
        }
        default_weight = cfg["training"]["trashnet_sample_weight"]

        weights = []
        for img_path, _ in self.samples:
            fname = Path(img_path).stem
            w = default_weight
            for prefix, pw in pi_weights_by_prefix.items():
                if fname.startswith(prefix + "_"):
                    w = pw
                    break
            weights.append(w)
        return weights

    def print_composition(self, cfg):
        """Print dataset composition with Pi vs TrashNet breakdown."""
        if cfg is None:
            return
        pi_paper_w = cfg["training"]["pi_sample_weight_paper"]
        pi_trash_w = cfg["training"]["pi_sample_weight_trash"]
        tn_w = cfg["training"]["trashnet_sample_weight"]

        n_pi_paper = 0
        n_pi_trash = 0
        n_empty = 0
        n_tn = 0
        for img_path, label in self.samples:
            fname = Path(img_path).stem
            if label == LABEL_MAP["empty"]:
                n_empty += 1
            elif fname.startswith("paper_"):
                n_pi_paper += 1
            elif fname.startswith("trash_"):
                n_pi_trash += 1
            else:
                n_tn += 1

        print("Dataset composition:")
        print(f"  Pi paper:   {n_pi_paper:>5} × {pi_paper_w:>4.1f} = {n_pi_paper * pi_paper_w:.0f}")
        print(f"  Pi trash:   {n_pi_trash:>5} × {pi_trash_w:>4.1f} = {n_pi_trash * pi_trash_w:.0f}")
        print(f"  Empty:      {n_empty:>5} × {tn_w:>4.1f} = {n_empty * tn_w:.0f}")
        print(f"  TrashNet:   {n_tn:>5} × {tn_w:>4.1f} = {n_tn * tn_w:.0f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


def get_dataloaders(data_dir: str, image_size: int = 224, batch_size: int = 32,
                    train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42,
                    cfg: dict = None):

    # Three separate datasets with correct transforms
    train_dataset = TrashDataset(data_dir, image_size, train=True, cfg=cfg)
    val_dataset   = TrashDataset(data_dir, image_size, train=False, cfg=cfg)
    test_dataset  = TrashDataset(data_dir, image_size, train=False, cfg=cfg)

    train_dataset.print_composition(cfg)

    total = len(train_dataset)
    train_size = int(total * train_ratio)
    val_size   = int(total * val_ratio)
    test_size  = total - train_size - val_size

    # Stratified split — preserve class ratio across train/val/test.
    # Group indices per class so every class is split independently.
    class_indices = defaultdict(list)
    for i, (_, label) in enumerate(train_dataset.samples):
        class_indices[label].append(i)

    def split_indices(idx_list, train_r, val_r, gen):
        idx_list = torch.tensor(idx_list)[torch.randperm(len(idx_list), generator=gen)].tolist()
        t = int(len(idx_list) * train_r)
        v = int(len(idx_list) * val_r)
        return idx_list[:t], idx_list[t:t+v], idx_list[t+v:]

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices, test_indices = [], [], []
    for label in sorted(class_indices):
        t, v, te = split_indices(class_indices[label], train_ratio, val_ratio, generator)
        train_indices += t
        val_indices += v
        test_indices += te

    # Apply the index splits to the correct dataset
    from torch.utils.data import Subset
    train_set = Subset(train_dataset, train_indices)
    val_set   = Subset(val_dataset,   val_indices)
    test_set  = Subset(test_dataset,  test_indices)

    # Weighted sampling for training: Pi-captured images are drawn more often
    train_weights = [train_dataset.sample_weights[i] for i in train_indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights),
                                    replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
