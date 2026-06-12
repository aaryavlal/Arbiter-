"""
Fine-tune MobileNetV3-small for binary waste classification.

Trains on a regular machine, saves best checkpoint to models/.
"""

import argparse
from pathlib import Path
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import timm
import yaml
from dataset import get_dataloaders, CLASS_NAMES
import random
import numpy as np

from dotenv import load_dotenv
load_dotenv()

def build_model(name: str, dropout: float = 0.3):
    """Classification head (one logit per class) on the configured backbone."""
    num_classes = len(CLASS_NAMES)
    model = timm.create_model(name, pretrained=True, num_classes=num_classes)
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=dropout),
        torch.nn.Linear(in_features, num_classes)
    )
    return model


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    image_size = cfg["model"]["image_size"]
    threshold = cfg["inference"]["confidence_threshold"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["learning_rate"]
    checkpoint_path = Path(cfg["model"]["checkpoint_path"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=cfg["data"]["dir"], image_size=image_size, batch_size=batch_size, seed=seed,
        cfg=cfg
    )

    model = build_model(cfg["model"]["name"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        val_acc = evaluate_epoch(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} — "
              f"Loss: {running_loss/len(train_loader):.4f} — "
              f"Train Acc: {train_acc:.4f} — Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best checkpoint ({val_acc:.4f})")

        scheduler.step(val_acc)

    print(f"Training complete. Best val acc: {best_val_acc:.4f}")

    # Per-class breakdown — run after training loop completes
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print("\nPer-class breakdown (test set):")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))


def evaluate_epoch(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
