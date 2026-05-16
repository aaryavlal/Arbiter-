"""
Fine-tune MobileNetV3-small for binary waste classification.

Trains on a regular machine, saves best checkpoint to models/.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import timm
import yaml

from dataset import get_dataloaders


def build_model():
    """MobileNetV3-small with binary classification head."""
    model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=2)
    return model


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    image_size = cfg["model"]["image_size"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["learning_rate"]
    class_weights = cfg["training"]["class_weights"]
    checkpoint_path = Path(cfg["model"]["checkpoint_path"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders(
        "data/processed", image_size=image_size, batch_size=batch_size
    )

    model = build_model().to(device)

    # Class-weighted loss: bias toward recycle recall
    weights = torch.tensor([class_weights["waste"], class_weights["recycle"]]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            _, predicted = outputs.max(1)
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

    print(f"Training complete. Best val acc: {best_val_acc:.4f}")


def evaluate_epoch(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
