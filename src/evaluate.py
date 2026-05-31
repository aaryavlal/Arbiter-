"""
Evaluate trained model on test set. Reports F1, precision, recall per class.
"""

import argparse

import torch
import timm
import yaml
from sklearn.metrics import classification_report

from dataset import get_dataloaders, CLASS_NAMES


def evaluate(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    image_size = cfg["model"]["image_size"]
    checkpoint_path = cfg["model"]["checkpoint_path"]
    processed_dir = cfg["data"]["processed_dir"]
    batch_size = cfg["training"]["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(
        processed_dir, image_size=image_size, batch_size=batch_size
    )

    model = timm.create_model(cfg["model"]["name"], pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    evaluate(args.config)
