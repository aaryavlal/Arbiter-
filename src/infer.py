"""
Single-image inference. Returns 'recycle', 'waste', or 'empty' with a
confidence score.

Used standalone for testing and called by pipeline/main.py on the Pi.
"""

import argparse
from pathlib import Path

import torch
import timm
import yaml
from PIL import Image
from torchvision import transforms

from dataset import CLASS_NAMES


class WasteClassifier:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.image_size = cfg["model"]["image_size"]
        self.threshold = cfg["inference"]["confidence_threshold"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = len(CLASS_NAMES)
        self.model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=num_classes)
        in_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(in_features, num_classes)
)
        self.model.load_state_dict(
            torch.load(cfg["model"]["checkpoint_path"], map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def predict(self, image) -> tuple[str, float]:
        """
        Classify an image as recycle, waste, or empty.

        Args:
            image: PIL Image or path to image file.

        Returns:
            (class_name, confidence) e.g. ('recycle', 0.92)
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)[0]

        empty_idx = CLASS_NAMES.index("empty")
        waste_idx = CLASS_NAMES.index("waste")
        recycle_idx = CLASS_NAMES.index("recycle")

        # Nothing under the camera — short-circuit before any sorting decision.
        if probs.argmax().item() == empty_idx:
            return "empty", probs[empty_idx].item()

        # Conservative bias toward waste: only call it recycle when the model
        # is confident it isn't waste.
        waste_prob = probs[waste_idx].item()
        if waste_prob >= self.threshold:
            return "waste", waste_prob
        return "recycle", probs[recycle_idx].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    classifier = WasteClassifier(args.config)
    label, conf = classifier.predict(args.image)
    print(f"{label} ({conf:.2%})")
