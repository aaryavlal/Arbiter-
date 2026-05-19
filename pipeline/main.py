"""
Arbiter sorting pipeline — laptop webcam version for testing/training.

Flow: press Enter → capture frame from webcam → classify → print result.

Usage:
    python pipeline/main.py --config configs/config.yaml
"""

import argparse
import sys
import time

import cv2
import yaml
from PIL import Image

sys.path.insert(0, "src")
from infer import WasteClassifier


def capture_frame(cap):
    """Capture a frame from the laptop webcam."""
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture frame from webcam")

    # OpenCV uses BGR, PIL expects RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def run(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    classifier = WasteClassifier(config_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        sys.exit(1)

    print("Arbiter ready. Using laptop webcam.")
    print("Press Enter to capture and classify, or 'q' to quit.\n")

    try:
        while True:
            cmd = input("> ").strip()
            if cmd == "q":
                break

            frame = capture_frame(cap)
            label, conf = classifier.predict(frame)
            print(f"  → {label} ({conf:.2%})")

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arbiter waste sorting pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)
