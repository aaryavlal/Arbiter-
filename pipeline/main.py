"""
Arbiter sorting pipeline — Pi Camera version.

Flow: press Enter → capture frame from picamera2 → classify → print result.

Set CROP below after using preview.py to determine bounds.
CROP format: (x, y, width, height) in pixels at CAPTURE_SIZE resolution.
Set CROP = None to use full frame.

Usage:
    python pipeline/main.py --config configs/config.yaml
"""
import os
from datetime import datetime
import argparse
import sys
import time
import threading
from libcamera import controls

from PIL import Image
from picamera2 import Picamera2

sys.path.insert(0, "src")
from infer import WasteClassifier


# --- configure after using preview.py ---
CAPTURE_SIZE = (1280, 720)
CROP = None  # e.g. (160, 0, 720, 720) for a centered square crop


def apply_crop(frame, crop):
    if crop is None:
        return frame
    x, y, w, h = crop
    return frame[y:y+h, x:x+w]


def run(config_path: str):
    classifier = WasteClassifier(config_path)

    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": CAPTURE_SIZE, "format": "BGR888"}
    )
    cam.configure(config)
    cam.start()
    cam.set_controls({
        "AwbMode": controls.AwbModeEnum.Fluorescent,
        "AfMode": controls.AfModeEnum.Continuous
    })
    time.sleep(2)

    crop_info = f"crop={CROP}" if CROP else "full frame (no crop)"
    print(f"Arbiter ready. Using picamera2. Resolution: {CAPTURE_SIZE}, {crop_info}")
    print("Press Enter to capture and classify. \n")

    capture_event = threading.Event()
    quit_event = threading.Event()

    def input_listener():
        while not quit_event.is_set():
            try:
                input()
                capture_event.set()
            except EOFError:
                break

    threading.Thread(target=input_listener, daemon=True).start()

    try:
        while not quit_event.is_set():
            if capture_event.is_set():
                capture_event.clear()
                frame = cam.capture_array()
                frame = apply_crop(frame, CROP)
                image = Image.fromarray(frame)
                label, conf = classifier.predict(image)
                # os.makedirs("data/test", exist_ok=True)
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # image.save(f"data/test/{timestamp}_{label}_{conf:.2f}.jpg", "JPEG", quality=95)
                print(f"  → {label} ({conf:.2%})")
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        quit_event.set()
        cam.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arbiter waste sorting pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)
