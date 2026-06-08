"""
Pi Camera capture tool for collecting training images.
Set CROP below after using preview.py to determine bounds.

CROP format: (x, y, width, height) in pixels at 1280x720
Set CROP = None to use full frame.

Press Enter to capture, 'q' + Enter to quit.

Usage:
    python pipeline/capturetrash.py [waste|recycle]
"""

import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
from picamera2 import Picamera2

# --- configure these after using preview.py ---
CAPTURE_SIZE = (1280, 720)
CROP = None  # e.g. (160, 0, 720, 720) for a centered square crop
                # (x, y, width, height) — set after previewing

SAVE_DIR_WASTE   = Path("data/raw/waste")
SAVE_DIR_RECYCLE = Path("data/raw/recycle")


def apply_crop(frame, crop):
    if crop is None:
        return frame
    x, y, w, h = crop
    return frame[y:y+h, x:x+w]


def capture_loop(label: str):
    save_dir = SAVE_DIR_WASTE if label == "waste" else SAVE_DIR_RECYCLE
    save_dir.mkdir(parents=True, exist_ok=True)

    cam = Picamera2()
    config = cam.create_still_configuration(
        main={"size": CAPTURE_SIZE, "format": "RGB888"}
    )
    cam.configure(config)
    cam.start()
    time.sleep(2)

    crop_info = f"crop={CROP}" if CROP else "full frame (no crop)"
    print(f"Arbiter Capture Tool — label: {label}")
    print(f"Resolution: {CAPTURE_SIZE}, {crop_info}")
    print(f"Saving to: {save_dir.resolve()}")
    print("Press Enter to capture, 'q' + Enter to quit.\n")

    count = 0

    try:
        while True:
            cmd = input("> ").strip()

            if cmd == "q":
                print(f"\nDone. {count} images saved to {save_dir.resolve()}")
                break

            frame = cam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = apply_crop(frame, CROP)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = save_dir / f"{label}_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)

            count += 1
            print(f"  Saved {filename.name} — size: {frame.shape[1]}x{frame.shape[0]} (total: {count})")

    except KeyboardInterrupt:
        print(f"\nDone. {count} images saved to {save_dir.resolve()}")
    finally:
        cam.stop()


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "waste"
    if label not in ("waste", "recycle"):
        print("Usage: python pipeline/capturetrash.py [waste|recycle]")
        sys.exit(1)
    capture_loop(label)