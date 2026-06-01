"""
Webcam capture tool for collecting training images.

Hold Enter to continuously capture frames, release to stop.
Images are saved to data/raw/trash/ with timestamps.

Usage:
    python pipeline/capturetrash.py
"""

import sys
import time
import threading
from pathlib import Path
from datetime import datetime

import cv2

SAVE_DIR = Path("data/raw/trash")
CAMERA_INDEX = 0
CAPTURE_INTERVAL = 4 # seconds between captures when holding Enter


def capture_loop():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: could not open webcam.")
        sys.exit(1)

    print("Arbiter Capture Tool")
    print(f"Saving to: {SAVE_DIR.resolve()}")
    print("Hold Enter to capture continuously, 'q' + Enter to quit.\n")

    count = 0
    capturing = False
    capture_thread = None

    def continuous_capture():
        nonlocal count
        while capturing:
            ret, frame = cap.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = SAVE_DIR / f"waste_{timestamp}.jpg"
                cv2.imwrite(str(filename), frame)
                count += 1
                print(f"  Saved {filename.name} (total: {count})", flush=True)
            time.sleep(CAPTURE_INTERVAL)

    def preview_loop():
        cv2.namedWindow("Arbiter Capture — press Q to quit preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Arbiter Capture — press Q to quit preview", 1280, 720)
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Arbiter Capture — press Q to quit preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    preview_thread = threading.Thread(target=preview_loop, daemon=True)
    preview_thread.start()

    try:
        while True:
            cmd = input("> ").strip()

            if cmd == "q":
                print(f"\nDone. {count} images saved to {SAVE_DIR.resolve()}")
                break

            # Any Enter press starts/continues capture
            capturing = True
            capture_thread = threading.Thread(target=continuous_capture, daemon=True)
            capture_thread.start()
            input("  Capturing... press Enter to stop.\n")
            capturing = False
            if capture_thread:
                capture_thread.join()

    except KeyboardInterrupt:
        capturing = False
        print(f"\nDone. {count} images saved to {SAVE_DIR.resolve()}")
    finally:
        cap.release()


if __name__ == "__main__":
    capture_loop()