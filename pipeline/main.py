"""
Arbiter sorting pipeline — laptop webcam version for testing/training.

Flow: press Enter → capture frame from webcam → classify → print result.

Usage:
    python pipeline/main.py --config configs/config.yaml
"""

import argparse
import sys
import threading

import cv2
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
    classifier = WasteClassifier(config_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        sys.exit(1)

    win = "Arbiter Preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    print("Arbiter ready. Using laptop webcam.")
    print("Press Enter in this terminal to capture and classify, or Ctrl+C to quit.\n")

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

    frame = None
    try:
        while not quit_event.is_set():
            ret, grabbed = cap.read()
            if ret:
                frame = grabbed
                cv2.imshow(win, frame)

            cv2.waitKey(30)

            if capture_event.is_set() and frame is not None:
                capture_event.clear()
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                flash = frame.copy()
                flash[:] = (255, 255, 255)
                cv2.addWeighted(flash, 0.4, frame, 0.6, 0, frame)
                cv2.imshow(win, frame)
                cv2.waitKey(80)
                label, conf = classifier.predict(image)
                print(f"  DEBUG waste_prob check: {label} {conf:.4f}")
                print(f"  → {label} ({conf:.2%})")

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        quit_event.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arbiter waste sorting pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)
