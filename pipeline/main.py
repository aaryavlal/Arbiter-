"""
Arbiter sorting pipeline — continuous / autonomous Pi Camera version.

Unlike pipeline/main.py (which waits for Enter), this script runs on its own:

  every 2s  → capture a frame and classify it (presence check)
  non-empty → wait for the camera to autofocus, capture a sharp frame,
              re-classify, then drive the servo to sort the item
  recycle   → servo travels fully one way, then back to mid
  trash     → servo travels fully the other way, then back to mid
  empty     → keep polling

The servo (from swervo/motor-test.py) is driven with raw numeric values instead
of .mid()/.min()/.max() so the end positions are easy to tune below.

Set CROP after using preview.py to determine bounds.
CROP format: (x, y, width, height) in pixels at CAPTURE_SIZE resolution.
Set CROP = None to use full frame.

Usage:
    python pipeline/main_continuous.py --config configs/config.yaml
"""
import argparse
import sys
import time

from gpiozero import Servo
from libcamera import controls
from PIL import Image
from picamera2 import Picamera2

sys.path.insert(0, "src")
from infer import WasteClassifier



SERVO_PIN = 18           # 12 pin 


SERVO_MID = 0.6 #tunable for midpoint (its still a little off)
SERVO_TRAVEL = 0.4       # how far to swing from mid in either direction

SERVO_RECYCLE = max(-1.0, min(1.0, SERVO_MID + SERVO_TRAVEL))  # one way for recycle
SERVO_WASTE = max(-1.0, min(1.0, SERVO_MID - SERVO_TRAVEL))    # other way for trash
SERVO_MOVE_PAUSE = 2.0   # seconds to hold a position before moving again

SERVO_SPEED = 0.75     
SERVO_STEP_INTERVAL = 0.02 

# --- loop timing ---
POLL_INTERVAL = 2.0      
SORT_COOLDOWN = 4.0     


CAPTURE_SIZE = (1280, 720)
CROP = None  


def apply_crop(frame, crop):
    if crop is None:
        return frame
    x, y, w, h = crop
    return frame[y:y+h, x:x+w]


def capture_image(cam):
    """Grab the current frame as a (cropped) PIL Image. No focus wait."""
    frame = apply_crop(cam.capture_array(), CROP)
    return Image.fromarray(frame)


def focus_and_capture(cam):
    """
    Run a deliberate autofocus scan, wait for it to lock, then capture.

    cam.autofocus_cycle() blocks until the AF scan completes and returns True on
    success. We capture regardless of the result (a "failed" lock still gives the
    best available focus), but report it so a blurry capture is explainable.
    """
    focused = cam.autofocus_cycle()
    if not focused:
        print("  (autofocus did not lock — capturing anyway)")
    return capture_image(cam)


def glide_servo(servo, start, target):
    """Move the servo from start to target gradually at SERVO_SPEED (value-units/sec)."""
    distance = target - start
    steps = max(1, int(abs(distance) / SERVO_SPEED / SERVO_STEP_INTERVAL))
    for i in range(1, steps + 1):
        servo.value = start + distance * (i / steps)
        time.sleep(SERVO_STEP_INTERVAL)
    servo.value = target


def move_servo(servo, target):
    """Drive the servo to an extreme, hold, then return to mid."""
    glide_servo(servo, SERVO_MID, target)
    time.sleep(SERVO_MOVE_PAUSE)
    glide_servo(servo, target, SERVO_MID)
    time.sleep(SERVO_MOVE_PAUSE)
    # stop sending pulses so the servo doesn't jitter while idle.
    servo.detach()


def sort(servo, label):
    """Move the motor according to the model's verdict."""
    if label == "recycle":
        print("  → sorting RECYCLE")
        move_servo(servo, SERVO_RECYCLE)
    elif label == "waste":
        print("  → sorting TRASH")
        move_servo(servo, SERVO_WASTE)


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
       #focus scan per object detected
        "AfMode": controls.AfModeEnum.Auto,
    })
    time.sleep(2)

    # Motor starts in mid.
    servo = Servo(SERVO_PIN)
    servo.value = SERVO_MID
    time.sleep(SERVO_MOVE_PAUSE)
    servo.detach()

    crop_info = f"crop={CROP}" if CROP else "full frame (no crop)"
    print(f"Arbiter running (continuous). Resolution: {CAPTURE_SIZE}, {crop_info}")
    print(f"Polling every {POLL_INTERVAL:.0f}s. Press Ctrl-C to stop.\n")

    try:
        while True:
            # Presence check — quick capture + classify. Empty when nothing is here.
            label, conf = classifier.predict(capture_image(cam))

            if label == "empty":
                print(f"empty ({conf:.2%})")
                time.sleep(POLL_INTERVAL)
                continue

            # Something is here — get a focused frame and make the real decision.
            print(f"object detected ({label} {conf:.2%}) — focusing…")
            image = focus_and_capture(cam)
            label, conf = classifier.predict(image)
            print(f"  → {label} ({conf:.2%})")

            if label != "empty":
                sort(servo, label)
                time.sleep(SORT_COOLDOWN)
            else:
                time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        servo.value = SERVO_MID
        time.sleep(SERVO_MOVE_PAUSE)
        servo.detach()
        cam.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arbiter continuous waste sorting pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)
