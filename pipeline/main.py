"""
Arbiter sorting pipeline — runs on Raspberry Pi.

Flow: IR sensor interrupt → capture frame → classify → servo sort.

Usage:
    python pipeline/main.py --config configs/config.yaml
    python pipeline/main.py --dry-run   # skips GPIO, prints decisions only
"""

import argparse
import sys
import time
import infer 

import yaml

sys.path.insert(0, "src")
from infer import WasteClassifier


def setup_gpio(cfg):
    """Initialize GPIO pins for IR sensor and servo."""
    import RPi.GPIO as GPIO

    ir_pin = cfg["hardware"]["gpio"]["ir_sensor_pin"]
    servo_pin = cfg["hardware"]["gpio"]["servo_pin"]

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(ir_pin, GPIO.IN)
    GPIO.setup(servo_pin, GPIO.OUT)

    servo = GPIO.PWM(servo_pin, 50)  # 50 Hz
    servo.start(0)

    return ir_pin, servo


def set_servo_angle(servo, angle):
    """Move servo to given angle (0-180)."""
    duty = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)


def capture_frame():
    """Capture a frame from the Pi camera."""
    from picamera2 import Picamera2
    from PIL import Image
    import numpy as np

    cam = Picamera2()
    cam.configure(cam.create_still_configuration())
    cam.start()
    time.sleep(1)  # warm-up
    array = cam.capture_array()
    cam.stop()

    return Image.fromarray(array)


def run(config_path: str, dry_run: bool = False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dry_run = dry_run or cfg.get("dry_run", False)
    classifier = WasteClassifier(config_path)

    if not dry_run:
        ir_pin, servo = setup_gpio(cfg)
        recycle_angle = cfg["hardware"]["servo"]["recycle_angle"]
        waste_angle = cfg["hardware"]["servo"]["waste_angle"]

    print("Arbiter ready. Waiting for items...")

    try:
        while True:
            if dry_run:
                input("Press Enter to simulate item detection...")
                # In dry-run, use a test image or prompt for path
                path = input("Image path (or 'q' to quit): ").strip()
                if path == "q":
                    break
                label, conf = classifier.predict(path)
            else:
                import RPi.GPIO as GPIO
                GPIO.wait_for_edge(ir_pin, GPIO.FALLING)
                print("Item detected — capturing...")
                frame = capture_frame()
                label, conf = classifier.predict(frame)

            print(f"  → {label} ({conf:.2%})")

            if not dry_run:
                angle = recycle_angle if label == "recycle" else waste_angle
                set_servo_angle(servo, angle)
                time.sleep(1)
                set_servo_angle(servo, 90)  # return to neutral

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        if not dry_run:
            import RPi.GPIO as GPIO
            servo.stop()
            GPIO.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arbiter waste sorting pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip GPIO, simulate with manual input")
    args = parser.parse_args()
    run(args.config, dry_run=args.dry_run)
