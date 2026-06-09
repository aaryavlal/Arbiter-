from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()
config = picam2.create_preview_configuration()
picam2.configure(config)
picam2.start_preview(Preview.NULL)
picam2.start()

# Let AWB run and converge for 3 seconds
print("Waiting for AWB to settle...")
time.sleep(3)

# Print what gains AWB settled on
metadata = picam2.capture_metadata()
gains = metadata.get("ColourGains")
print(f"AWB colour gains (red, blue): {gains}")
print(f"Colour temperature estimate: {metadata.get('ColourTemperature')} K")

# Keep running so you can view the MJPEG stream or just leave it open
print("Camera running. Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

picam2.stop()