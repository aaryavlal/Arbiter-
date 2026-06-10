"""
Pi Camera capture tool with live HTTP preview.
Open http://<PI_IP>:8000 in your browser to see the live feed.
Press Enter in the terminal to capture, 'q' + Enter to quit.

Matches preview.py characteristics: same video config, fixed white
balance / colour gains, continuous autofocus, and warmup, so captured
images look the same as the live stream and as inference-time frames.

CROP format: (x, y, width, height) in pixels at STREAM_SIZE.
Set CROP = None to use the full frame.

Usage:
    python pipeline/capturetrash.py
"""

import time
import threading
from pathlib import Path
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
from picamera2 import Picamera2
from libcamera import controls

# --- config (kept consistent with preview.py) ---
STREAM_SIZE = (1280, 720)
PORT = 8000
# COLOUR_GAINS = (2.2, 1.9)   # fixed AWB gains, matches preview.py
JPEG_QUALITY_STREAM = 70    # for the browser preview only
CROP = None                 # e.g. (160, 0, 720, 720); set after previewing

SAVE_DIR = Path("data/raw/paper")

# --- shared frame state ---
frame_lock = threading.Lock()
latest_array = None   # BGR array, used for saving
latest_jpeg = None    # encoded JPEG, used for the preview stream


def apply_crop(frame, crop):
    if crop is None:
        return frame
    x, y, w, h = crop
    return frame[y:y + h, x:x + w]


def camera_thread():
    global latest_array, latest_jpeg

    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": STREAM_SIZE, "format": "RGB888"}
    )
    cam.configure(config)
    cam.start()
    # cam.set_controls({"AfMode": 2})  # 2 = continuous autofocus

    metadata = cam.capture_metadata()
    print("ColourGains (auto, pre-lock):", metadata["ColourGains"])
    cam.set_controls({
        "AwbMode": controls.AwbModeEnum.Fluorescent,
        "AfMode": controls.AfModeEnum.Continuous ,
    })

    time.sleep(2)

    while True:
        frame = cam.capture_array()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  convert RGB to BGR for OpenCV
        frame = apply_crop(frame, CROP)

        _, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY_STREAM]
        )
        with frame_lock:
            latest_array = frame
            latest_jpeg = jpeg.tobytes()


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress request logs

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html><body style="margin:0;background:#000">
                <img src="/stream"
                     style="width:100%;height:100vh;object-fit:contain">
                </body></html>
            """)

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_jpeg
                    if frame:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.05)  # ~20fps
            except (BrokenPipeError, ConnectionResetError):
                pass


def capture_loop():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    crop_info = f"crop={CROP}" if CROP else "full frame (no crop)"
    print(f"Arbiter Capture Tool — saving to: {SAVE_DIR.resolve()}")
    print(f"Resolution: {STREAM_SIZE}, {crop_info}")
    print(f"Live preview: http://<PI_IP>:{PORT}")
    print("Press Enter to capture, 'q' + Enter to quit.\n")

    # wait for the first frame so an immediate Enter doesn't fail
    while True:
        with frame_lock:
            ready = latest_array is not None
        if ready:
            break
        time.sleep(0.1)

    count = 0
    while True:
        cmd = input("> ").strip()
        if cmd == "q":
            print(f"\nDone. {count} images saved to {SAVE_DIR.resolve()}")
            break

        with frame_lock:
            frame = None if latest_array is None else latest_array.copy()
        if frame is None:
            print("  No frame available yet, try again.")
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = SAVE_DIR / f"paper_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)

        count += 1
        print(f"  Saved {filename.name} — "
              f"size: {frame.shape[1]}x{frame.shape[0]} (total: {count})")


if __name__ == "__main__":
    cam_t = threading.Thread(target=camera_thread, daemon=True)
    cam_t.start()

    server = HTTPServer(("0.0.0.0", PORT), MJPEGHandler)
    srv_t = threading.Thread(target=server.serve_forever, daemon=True)
    srv_t.start()

    try:
        capture_loop()
    except KeyboardInterrupt:
        print("\nInterrupted.")