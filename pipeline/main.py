"""
Arbiter sorting pipeline — Pi Camera version.

Flow: press Enter → capture frame from picamera2 → classify → print result.
Also serves a live browser preview at http://<PI_IP>:8001

Set CROP below after using preview.py to determine bounds.
CROP format: (x, y, width, height) in pixels at CAPTURE_SIZE resolution.
Set CROP = None to use full frame.

Usage:
    python pipeline/main.py --config configs/config.yaml
"""

import argparse
import json
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from libcamera import controls

import cv2
from PIL import Image
from picamera2 import Picamera2

sys.path.insert(0, "src")
from infer import WasteClassifier


# --- configure after using preview.py ---
CAPTURE_SIZE = (1280, 720)
CROP = None  # e.g. (160, 0, 720, 720) for a centered square crop
PORT = 8001

# --- shared state ---
frame_lock = threading.Lock()
latest_frame = None   # raw numpy array
latest_jpeg = None    # JPEG bytes for stream

result_lock = threading.Lock()
latest_result = None  # {"label": str, "conf": float}
classifying = False


def apply_crop(frame, crop):
    if crop is None:
        return frame
    x, y, w, h = crop
    return frame[y:y+h, x:x+w]


def _encode_frame(frame):
    """Encode a raw RGB numpy frame to JPEG bytes with result overlay."""
    display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    with result_lock:
        result = latest_result
        busy = classifying

    if busy:
        label_text, color = "Classifying...", (0, 200, 255)
    elif result:
        label_text = f"{result['label']}  {result['conf']:.1%}"
        color = (0, 255, 80)
    else:
        label_text = None

    if label_text:
        cv2.putText(display, label_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(display, label_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2, cv2.LINE_AA)

    _, jpeg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpeg.tobytes()


HTML_PAGE = b"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Arbiter Preview</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #111; color: #eee; font-family: sans-serif;
           display: flex; flex-direction: column; align-items: center;
           padding: 16px; gap: 12px; }
    h1 { font-size: 1.2rem; letter-spacing: 0.05em; color: #aaa; }
    img { max-width: 100%; border-radius: 6px; }
    button {
      padding: 10px 28px; font-size: 1rem; border: none; border-radius: 6px;
      background: #2d9e4f; color: #fff; cursor: pointer; transition: background 0.15s;
    }
    button:hover { background: #38c060; }
    button:disabled { background: #555; cursor: default; }
    #result { font-size: 1.3rem; font-weight: bold; min-height: 1.8em; color: #5ef07a; }
    #status  { font-size: 0.85rem; color: #777; }
  </style>
</head>
<body>
  <h1>Arbiter Live Preview</h1>
  <img src="/stream" alt="camera feed">
  <button id="btn" onclick="classify()">Capture &amp; Classify</button>
  <div id="result"></div>
  <div id="status"></div>
  <script>
    async function classify() {
      const btn = document.getElementById('btn');
      const res = document.getElementById('result');
      const status = document.getElementById('status');
      btn.disabled = true;
      res.textContent = '';
      status.textContent = 'Classifying...';
      try {
        const r = await fetch('/classify', { method: 'POST' });
        const data = await r.json();
        if (data.error) { status.textContent = 'Error: ' + data.error; }
        else { res.textContent = data.label + '  ' + (data.conf * 100).toFixed(1) + '%'; status.textContent = ''; }
      } catch (e) { status.textContent = 'Request failed.'; }
      finally { btn.disabled = false; }
    }
  </script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, classifier, *args, **kwargs):
        self.classifier = classifier
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        jpeg = latest_jpeg
                    if jpeg:
                        self.wfile.write(b"--frame\r\n"
                                         b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global classifying, latest_result
        if self.path == "/classify":
            with frame_lock:
                frame = latest_frame
            if frame is None:
                body = json.dumps({"error": "no frame yet"}).encode()
            else:
                with result_lock:
                    classifying = True
                    latest_result = None
                try:
                    cropped = apply_crop(frame, CROP)
                    image = Image.fromarray(cropped)
                    label, conf = self.classifier.predict(image)
                    print(f"  → {label} ({conf:.2%})")
                    with result_lock:
                        latest_result = {"label": label, "conf": conf}
                    body = json.dumps({"label": label, "conf": conf}).encode()
                finally:
                    with result_lock:
                        classifying = False
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def run(config_path: str):
    global latest_frame, latest_jpeg

    classifier = WasteClassifier(config_path)

    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": CAPTURE_SIZE, "format": "RGB888"}
    )
    cam.configure(config)
    cam.start()
    cam.set_controls({
        "AwbMode": controls.AwbModeEnum.Fluorescent,
        "AfMode": controls.AfModeEnum.Continuous,
    })
    time.sleep(2)

    # Camera loop — runs in background, keeps latest_frame/jpeg fresh
    def camera_loop():
        global latest_frame, latest_jpeg
        while True:
            frame = cam.capture_array()
            jpeg = _encode_frame(frame)
            with frame_lock:
                latest_frame = frame
                latest_jpeg = jpeg

    threading.Thread(target=camera_loop, daemon=True).start()

    # HTTP server — pass classifier into handler via closure
    def handler_factory(*args, **kwargs):
        return _Handler(classifier, *args, **kwargs)

    server = HTTPServer(("0.0.0.0", PORT), handler_factory)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    crop_info = f"crop={CROP}" if CROP else "full frame (no crop)"
    print(f"Arbiter ready. Resolution: {CAPTURE_SIZE}, {crop_info}")
    print(f"Browser preview at http://<PI_IP>:{PORT}")
    print("Press Enter to capture and classify.\n")

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
                with frame_lock:
                    frame = latest_frame
                if frame is not None:
                    cropped = apply_crop(frame, CROP)
                    image = Image.fromarray(cropped)
                    label, conf = classifier.predict(image)
                    print(f"  → {label} ({conf:.2%})")
                    with result_lock:
                        global latest_result
                        latest_result = {"label": label, "conf": conf}
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
