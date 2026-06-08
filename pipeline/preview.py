"""
Live camera preview over HTTP.
Open http://<PI_IP>:8000 in your browser to see the feed.

Usage:
    python pipeline/preview.py
"""

from importlib.metadata import metadata
import io
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from picamera2 import Picamera2

STREAM_SIZE = (1280, 720)
PORT = 8000

# --- shared frame state ---
frame_lock = threading.Lock()
latest_frame = None


def camera_thread():
    global latest_frame
    import cv2

    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": STREAM_SIZE, "format": "RGB888"}
    )
    cam.configure(config)
    # cam.set_controls({"AwbMode": 4})  # 4 = fluorescent, close to most white LEDs
    cam.start()
    time.sleep(6)
    metadata = cam.capture_metadata()
    print("ColourGains:", metadata["ColourGains"])  # (red_gain, blue_gain)
    cam.set_controls({
        "AwbEnable": False,
        # "ColourGains": (2.8, 0.3)
         "ColourGains": (2.5, 1.7)
    })

    while True:
        frame = cam.capture_array()
        _, jpeg = cv2.imencode(
            ".jpg",
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )
        with frame_lock:
            latest_frame = jpeg.tobytes()


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
                        frame = latest_frame
                    if frame:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.05)  # ~20fps
            except (BrokenPipeError, ConnectionResetError):
                pass


if __name__ == "__main__":
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()

    print(f"Preview streaming at http://<PI_IP>:{PORT}")
    print("Ctrl+C to stop.\n")

    server = HTTPServer(("0.0.0.0", PORT), MJPEGHandler)
    server.serve_forever()