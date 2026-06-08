"""
Live camera preview over HTTP.
Open http://<PI_IP>:8000 in your browser to see the feed.

Usage:
    python pipeline/preview.py
"""

import io
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
from picamera2 import Picamera2
from libcamera import controls

STREAM_SIZE = (1280, 720)
PORT = 8000

# --- shared frame state ---
frame_lock = threading.Lock()
latest_frame = None


def camera_thread():
    global latest_frame

    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": STREAM_SIZE, "format": "RGB888"}
    )
    cam.configure(config)
    
    # Start camera before setting runtime controls
    cam.start()
    
    # 1. Enable AWB and set it to Fluorescent to handle the room lighting
    # 2. Set continuous autofocus
    cam.set_controls({
        "AwbEnable": True,
        "AwbMode": controls.AwbModeEnum.Fluorescent,
        "AfMode": controls.AfModeEnum.Continuous
    })

    # Give the ISP 2 seconds to settle AWB/Exposure before streaming starts
    time.sleep(2)

    while True:
        frame = cam.capture_array()
        
        # Convert RGB to BGR and encode to JPEG
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
