import cv2
from flask import Flask, render_template, Response, send_from_directory
import os

app = Flask(__name__)

# Camera setup
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Make sure static directory exists
SNAPSHOT_DIR = os.path.join("app", "static")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
SNAPSHOT_PATH = os.path.join(SNAPSHOT_DIR, "latest.jpg")


def save_frame():
    """Continuously capture frames and save them as latest.jpg"""
    while True:
        success, frame = camera.read()
        if success:
            cv2.imwrite(SNAPSHOT_PATH, frame)


@app.route("/")
def index():
    # HTML with auto-refreshing image
    return """
    <h1>Snapshot Camera Test</h1>
    <img src="/snapshot" width="640" height="480" id="cam">
    <script>
      function refreshImage() {
        var img = document.getElementById("cam");
        img.src = "/snapshot?rand=" + new Date().getTime();
      }
      setInterval(refreshImage, 1000); // refresh every 1 second
    </script>
    """


@app.route("/snapshot")
def snapshot():
    return send_from_directory(SNAPSHOT_DIR, "latest.jpg")


if __name__ == "__main__":
    # Start capturing frames
    import threading
    t = threading.Thread(target=save_frame, daemon=True)
    t.start()

    app.run(debug=True)
