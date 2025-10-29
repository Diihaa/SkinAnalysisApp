import os
from flask import Flask, render_template

# Base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tell Flask exactly where the templates folder is
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

@app.route("/")
def index():
    # Render our WebRTC test page
    return render_template("webrtc_test.html")

if __name__ == "__main__":
    # Run Flask in debug mode so we see errors
    app.run(debug=True)
