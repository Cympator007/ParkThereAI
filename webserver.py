from flask import Flask, send_file, send_from_directory
import cv2
import numpy as np
from cv2 import imshow
import cameradetection
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route("/")
def main_page():
    return send_from_directory("static", "index.html")
@app.route("/start")
def start():
    import sched, time
    print("main")
    my_scheduler = sched.scheduler(time.time, time.sleep)
    my_scheduler.enter(3, 1, cameradetection.check, (my_scheduler,))
    my_scheduler.run()
    return("started")

@app.route("/data")
def get_data():
    f = open("data.txt")
    return(f.read())

@app.route("/img")
def imf():
    return send_file("annotated_image.png", mimetype='image/gif')

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=5000, debug=True)