from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse

import cv2
import numpy as np
import torch
import config
from model import Generator, Discriminator
from torchvision.utils import make_grid

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def Draw():
    global outputframe, lock#, num_images, step, alpha
    while(True):
        with torch.no_grad():
            images = gen(torch.randn((num_images, 512, 1, 1)), step, alpha)
        with lock:
            outputframe = np.array(make_grid(images, nrow=2).permute(1,2,0)*0.5 + 0.5) * 255
            images = None

def generate():
    global outputFrame, lock
    while(True):
        with lock:
            #Checking if output successful
            if outputframe is None:
                continue
            #encode in jpg
            (flag, encodedImg) = cv2.imencode(".jpg", outputframe[:,:,::-1])
            if not flag:
                continue

        #generate byte
        yield(b'--frame\r\n' b'Content_Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--ip',type=str, required=True,
                    help='ip address of the device')
    ap.add_argument('-o','--port',type=int, required=True,
                    help='ephemeral port number of the server (1024-65535)')
    ap.add_argument('-s', '--step', type=int, default=5,
                    help='Output step of generator')
    ap.add_argument('-a', '--alpha', type=int, default=1,
                    help='Alpha value of fade in layer in generator')
    ap.add_argument('-n', '--num', type=int, default=6,
                    help='Numbers of generated images in each refresh')
    args = vars(ap.parse_args())

    # Initial the outputframe
    outputframe = None
    lock = threading.Lock()

    step = args['step']
    alpha = args['alpha']
    num_images = args['num']

    gen = Generator(latent_vector=512, factors=config.FACTORS[:step+1]).to("cpu")
    checkpoint = torch.load("step5_4.pth.tar", map_location=torch.device('cpu'))
    gen.load_state_dict(checkpoint["state_dict"])
    gen.eval()

    t = threading.Thread(target=Draw)
    t.daemon = True
    t.start()

    app.run(host=args['ip'], port=args['port'], debug=True, threaded=True, use_reloader=False)

