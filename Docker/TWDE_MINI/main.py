from flask import Flask, render_template
import cv2, torch, argparse, base64
import numpy as np
from generator import Generator

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/image")
def generate():
    global gen, generating, num_generated
    # Generate and transfer image to base64 string
    with torch.no_grad():       
        img = (np.array(gen(torch.randn(1,512,1,1))[0].permute(1,2,0)) * 0.5 + 0.5)[...,::-1] * 255

    _, img = cv2.imencode('.jpg', img)
    img = base64.b64encode(img)
    img = img.decode()
    return img
       
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--ip',type=str,  default="0.0.0.0", help='ip address of the device')
    ap.add_argument('-o','--port',type=int,  default=8000 ,help='ephemeral port number of the server (1024-65535)')
    ap.add_argument('-n', '--num', type=int, default=1, help='Numbers of generated images in each refresh')

    args = vars(ap.parse_args())

    num_imgs = args['num']

    output = None
    generating = True
    num_generated = 4
    html_block = ''

    gen = Generator(latent_vector=512, factors=[1, 1, 1, 2, 2, 2]).to("cpu")
    checkpoint = torch.load("weight/130_g.pth.tar", map_location=torch.device('cpu'))

    gen.load_state_dict(checkpoint["state_dict"])
    gen.eval()

    app.run(host=args['ip'], port=args['port'], debug=True, threaded=True, use_reloader=False)


