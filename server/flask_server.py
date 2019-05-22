# FLASK_APP=flask_server.py
# flask run

import base64
import io
import cv2
import numpy as np
from PIL import Image

from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
# from werkzeug.datastructures import ImmutableMultiDict


def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))


APP = Flask(__name__)
cors = CORS(APP, support_credentials=True)
APP.config['CORS_HEADERS'] = 'Content-Type'


@APP.route('/', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def route_func():
    if request.method == 'GET':
        return 'Hello, World!'
    else:
        data = request.form['image']
        img = stringToImage(data[22:])
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test.png', img)
        return 'received it'
