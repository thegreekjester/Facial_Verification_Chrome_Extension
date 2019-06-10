# export FLASK_APP=flask_server.py
# flask run

import base64
import io
import cv2
import numpy as np
from PIL import Image
import os

from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
from facial_recognition import video2dataset, train_faces, predict_image



def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))


APP = Flask(__name__)
cors = CORS(APP, support_credentials=True)
APP.config['CORS_HEADERS'] = 'Content-Type'


# localhost:5000/ route
@APP.route('/', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def route_func():
    if request.method == 'GET':
        return 'Hello, World!'
    else:
        print('sup')
        data = request.form['image']
        img = stringToImage(data[22:])
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test.png', img)
        return predict_image('test.png', 'labels.pickle', 'new_yml.yml')
        

# localhost:5000/video route
@APP.route('/video', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def new_func():
    if request.method == 'GET':
        return 'Hello, World!'
    else:
        # uses request.files becuase what I am sending over is in blob format
        print('this is files', request.files['video'])
        file = request.files['video']
        file.save('new_output.webm')
        person = 'Peter_Katsos'
        video2dataset('new_output.webm',5,'dataset', person)
        train_faces('dataset', 'new_pickle.pickle', 'new_yml.yml')
        # send back a response saying training was succesfully 
        return 'received the video'
