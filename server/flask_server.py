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
from facial_recognition import video2dataset, predict_image
import random
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
import keras_vggface 
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input



person = 'name_user' # replace with your name
model = None # do not change
graph = None # do not change

def load_model():
    """
    Function to load the VGGFace model as well as its current graph instance 
    for creating feature vectors
    """
    global model
    model = VGGFace(include_top=False, model='resnet50', weights='vggface', input_shape=[200,240] + [3])
    model = Model(inputs=model.inputs, outputs=model.outputs)
    global graph
    graph = tf.get_default_graph() 
    print('You model has been loaded sucessfully!')

load_model()

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
        data = request.form['image']
        img = stringToImage(data[22:])
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test.png', img)
        return predict_image('test.png', person, model, graph)
        

# localhost:5000/video route
@APP.route('/video', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def new_func():
    if request.method == 'GET':
        return 'Hello, World!'
    else:
        # uses request.files becuase what I am sending over is in blob format
        # print('this is files', request.files['video'])
        file = request.files['video']
        file.save('new_output.webm')
        video2dataset('new_output.webm',5,'dataset', person)
        # send back a response saying training was succesfully 
        return 'received the video'
