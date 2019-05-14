from flask import Flask
from flask import request
import cv2

APP = Flask(__name__)

@APP.route('/', methods=['GET', 'POST'])
def route_func():
    if request.method == 'GET':
        return 'Hello, World!'
    else:
        return 'SUPP'
