import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.objectives import categorical_crossentropy
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import numpy as np
print(tf.__version__)
print(keras.__version__)
import keras_vggface 
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import matplotlib.pyplot as plt
print(keras_vggface.__version__)

IMAGE_SIZE = [200,240]

IMG_PATH_1 = './dataset/s1/4.pgm'
IMG_PATH_2 = './dataset/s1/6.pgm'


img_1 = load_img(IMG_PATH_1, target_size=IMAGE_SIZE)
img_1 = img_to_array(img_1)
img_1 = img_1.reshape((1, img_1.shape[0], img_1.shape[1], img_1.shape[2]))
img_1 = preprocess_input(img_1)

img_2 = load_img(IMG_PATH_2, target_size=IMAGE_SIZE)
img_2 = img_to_array(img_2)
img_2 = img_2.reshape((1, img_2.shape[0], img_2.shape[1], img_2.shape[2]))
img_2 = preprocess_input(img_2)

model = VGGFace(include_top=False, model='resnet50', weights='vggface', input_shape=IMAGE_SIZE + [3])

model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

# At this point, the model outputs a vector of shape 1x2048

features_1 = model.predict(img_1)
features_2 = model.predict(img_2)

diff = np.abs(features_1 - features_2)

print(diff)
print(diff.shape)
print(np.sum(diff))

if(np.sum(diff) < 2500):
    print('match')
else:
    print('unauthorized')