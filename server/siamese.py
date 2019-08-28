import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.objectives import categorical_crossentropy
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import numpy as np
print(tf.__version__)
print(keras.__version__)
import os
import keras_vggface 
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import matplotlib.pyplot as plt
print(keras_vggface.__version__)

IMAGE_SIZE = [200,240]

model = VGGFace(include_top=False, model='resnet50', weights='vggface', input_shape=IMAGE_SIZE + [3])

model.summary()