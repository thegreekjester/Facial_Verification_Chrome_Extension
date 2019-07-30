import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.objectives import categorical_crossentropy
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD, Adam
import numpy as np
import cv2 
import pickle
from sklearn.preprocessing import LabelEncoder
import time
print(tf.__version__)
print(keras.__version__)
import os
import keras_vggface 
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import matplotlib.pyplot as plt
print(keras_vggface.__version__)

os.environ['KMP_DUPLICATE_LIB_OK']='True'



def train_faces():
    """
    Purpose: 
    Takes the dataset created by the video2dataset function and trains a VGGFACE model on those images

    """

    TRAIN_DIR = "FACES"
    BATCH_SIZE = 8
    IMAGE_SIZE = [224,224]
    epochs = 10

    data_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=0.2,
            preprocessing_function=preprocess_input)

    train_gen = data_gen.flow_from_directory(TRAIN_DIR, target_size=IMAGE_SIZE, shuffle=True, batch_size=BATCH_SIZE, subset='training')

    validation_gen = data_gen.flow_from_directory(TRAIN_DIR, target_size=IMAGE_SIZE, shuffle=True, batch_size=BATCH_SIZE, subset='validation')


    model = VGGFace(include_top=False, model='resnet50', weights='vggface', input_shape=IMAGE_SIZE + [3])

    for layer in model.layers:
        layer.trainable = False

    x = Flatten()(model.output)

    prediction = Dense(2, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=prediction)

    model.summary()

    model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
    )

    model.fit_generator(train_gen, validation_data = validation_gen, steps_per_epoch=train_gen.samples // BATCH_SIZE, validation_steps= validation_gen.samples // BATCH_SIZE, epochs = epochs)

    model.save('face_recognition.h5')



def video2dataset(vid_path, frame_skip, rel_dir, person, prototxt='deploy.prototxt.txt', caffe_model='res10_300x300_ssd_iter_140000.caffemodel'):
    """
    Keyword arguments:
    vid_path -- path to the video (string)
    frame_skip -- number of frames to skip before capturing for training data (integer)
    rel_dir -- relative path to the folder you want to populate with training images (string)
    person -- name of the person these photos are of (string)

    Purpose: 
    Takes a video file of one person and creates an instant image dataset full of
    only facial ROI images (in grayscale)

    """
    current_time = round(time.time())
    # Reading in the Caffe Model for Face Detection
    net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model) # pylint: disable=no-member

    # Grab the current directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print('*****this is base directory*****', BASE_DIR)

    # Check if the directory you want to put images in exists, if not, create it
    if not os.path.exists(BASE_DIR + '/' + rel_dir + '/' + person):
        # os.mkdir(BASE_DIR + '/' + rel_dir)
        os.mkdir(BASE_DIR + '/' + rel_dir + '/' + person)

    frame_num = 0
    # Set capture to the video file
    cap = cv2.VideoCapture(vid_path)

    while(cap.isOpened()):
        # grab return and frame from cap.read()
        ret, frame = cap.read()

        # Extra precaution, if video still has frames
        if ret:
            # frame is (h,w,color_channels)
            [h, w] = frame.shape[:2]
            
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), # pylint: disable=no-member
            1.0, (300, 300), (104.0, 177.0, 123.0)) 
            net.setInput(blob)
            detections = net.forward()
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]
        
                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence < .5:
                    continue
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
        
                if frame_num % frame_skip == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_roi = gray[startY+10:endY + 10, startX+10:endX+10]
                    cv2.imwrite(rel_dir + '/' + person + '/' + str(current_time) + str(frame_num) + '.png', gray_roi)
                    print(rel_dir + '/' + person + '/' + str(frame_num) + '.png')
            
            frame_num+=1
        else:
            break
    # do a bit of cleanup
    cap.release()            


def predict_image(img_path, user):
    """
    Keyword arguments:
    img_path -- path to the color image of a single person (string)
    
    Purpose:
    This function takes a color image of one person and returns the 
    facial classification value as a string
    """
    
    model = load_model('face_recognition.h5')

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img, dtype='float')
    print(img.shape)

    img = preprocess_input(img) # normalize/process data to how the model was trained
    print(img.shape)

    img = img.reshape(-1, 224, 224, 3) # Needed because the model expects a 4-d tensor
    prediction = model.predict(img)
    max_val = np.argmax(prediction)
    print(prediction)
    keras.backend.clear_session()
    if(max_val == 1):
        return user
    else:
        return 'not_recognized'
