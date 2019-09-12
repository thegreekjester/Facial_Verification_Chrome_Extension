import numpy as np 
import cv2 
import random
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import keras_vggface 
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


def preprocess_image(IMG_PATH):
    """
    Arguments:
    IMG_PATH - relative path to image to be proprocessed

    This function takes image and runs it through VGGFACE preprocess then 
    preforms histogram equalization and resizing
    """

    img = load_img(IMG_PATH, target_size=(200, 240))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img
    
 
def findEuclideanDistance(image_1, image_2):
    """
    Arguments:
    image_1: the feature vector of a stored image of user
    image_2: the feature vector of the image taken by whoever is trying to verify
    """
    euclidean_distance = image_1 - image_2
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)

    return euclidean_distance

def verifyFace(image_1, image_2,model,epsilon, graph):
    """
    Arguments:
    image_1: the feature vector of a stored image of user
    image_2: the feature vector of the image taken by whoever is trying to verify
    model: keras model (siamese half)
    graph: current instance of model being used
    """
    with graph.as_default():
        img1_representation = model.predict(preprocess_image(image_1))
        img2_representation = model.predict(preprocess_image(image_2))
        
        euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)

    if(euclidean_distance < epsilon):
        return 'verified'
    else:
        return 'unverified'



def predict_image(img_path, person, model, graph):
    """
    Arguments:
    img_path -- path to the color image of a single person (string)
    person - person we are verifying (string)
    model - base neural network used within the siamese network architecture (keras model)
    graph - current instance of the model (tf backend graph)

    Purpose:
    This function takes a color image of one person and returns the 
    facial verification response using eucledian distance of siamese network generated image vectors
    """
    answer = 'nothing found'
    # the neural net loaded from a pre-trained caffe model for face detection
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel') # pylint: disable=no-member

    # read in image as grayscale because it is needed for facial verification
    img = cv2.imread(img_path)

    # grab height and width from the image shape (h, w, color_channels)
    [h,w] = img.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), # pylint: disable=no-member
        1.0, (300, 300), (104.0, 177.0, 123.0)) 
    net.setInput(blob)
    detections = net.forward()
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
           
            img_roi = img[startY:endY, startX:endX] # the face cropped out of image
            cv2.imwrite('test.png', img_roi)
            # As long as the img_roi is bigger than 0, predict the facial verification
            if img_roi.shape[0] > 0 and img_roi.shape[1] > 0:

                    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Grabbing base directory

                    if os.path.exists(BASE_DIR + '/' + 'dataset'):
                        random_me_path = './dataset/' + person + '/' + random.choice(os.listdir('./dataset/' + person)) # pick random image of myself
                        answer = verifyFace(random_me_path,img_path, model, 120, graph) # change the epsilon to adjust verication performance

    return person if answer == 'verified' else 'nothing found'

def video2dataset(vid_path, frame_skip, rel_dir, person, prototxt='deploy.prototxt.txt', caffe_model='res10_300x300_ssd_iter_140000.caffemodel'):
    """
    Arguments:
    vid_path -- path to the video (string)
    frame_skip -- number of frames to skip before capturing for training data (integer)
    rel_dir -- relative path to the folder you want to populate with training images (string)
    person -- name of the person these photos are of (string)

    Purpose: 
    Takes a video file of one person and creates an instant image dataset full of
    only facial ROI images

    """
    current_time = round(time.time())
    # Reading in the Caffe Model for Face Detection
    net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model) # pylint: disable=no-member

    # Grab the current directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Check if the directory you want to put images in exists, if not, create it
    if not os.path.exists(BASE_DIR + '/' + rel_dir + '/' + person):
        print('hey')
        os.makedirs(BASE_DIR + '/' + rel_dir + '/' + person)

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
        
                if frame_num % frame_skip == 0: #grab every frame skip (number) frame
                    frame_roi = frame[startY:endY, startX:endX]
                    cv2.imwrite(rel_dir + '/' + person + '/' + str(current_time) + str(frame_num) + '.png', frame_roi) # write to dataset folder
                    print(rel_dir + '/' + person + '/' + str(frame_num) + '.png')
            
            frame_num+=1
        else:
            break
    # do a bit of cleanup
    cap.release()            