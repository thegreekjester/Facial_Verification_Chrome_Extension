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
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.objectives import categorical_crossentropy
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
print(tf.__version__)
print(keras.__version__)
import keras_vggface 
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import matplotlib.pyplot as plt
print(keras_vggface.__version__)

def preprocess_image(IMG_PATH, IMAGE_SIZE):
    """
    This takes image and runs it through VGGFACE preprocess then preforms histogram equalization and resizing
    """

    img = load_img(IMG_PATH, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    # equ = cv2.equalizeHist(img) # histogram equalization to help deal with inconsistent lighting
    # norm = cv2.normalize(equ, None, 0, 255, cv2.NORM_MINMAX) #normalize between 0-255
    # final_img = cv2.fastNlMeansDenoising(norm) # remove all noise
    
    return img


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
 
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def verifyFace(img1, img2,vgg_face_descriptor,epsilon, omicron, graph):
    with graph.as_default():
        img1_representation = vgg_face_descriptor.predict(preprocess_image(img1, [200,240]))[0,:]
        img2_representation = vgg_face_descriptor.predict(preprocess_image(img2, [200,240]))[0,:]
        
        cosine_similarity = findCosineDistance(img1_representation, img2_representation)
        euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    
    if(cosine_similarity < epsilon and euclidean_distance < omicron):
        return 'verified'
    else:
        return 'unverified'



def predict_image(img_path, person, model, graph):
    """
    Keyword arguments:
    img_path -- path to the color image of a single person (string)
    pickle_path -- path to the labels pickle file (string)
    yml_path -- path to the yml file to use for the recognizer (string)

    Purpose:
    This function takes a color image of one person and returns the 
    facial classification value as a string
    """
    answer = 'nothing found'
    # the neural net loaded from a pre-trained caffe model for face detection
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel') # pylint: disable=no-member

    # oepn the pickle file and load them into the labels dictionary previously created
    # with open(pickle_path, 'rb') as f:
    #     labels = pickle.load(f)

    # read in image as grayscale because it is needed for facial recognition
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
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # this roi_gray is just region of the image that has been identified as a face
            #roi_gray = gray[startY:endY, startX:endX]
            img_roi = img[startY:endY, startX:endX]
            cv2.imwrite('test.png', img_roi)
            img_roi = preprocess_image('test.png', [200,240])
            cv2.imwrite('test.png', img_roi)
            # As long as the roi_gray is bigger than 0, predict the facial classification
            if img_roi.shape[0] > 0 and img_roi.shape[1] > 0:
                random_me_path = './dataset/' + person + '/' + random.choice(os.listdir('./dataset/' + person))
                answer = verifyFace(random_me_path,img_path, model, 40, 120, graph)

    return answer

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

    # Check if the directory you want to put images in exists, if not, create it
    # if not os.path.exists(BASE_DIR + '/' + rel_dir ):
    #     print('sup')
    #     os.mkdir(BASE_DIR + '/' + rel_dir)
    #     os.mkdir(BASE_DIR + '/' + rel_dir + '/' + person)
    if not os.path.exists(BASE_DIR + '/' + rel_dir + '/' + person):
        print('hey')
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
                    frame_roi = frame[startY:endY, startX:endX]
                    cv2.imwrite(rel_dir + '/' + person + '/' + str(current_time) + str(frame_num) + '.png', frame_roi)
                    print(rel_dir + '/' + person + '/' + str(frame_num) + '.png')
            
            frame_num+=1
        else:
            break
    # do a bit of cleanup
    cap.release()            