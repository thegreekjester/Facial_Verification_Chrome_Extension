import numpy as np 
import cv2 
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import time

def preprocess_input(img_path):

    img = cv2.imread(img_path,0) # load BGR color image as gray
    img = cv2.resize(img, (92,112)) # un-needed for LBPH algorithm but its habit

    equ = cv2.equalizeHist(img) # histogram equalization
    norm = cv2.normalize(equ, None, 0, 255, cv2.NORM_MINMAX) #normalize between 0-255
    final_img = cv2.fastNlMeansDenoising(norm) # remove all noise
    # res = np.hstack((equ, de_noise)) #stacking images side-by-side
    # plt.imshow(res, cmap='gray') # display equalized image 
    # plt.show()
    
    return final_img



def predict_image(img_path, pickle_path, yml_path):
    """
    Keyword arguments:
    img_path -- path to the color image of a single person (string)
    pickle_path -- path to the labels pickle file (string)
    yml_path -- path to the yml file to use for the recognizer (string)

    Purpose:
    This function takes a color image of one person and returns the 
    facial classification value as a string
    """

    labels = {}
    # the neural net loaded from a pre-trained caffe model for face detection
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel') # pylint: disable=no-member

    # oepn the pickle file and load them into the labels dictionary previously created
    with open(pickle_path, 'rb') as f:
        labels = pickle.load(f)

    # read in image as grayscale because it is needed for facial recognition
    img = cv2.imread(img_path)

    # grab height and width from the image shape (h, w, color_channels)
    [h,w] = img.shape[:2]

    # This loads the recognizer with info (trainer.yml) about the training
    recognizer = cv2.face.LBPHFaceRecognizer_create() # pylint: disable=no-member
    recognizer.read(yml_path)

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
            img_roi = preprocess_input('test.png')
            cv2.imwrite('test.png', img_roi)
            # As long as the roi_gray is bigger than 0, predict the facial classification
            if img_roi.shape[0] > 0 and img_roi.shape[1] > 0:
                id_, uncertainty = recognizer.predict(img_roi)
                print(uncertainty)
                print(id_)
            # if the uncertainty of the classification is less than 60, print the name value
            if uncertainty <85:
                name = labels[id_]
                print(name)
                return name
    print('nothing found')
    return 'nothing found'


def train_faces(dataset, pickle_path, yml_file):
    """
    Keyword arguments:
    dataset -- path to the folder of face roi images (string)
    pickle_path -- path to the labels pickle file (string)
    yml_path -- path to save the yml file (string)

    Purpose: 
    Takes in folder of face roi images (not full person images) and creates a pickle file of labels 
    and a yml file that reflects the training done

    """
    # Frontal Face Recognizer that comes in opencv standard
    recognizer = cv2.face.LBPHFaceRecognizer_create() # pylint: disable=no-member

    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # grabs the absolute path of this file's directory

    IMAGE_DIR = os.path.join(BASE_DIR, dataset) # take BASE_DIR and grabs a reference to the images folder within it

    # create x_train, y_train arrays that will later be populated to train recognizer
    x_train = []
    y_train = []
    # Looping through the files in the image_dir
    for root, dirs, files in os.walk(IMAGE_DIR):
        for i,file in enumerate(files): #for each file
            if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg') or file.endswith('JPG') or file.endswith('pgm'): #if it ends with .png or .jpg
                path = os.path.join(root, file) #print its path
                label = os.path.basename(root).replace(' ', '_').lower() # returns the directory that the image is in (to be used as a label)

                y_train.append(label) # put the label (dir name) into y train
                # read the image as a grayscale because the recognizer expects it
                img = preprocess_input(path)
                x_train.append(img)

                
    # Label encoder code used to create integer labels per person classifying
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_train_labels = label_encoder.inverse_transform(y_train)

    label_ids = dict(zip(y_train, y_train_labels))

    # save the labels to a pickle file to be used later
    with open(pickle_path, 'wb') as f:
        pickle.dump(label_ids, f)

    # train the recognizer object with the data and labels
    recognizer.train(x_train, y_train)
    recognizer.save(yml_file)
    print('all done!')


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
                    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # gray_roi = gray[startY:endY, startX:endX]
                    frame_roi = frame[startY:endY, startX:endX]
                    cv2.imwrite(rel_dir + '/' + person + '/' + str(current_time) + str(frame_num) + '.png', frame_roi)
                    print(rel_dir + '/' + person + '/' + str(frame_num) + '.png')
            
            frame_num+=1
        else:
            break
    # do a bit of cleanup
    cap.release()            