import numpy as np 
import os 
import random


def siamese_pair_generator(imgs_path, num_pairs):
    """
    Arguments:
    imgs_path: directory of images 
    num_pairs: number of images pairs to create (use even number to guarentee equal positive and negatives) 

    This function takes a directory of images who are seperated into sub-directories
    according to their class and returns an array of image pairs that is specified by the parameter 'num_pairs' and another array of labels 
    (0 for not same person, 1 for same person). These arrays are made up of pairs of two images, can be matches or negatives and are shuffled. The images are of size 
    [200,240] and are preprocessed using histogram equalization, normalization, and the removal of excess noise. 
    """

    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # grabs the absolute path of this file's directory

    IMAGE_DIR = os.path.join(BASE_DIR, imgs_path) # take BASE_DIR and grabs a reference to the images folder within it

    # create x_train, y_train arrays that will later be populated 
    x_train = []
    y_train = []
   
    for i in range(num_pairs):
        dir_array = os.listdir(IMAGE_DIR)
        random_dir1 = random.choice(dir_array)
        random_file1 = random.choice(os.listdir(IMAGE_DIR + '/' + random_dir1))
        if i % 2 == 0:
            dir_array.remove(random_dir1)
            random_dir2 = random.choice(dir_array)
            random_file2 = random.choice(random_dir2)
        else:
            random_dir2 = random_dir1
            dir2_files = os.listdir(IMAGE_DIR + '/' + random_dir1)
            dir2_files.remove(random_file1)
            random_file2 = random.choice(dir2_files)

        file_array = (imgs_path + '/' + random_dir1 + '/' + random_file2, imgs_path + '/' + random_dir2 + '/' + random_file2)
        x_train.append(file_array)

        y = 1 if random_dir1 == random_dir2 else 0
        y_train.append(y)

    combined = list(zip(x_train, y_train))

    random.shuffle(combined)

    x_train, y_train = zip(*combined)


    return list(x_train), list(y_train)

