import cv2
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt
from constants import IMG_HEIGHT, IMG_WIDTH
import sys
from PIL import Image
import os
import random
import keras
from keras.utils import to_categorical
import h5py
import json

train_dir = './train_data/'
test_dir = './test_data/'

def get_end_slash(f):
    return f[f.rindex("/",0,f.rindex("/"))+1:f.rindex("/")]

def get_pic_name(f):
    return f[f.rindex("/")+1: ]

def get_leading_directory(f):
    return f[:f.rindex("/")+1]
            
def scale_X(X):
    return X/255.0

def display_data(X,y):
    plt.figure(figsize=[5,5])

    # Display the first image in data
    plt.subplot(121)
    plt.imshow(X[0,:,:])
    plt.title("Class : {}".format(y[0]))

    # Display the second image
    plt.subplot(122)
    plt.imshow(X[1,:,:])
    plt.title("Class : {}".format(y[1]))
    plt.colorbar(orientation="horizontal")
    # plt.show()


def load_dataset_from_file(fname):
    dataset = np.load(fname)
    return (dataset[0], dataset[1], dataset[2], dataset[3])

def filter_new_images(pth, subdirs):
    folders = glob(pth+"*/")
    classes = [get_end_slash(f) for f in folders]
    for i, folder in enumerate(folders):
        curr_class = classes[i]
        if(curr_class not in subdirs):
            continue
        #convert pngs and other types to jpgs
        images = []
        im_types = ['png','jpeg', 'gif']
        for im_type in im_types:
            images = images + glob(folder+"*."+im_type)
        for image in images:
            print(image)
            im = Image.open(image)
            jpg = im.convert('RGB')
            pname = image[:image.rindex(".")]+'.jpg'
            jpg.save(pname)
            os.remove(image)
            print(pname)

        #make sure jpgs can be opened properly
        images_jpg = glob(folder+"*.jpg")
        for image in images_jpg:
            try:
                img = cv2.imread(image)
                img.shape
            except:
                os.remove(image)
                print(curr_class,get_pic_name(image), "failed to load and was deleted")
        
        #rename all jpgs to classname + number
        images_jpg = glob(folder+"*.jpg")
        for j, image in enumerate(images_jpg):
            new_name = get_leading_directory(image) +curr_class+str(j)+".jpg"
            os.rename(image, new_name )
            print(new_name)
            
def split_train_test(pth,subdirs, proportion):
    folders = glob(pth+"*/")
    classes = [get_end_slash(f) for f in folders]
    for i, folder in enumerate(folders):
        curr_class = classes[i]
        if(curr_class not in subdirs and subdirs!="all"):
            continue
        new_dir = "test_data/" + curr_class + "/"
        os.makedirs(os.path.dirname(new_dir), exist_ok=True)
        images = glob(folder+"*.jpg")
        rand_samp = random.sample(images,int(proportion*len(images)))
        print(curr_class, len(rand_samp), len(images))
        for image in rand_samp:
            new_im_loc = new_dir + get_pic_name(image)
            os.rename(image, new_im_loc)
        print("new_train_len",len(glob(folder+"*.jpg")))

def to_h5py(pth):
    folders = glob(pth+"*/")
    classes = [get_end_slash(f) for f in folders]
    input_fname = 'processed_datasets/'+get_end_slash(pth) + str(IMG_WIDTH) + "x" + str(IMG_HEIGHT)+  '.h5'
    if(os.path.isfile(input_fname)):
        inp = input('overwrite file ' + input_fname + '?, y/n: ')
        if inp.lower() =="y":
            print("file will be overwritten")
            os.remove(input_fname)
        elif inp.lower()=="n":
            input_fname = input("enter a new filename: ") + '.h5'
            print(input_fname)
        else:
            print("incorrect input, preprocessing failed")
            return     

    #open hf
    hf = h5py.File(input_fname)
    all_images = glob(pth+"**/*.jpg",recursive=True)
    n_images = len(all_images)
  
    #create dataset X and label list
    X = hf.create_dataset(
        name= 'X',
        shape=(n_images,IMG_WIDTH, IMG_HEIGHT, 3),
        maxshape=(None, IMG_WIDTH, IMG_HEIGHT,None),
        compression="gzip",
        compression_opts=9)
    label_lis = []
    x_ind =0
    for i, folder in enumerate(folders):
        images = glob(folder+"*.jpg")
        total_images = len(images)
        print(classes[i],total_images)
        for j, image_pth in enumerate(images):
            img = cv2.imread(image_pth)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = scale_X(img)
            X[x_ind] = img
            label_lis.append(i)
            print("{}/{} fname = {}".format(j,total_images, get_pic_name(image_pth)))
            x_ind+=1
    hf.create_dataset(
        name= 'y',
        compression="gzip",
        compression_opts=9,
        data=label_lis)
    
    y_one_hot = to_categorical(np.array(label_lis))
    hf.create_dataset(
        name= 'y_one_hot',
        compression="gzip",
        compression_opts=9,
        data=y_one_hot)

    hf.close()


def load_dataset(fname):
    hf = h5py.File(fname, 'r')
    print(hf.keys())
    X = hf['X'][()]
    y = hf['y'][()]
    y_one_hot = hf['y_one_hot'][()]
    classes = get_class_list()
    hf.close()
    return (X, y, y_one_hot, classes)

def load_history(fname):
    history = {}
    with open(fname, 'r') as f:
        history = json.load(f)
    return history

def get_class_list():
    folders = glob(train_dir+"*/")
    classes = [get_end_slash(f) for f in folders]
    return classes

def process_single_img(image_pth, width, height):
    img = cv2.imread(image_pth)
    img = cv2.resize(img, (width, height))
    img = scale_X(img)
    return img

print(get_class_list())
# to_h5py(test_dir)
# to_h5py(train_dir)
# load_dataset("test_data200x200.h5")
# load_dataset()
# (X, y, y_oh, classes) = load_dataset("test_data200x200.h5")
# print(X.shape, y.shape, y_oh.shape)
# filter_images(image_dir, ['fruit'])
# process_images_to_file(train_dir)
# process_images_to_file(test_dir)
# (X,y,classes) = load_dataset_from_file("train_data.npy")
# print(X.shape,y.shape)
# display_data(X,y)
# split_train_test(train_dir, "nom", 0.3)
# print(dat[0].shape)

# plt.show()


