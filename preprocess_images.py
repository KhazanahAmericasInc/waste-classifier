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

train_dir = './train_data/'
test_dir = './test_data/'

def get_end_slash(f):
    return f[f.rindex("/",0,f.rindex("/"))+1:f.rindex("/")]

def get_pic_name(f):
    return f[f.rindex("/")+1: ]

def get_leading_directory(f):
    return f[:f.rindex("/")+1]

def resize_and_label(pth):
    # input_lis = []
    label_lis = []
    folders = glob(pth+"*/")
    classes = [get_end_slash(f) for f in folders]
    input_fname = get_end_slash(pth) + 'temp' + '.npy'
    # create_data_batch([], input_fname)
    for i, folder in enumerate(folders):
        images = glob(folder+"*.jpg")
        total_images = len(images)
        print(classes[i],total_images)
        for j, image_pth in enumerate(images):
            img = cv2.imread(image_pth)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            # input_lis.append((np.array(img)))
            if i==0 and j==0:
                create_data_batch([np.array(img)],input_fname)
            else:
                append_data_batch([np.array(img)], input_fname)
            label_lis.append(i)
            print("{}/{} fname = {}".format(j,total_images, get_pic_name(image_pth)))
            
            
    input_lis = load_batch(input_fname)
    os.remove(input_fname)        
    return (np.array(input_lis), 
        np.array(label_lis), np.array(classes))
            
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
    plt.show()

def process_images_to_file(imdir):
    (X, y,classes) = resize_and_label(imdir)
    X = scale_X(X)
    X = X.reshape(-1, IMG_WIDTH,IMG_HEIGHT, 3)
    one_hot_y = to_categorical(y)
    data = np.array([X, y, one_hot_y, classes])
    fname = get_end_slash(imdir) + str(IMG_WIDTH) + "x" + str(IMG_HEIGHT)+ '.npy'
    np.save(fname, data)
    print("saved to ", fname)

def load_dataset_from_file(fname):
    dataset = np.load(fname)
    return (dataset[0], dataset[1], dataset[2], dataset[3])

def append_data_batch(data, fname):
    prev_data = np.load(fname)
    data = np.array(data)
    updated_data = np.concatenate((prev_data, np.array(data)))
    np.save(fname, updated_data)
    

def create_data_batch(data, fname):
    np.save(fname,np.array(data))

def load_batch(fname):
    batch = np.load(fname)
    return batch

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
    input_fname = get_end_slash(pth) + str(IMG_WIDTH) + "x" + str(IMG_HEIGHT)+  '.h5'
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

    for i, folder in enumerate(folders):
        images = glob(folder+"*.jpg")
        total_images = len(images)
        print(classes[i],total_images)
        for j, image_pth in enumerate(images):
            img = cv2.imread(image_pth)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            X[i] = img
            label_lis.append(i)
            print("{}/{} fname = {}".format(j,total_images, get_pic_name(image_pth)))
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

# def process_images(imdir):
#     (X, y,classes) = resize_and_label(imdir)
#     X = scale_X(X)
#     X = X.reshape(-1, IMG_WIDTH,IMG_HEIGHT, 3)
#     one_hot_y = to_categorical(y)
#     data = np.array([X, y, one_hot_y, classes])
#     fname = get_end_slash(imdir) + str(IMG_WIDTH) + "x" + str(IMG_HEIGHT)+ '.npy'
#     np.save(fname, data)
#     print("saved to ", fname)


def load_dataset(fname):
    hf = h5py.File(fname, 'r')
    print(hf.keys())
    hf.close()

def get_class_list():
    folders = glob(train_dir+"*/")
    classes = [get_end_slash(f) for f in folders]
    return classes

to_h5py(test_dir)
# load_dataset("test_data200x200.h5")
# load_dataset()

# filter_images(image_dir, ['fruit'])
# process_images_to_file(train_dir)
# process_images_to_file(test_dir)
# (X,y,classes) = load_dataset_from_file("train_data.npy")
# print(X.shape,y.shape)
# display_data(X,y)
# split_train_test(train_dir, "nom", 0.3)
# print(dat[0].shape)


