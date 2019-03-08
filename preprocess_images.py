import cv2
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt
from constants import *
import sys
from PIL import Image
import os
import random
import keras
from keras.utils import to_categorical
import h5py
import json

#declare the directories that hold the training and test data sets
train_dir = './clumped_kitchen_train_data/'
test_dir = './clumped_kitchen_test_data/'

#strip the folder name from a path
def get_end_slash(f):
    return f[f.rindex("/",0,f.rindex("/"))+1:f.rindex("/")]

#strip the picture name from a path
def get_pic_name(f):
    return f[f.rindex("/")+1: ]

#get the parent directory
def get_leading_directory(f):
    return f[:f.rindex("/")+1]

#normalize the inputs   
def scale_X(X):
    return X/255.0

'''
display a couple images in the dataset to check if it turned out as expected
Input: inputs and labels 
Output: plots to a matplotlib plot, type plt.show() or uncomment last line of method
        to show images
'''
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

'''
load the dataset from an h5py file given filename
input: filename
output: (X, y, y_one_hot, classes)
'''
def load_dataset_from_file(fname):
    dataset = np.load(fname)
    return (dataset[0], dataset[1], dataset[2], dataset[3])

#filter new images added, converting non jpg images, and naming them to a generic enumerated name
#to jpg and removing the images that cannot be opened by opencv
#must be given the path and the sudirectories desired to filter
def filter_new_images(pth, subdirs):
    folders = glob(pth+"*/")
    classes = [get_end_slash(f) for f in folders]
    print(classes)
    for i, folder in enumerate(folders):
        curr_class = classes[i]
        if(curr_class not in subdirs and subdirs!="all"):
            # print(classes)
            # print("potato")
            continue
        #convert pngs and other types to jpgs
        images = []
        im_types = ['png','jpeg', 'gif', 'JPG', 'svg', 'ashx', 'Jpg']
        for im_type in im_types:
            images = images + glob(folder+"*."+im_type)
        for image in images:
            try:
                im = Image.open(image)
                jpg = im.convert('RGB')
                pname = image[:image.rindex(".")]+'.jpg'
                if pname!= image:
                    os.remove(image)
                    jpg.save(pname)
                    print(pname)
            except:
                print("could not open image", image)
                os.remove(image)


        #make sure jpgs can be opened properly
        images_jpg = glob(folder+"*.jpg")
        for image in images_jpg:
            try:
                img = cv2.imread(image)
                img.shape
            except:
                os.remove(image)
                print(curr_class,get_pic_name(image), "failed to load in opencv and was deleted")
        
        #rename all jpgs to classname + number
        images_jpg = glob(folder+"*.jpg")
        #store in temporary directory in case that name exists in present directory
        temp_dir = folder+"temp/"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        for j, image in enumerate(images_jpg):
            new_name = get_leading_directory(image) +"temp/"+curr_class+str(j)+".jpg"
            os.rename(image, new_name )
            print(new_name)
        
        #move it back to the previous directory and remove temporary directory
        images_temp_dir = glob(temp_dir+"*.jpg")
        for j, image in enumerate(images_temp_dir):
            upper_dir = folder + curr_class+str(j)+".jpg"
            print(upper_dir)
            os.rename(image, upper_dir)
        os.rmdir(temp_dir)

#split       
def split_train_test(train_pth, test_pth,subdirs, proportion):
    folders = glob(train_pth+"*/")
    classes = [get_end_slash(f) for f in folders]
    for i, folder in enumerate(folders):
        curr_class = classes[i]
        if(curr_class not in subdirs and subdirs!="all"):
            continue
        new_dir = test_pth + curr_class + "/"
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
    input_fname = 'processed_datasets/'+get_end_slash(pth) + str(IMG_WIDTH) + "x" + str(IMG_HEIGHT)+"x"+str(CHANNELS)+  '.h5'
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
    # if(CHANNELS!=1):
    X = hf.create_dataset(
        name= 'X',
        shape=(n_images,IMG_WIDTH, IMG_HEIGHT, CHANNELS),
        maxshape=(None, IMG_WIDTH, IMG_HEIGHT,None),
        compression="gzip",
        compression_opts=9)
    # else:
    #     X = hf.create_dataset(
    #         name= 'X',
    #         shape=(n_images,IMG_WIDTH, IMG_HEIGHT),
    #         maxshape=(None, IMG_WIDTH, IMG_HEIGHT),
    #         compression="gzip",
    #         compression_opts=9)
    label_lis = []
    x_ind =0
    for i, folder in enumerate(folders):
        images = glob(folder+"*.jpg")
        total_images = len(images)
        print(classes[i],total_images)
        for j, image_pth in enumerate(images):
            img = process_single_img(image_pth, IMG_WIDTH, IMG_HEIGHT)
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

def process_single_img(image_pth, width, height, channels = CHANNELS):
    img = cv2.imread(image_pth)
    if(channels==1):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img = cv2.resize(img, (width, height))
    img = img.reshape(width,height,channels)
    img = scale_X(img)
    return img


'''functions to add noise to an image to diversify the dataset
    must be fed a cv2 object'''
def translate_image_random(img):
    print(img.shape)
    (rows,cols, channels) = img.shape
    ty = get_rand_amount(-rows/2,rows/2)
    tx = get_rand_amount(-cols/2,cols/2)
    M = np.float32([[1,0,tx],[0,1, ty]])
    dst = cv2.warpAffine(img,M,(cols,rows), borderValue=(255,255,255))
    return dst

def get_rand_amount(range_lower, range_higher, get_float = False):
    randamt = 0
    if(get_float):
        randamt = random.uniform(-range_lower, range_higher)
    else:
        randamt = random.randint(int(range_lower), int(range_higher))
    return randamt

def rotate_rand_amount(img):
    (rows,cols, channels) = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),get_rand_amount(0,360),1)
    dst = cv2.warpAffine(img,M,(cols,rows), borderValue=(255,255,255))
    return dst

def shear_rand_amount(img):
    (rows, cols, channels) = img.shape
    sx = get_rand_amount(-0.2,0.2, True)
    sy = get_rand_amount(-0.2,0.2, True)
    M = np.float32([[1,sx,0],[sy,1, 0]])
    dst = cv2.warpAffine(img,M,(cols,rows), borderValue=(255,255,255))
    return dst


def vertical_flip(img):
    #(rows, cols, channels) = img.shape
    dst = cv2.flip( img, 0 )
    return dst

def horizontal_flip(img):
    #(rows, cols, channels) = img.shape
    dst = cv2.flip( img, 1 )
    return dst


def show_img(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)

#randomly changes some pixels throughout the image to black or white
def add_noise(image):
    row,col,channel = image.shape
    s_vs_p = 0.5
    amount = 0.004
    out = image
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
    out[coords] = 1
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
    out[coords] = 0
    return out

def blur_image(image):
    blur_size = get_rand_amount(7,15)
    median = cv2.blur(image,(blur_size,blur_size))
    return median

def augment_dataset(pth, subdirs, to_size):
    folders = glob(pth+"*/")
    classes = [get_end_slash(f) for f in folders]
    for i, folder in enumerate(folders):
        curr_class = classes[i]
        if(curr_class not in subdirs and subdirs!="all"):
            continue
        print(folders)
        images = glob(folder+"*.jpg")
        total_images = len(images)
        # print(classes[i],total_images)
        num_to_add = to_size - total_images
        if(num_to_add<=0):
            continue
        num_transforms_needed = int(num_to_add/7+1)
        if(num_transforms_needed>total_images):
            print("NOT ENOUGH IMAGES TO AUGMENT DATASET", curr_class)
            continue
        print(curr_class,num_transforms_needed)
        transforms = [blur_image, horizontal_flip, vertical_flip,
             rotate_rand_amount, shear_rand_amount, translate_image_random, add_noise]
        for i in range(num_transforms_needed):
            # print(i)
            pname= get_pic_name(images[i])
            im_store_dir = get_leading_directory(images[i])
            img = cv2.imread(images[i])
            for ind,transform in enumerate(transforms):
                transformed_img = transform(img)
                imname = im_store_dir+ pname[:pname.index(".jpg")] + "_transform"+str(ind)+".jpg"
                cv2.imwrite(imname,transformed_img)
                print(imname)
                # cv2.imstore(imname, transformed_img)
        
        #7 possible operations: blur, noise, hflip, vflip, rand_rotate, rand_shear, rand translate
        

# (X,y,y_oh,classes) = load_dataset("processed_datasets/clumped_kitchen_train_data64x64x3.h5")
# display_data(X,y)

# img = cv2.imread("test_classify/fruit116.jpg")
# img_1 = blur_image(img)
# show_img(img_1)
# augment_dataset(train_dir, "all", 240)

# filter_new_images("./new_train_data/",["Metal"])
# split_train_test(train_dir,test_dir,"all",0.2)
# to_h5py(train_dir)
# to_h5py(test_dir)


# print(get_class_list())
# print(get_class_list())
# to_h5py("./new_train_data/")
# to_h5py("./new_test_data/")

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
# split_train_test("./new_train_data/", "./new_test_data/", "all", 0.2)
# print(dat[0].shape)



# plt.show()


