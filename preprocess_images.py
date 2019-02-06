import cv2
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt
from constants import IMG_HEIGHT, IMG_WIDTH

image_dir = './train_data/'

def get_end_slash(f):
    return f[f.rindex("/",0,f.rindex("/"))+1:f.rindex("/")]

def resize_and_label(pth):
    input_lis = []
    label_lis = []
    folders = glob(pth+"*/")
    classes = [get_end_slash(f) for f in folders]
    for i, folder in enumerate(folders):
        images = glob(folder+"*.jpg")
        print(classes[i],len(images))
        for j, image_pth in enumerate(images):
            img = cv2.imread(image_pth, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            input_lis.append((np.array(img)))
            label_lis.append(i)
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
    data = np.array([X, y, classes])
    fname = get_end_slash(imdir) + '.npy'
    np.save(fname, data)
    print("saved to ", fname)

def load_dataset_from_file(fname):
    dataset = np.load(fname)
    return (dataset[0], dataset[1], dataset[2])

# process_images_to_file(image_dir)
# dat = load_dataset_from_file("train_data.npy")

# print(dat[0].shape)


