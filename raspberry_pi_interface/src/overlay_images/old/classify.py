import os
import tensorflow as tf
import keras
from keras.models import load_model
import cv2
from glob import glob
from constants import *

#load the pretrained model
model = load_model("model.h5py")
TEMP_IMG_PATH = "img.jpg"
STORE_DIRECTORY = "Classified_Images"
WIDTH=299
HEIGHT=299
CLASS_LIST = ['plastic', 'cardboard', 'paper',
     'fruit', 'trash', 'metal', 'glass']
COMPOST_LIST = ['paper', 'fruit']
TRASH_LIST = ['metal','trash']
RECYCLE_LIST=['plastic','cardboard','glass']

def scale_X(X):
    return X/255.0

#preprocess the image given to be classified
def process_single_img(img):
    # img = cv2.imread(TEMP_IMG_PATH)
    #resize and normalize images
    img = cv2.resize(img, (WIDTH, HEIGHT))
    img = scale_X(img)
    return img

#predicts a single image given the numpy array of the image
def predict_single_img(img):
    #preprocesss the image
    processed_img = process_single_img(img)
    
    #predict the image using the preloaded model
    prediction = model.predict(np.array([processed_img]))
    pred=np.argmax(prediction)
    
    #match the numerica predicted class to the name
    pred_class = CLASS_LIST[pred]
    print(pred_class)
    
    #sort into trash, recycling or compost   
    waste_type = "Trash"
    if(pred_class in COMPOST_LIST):
		waste_type="Compost"
	else if pred_class in RECYCLE_LIST:
		waste_type= "Recycling"
    return (waste_type , pred_class)
    
def store_in_folder(waste_type):
	parent_dir = STORE_DIRECTORY+"/"+waste_type+"/"
	num = len(glob(parent_dir+"*.jpg"))
	print("current num images:",num)
	os.rename(TEMP_IMG_PATH, parent_dir
		+waste_type +str(num+1)+".jpg")
    
print("loaded")
