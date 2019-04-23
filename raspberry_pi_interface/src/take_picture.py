from time import sleep
import cv2
import picamera
import numpy as np
import io
from picamera.array import PiRGBArray
from picamera import PiCamera
from PIL import Image as Img
from constants import *


#set up camera and wait small bit of time for it to initialize
camera = PiCamera()
sleep(0.1)

#flip horizontally
camera.hflip=True

#sets up a stream to convert from the camera input to OpenCV object later
stream = PiRGBArray(camera)

#generates the overlay object with appropriate padding
#can specify layer of overlay - defaults to 4
def generate_overlay(img, layer=4):
	print(img.size)
	pad = Img.new('RGB', (
            ((img.size[0]+31)//32)*32,
          		((img.size[1] + 15)//16)*16,
        ))
	pad.paste(img, (0, 0))
	o = camera.add_overlay(pad.tobytes(), size=pad.size)
	o.alpha = 32
	o.layer = layer
	return o


#method to take picture of waste user holds up in front of camera
#returns a numpy array/opencv image containing the picture taken
#also saves image in file 'img.jpg'
def take_waste_pic():
	#start preview and set resolution of camera
	camera.resolution = (480, 480)  # 1280,720
	camera.start_preview()

	#open up countdown and box images to overlay on top of camera
	img_3 = Img.open(OVERLAY_FOLDER+'/num_3.png')
	img_2 = Img.open(OVERLAY_FOLDER+'/num_2.png')
	img_1 = Img.open(OVERLAY_FOLDER+'/num_1.png')
	img_message = Img.open(OVERLAY_FOLDER+'/message.png')
	img_square = Img.open(OVERLAY_FOLDER+'/square_outline.png')

	#count down and overlay images one by one, first simply overlaying
	#the box for 3 seconds and then the numbers
	o_sq = generate_overlay(img_square, layer=3)
	o_message = generate_overlay(img_message)
	sleep(3)
	camera.remove_overlay(o_message)

	o_3 = generate_overlay(img_3)
	sleep(1)
	camera.remove_overlay(o_3)

	o_2 = generate_overlay(img_2)
	sleep(1)
	camera.remove_overlay(o_2)

	o_1 = generate_overlay(img_1)
	sleep(1)
	camera.remove_overlay(o_1)

	#take the picture and convert it into an opencv object
	camera.capture(stream, format="bgr")
	image = stream.array

	#crop the image to the bounding box
	print(image.shape[0], image.shape[1])
	height = 299
	width = 299
	x = int(image.shape[0]/2-width/2)
	y = int(image.shape[1]/2 - height/2)
	print(x, y)
	crop_img = image[y:y+height, x:x+height]

	#write image to file and return opencv object
	#cv2.imwrite(TEMP_IMG_PATH, crop_img)
	cv2.imwrite(TEMP_IMG_PATH, image)
	print(crop_img.shape)
	camera.remove_overlay(o_sq)
	sleep(1)
	camera.stop_preview()
	stream.truncate(0)
	return image

