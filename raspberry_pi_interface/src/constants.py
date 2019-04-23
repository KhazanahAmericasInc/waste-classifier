#name the temporary image path
TEMP_IMG_PATH = "img.jpg"

#directory to store images that have been classified
STORE_DIRECTORY = "Classified_Images"

#directory that stores the overlayed images
OVERLAY_FOLDER = "overlay_images"

#input dimensions for the images
WIDTH = 64
HEIGHT = 64
CHANNELS=3

#define what the classes are, and what is compost, trash and recycling
'''
CLASS_LIST = ['Metal','PaperTowel', 'Fruit', 'Pen',
     'Tea', 'Plastic', 'Cardboard', 'Wrappers','PaperCup','Glass']
COMPOST_LIST = ['PaperCup', 'Fruit', 'PaperTowel','Tea']
TRASH_LIST = ['Wrappers', 'Pen']
RECYCLE_LIST = ['Plastic', 'Cardboard', 'Glass','Metal']
'''

#clumped model
CLASS_LIST = ['Compost','Recycling','Trash']
COMPOST_LIST = ['Compost']
TRASH_LIST = ['Trash']
RECYCLE_LIST = ['Recycling']

#gui constants
LARGE_FONT = ("Verdana", 25)
MED_FONT = ("Verdana", 18)  
SM_FONT = ("Verdana", 12) 
XS_FONT = ("Verdana", 5)
THANK_YOU_TIME = 3000  # time the thank you page should hold for


