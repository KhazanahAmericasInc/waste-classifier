Overview


The original goal of the waste classifier was to create an in house tool that would help people sort their trash before throwing it out. This required a machine learning to classify the trash, and a GUI that would tell the user what type of trash. The model was trained using Tensorflow Keras and the output of the classification is displayed on a touchscreen being run by a raspberry pi. Originally, the model tried to sort it into various categories including paper, metal, cardboard, etc. Later this was changed to classifying it directly into recycling, compost or trash, as this yielded higher accuracy. 


The older models can be found on the linux part of the dell innovation laptop, or in the Box. The maximum accuracy achieved on the test set was 82%. This can possibly be improved with possibly a larger CNN trained on a transfer learning model, or with some sensor input to create a multilayer hybrid system ( https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6236983/ ). The main issue with using a larger network is that the raspberry pi running the model will not be able to handle it.


Link to github repo:  https://github.com/KhazanahAmericasInc/waste-classifier


Tools Used
* Python3
    * The version of python used for this project was python3, specifically python 3.6
    * Python was used for all of the code written in order to make the code very understandable
* Keras/ Tensorflow
    * This was the library used to build and train the machine learning models
    * Keras also has some transfer learning models that can be used to make better, but larger models
* OpenCV (put version)
    * OpenCV was used for most of the preprocessing done to train the model on the laptop, as well as process the image on the scanner itself
* NumPy
    * NumPy was used to resize and reshape the images
* Scikit Learn
    * Used to analyze the results of the model
* Matplotlib
    * Used to plot graphs that help analyze the models created
* Virtualenv
    * Used to create the python virtual environment
    * To create type on linux shell (in the directory you want to create it): virtualenv venv


How to train the model
Notes:
* To run any method in any file, first start the virtual environment (created in step 1). Then you can do 1 of 2 possible methods:
    1. Shell method:
        1. Open up a python shell in the same directory as the file
        2. Import the file as a python module
        3. Run the methods with the appropriate parameters
    2. In-file method:
        1. Call the function in the file itself
        2. Run the file with python3 file_name.py
        3. There are examples of this in the files


1. Clone the waste classifier repository on github and install the requirements
    * To install the requirements first create a virtual environment under and then do the pip install -r requirements.txt on the shell
    * Run the create_needed_dirs method in preprocess_images.py
        * This creates necessary directories to run the program
    * Skip this step and if using the already set up project on the Khazanah
2. Gather the set of data and put each class of images in different folders (i.e. 1 folder with all images of glass, 1 folder with all images of metal, etc.)
    * To gather the data scrape images from google using Fatkun Bunch Extension
    * Also can take pictures manually
    * Put all of these folders of images in a new folder and place this folder in the root directory of the project (i.e. the same directory holding preprocess_images.py


    * Additionally make an extra folder to hold the test images (this folder can be empty)
    * Declare the folder names of the train directory and test directory inside preprocess_images.py like this:
3. Filter the images
    * Use the filter_new_images method in preprocess_images.py
        * This method converts each image into jpg, labels/enumerates the images, and ensures they can be opened by opencv
    * After filtering go through your images once to make sure everything is the way you want it to be
4. Split the dataset into train and test images
    * Use the split_train_test method in preprocess_images.py which randomly splits the images inside the train directory into the train and test directories
5. Augment the dataset (if desired)
    * This will add noise, rotation, blurs, etc. to your images and can make your small dataset up to 7 times larger artificially
    * Also useful if the dataset is overfitting
    * Use the augment_dataset method to do this
6. Set the correct constants for the width, height, and number of channels in the constants.py file
    * Setting channels to 3 processes the data as a bgr image, and to 1 makes the image bgr
    * Set the number of epochs and the batch size
7. Build the model to be used for training inside the cnn_trainer.py file:
    * Some sample models are created and loaded
    * Examples include: Transfer learning models, multi-layer perceptron models, etc.
    * Make sure the correct test/train set names are used at the very top 


    * You can also change the output file name if desired by adding/ removing strings from it:


8. Run cnn_trainer.py:
    * Can take between 1 minute and over 2 hours to train on CPU depending on input size and number of training parameter
    * To improve this training time, GPUs or TPUs can be using google cloud compute engine or a small GPU can be bought for the office
9. Compare/ analyze models:
    * Use tools in compare_models.py to analyze various things including how fast the model trained, find the precision and recall of the model, check out whether the model is underfitting or overfitting, and plot the confusion matrix
    * Use the analyze_model method to do all of these comparisons
    * Make changes to the model and then retrain it after


How to use the raspberry pi interface


If setting it up for the first time:
1. Flash an sd card with raspbian (version) and set it up on the raspberry pi. Also connect the raspberry pi camera to the pi. Test the camera to make sure it is working.
2. Copy the src folder in the raspberry_pi_interface folder (in the github repo) to the raspberry pi.
3. Create a virtual environment and start it.
4. Install the requirements (from the requirements.txt file) and deal with any dependency errors. You may need to sudo apt-get install some things.
5. Run gui.py and the application should load up.


To replace the model:
1. Paste the latest model (as an h5py file) into the src folder.
2. Replace the name of the model being used in the classify.py file. 


3D prints for the interface/ scanner


For the older interface:
(Place pic here)




Pi Camera Case:




_________________________________________________________________________________
For the latest interface: (exp laser vs 3d print)


* This was unable to be built on the 3D printer
* It used too much filament, and did not have a good finish
* It is best to laser cut the main base, and 3D print the top parts
    * These are all below
    * I was unable to do this on time, but if the following are laser cut and 3D printed, they can be put together to create a better interface for the waste classifier
* The final product should look like this:




Laser cut files:
    
  


Parts to 3D print for final design:


  
  
