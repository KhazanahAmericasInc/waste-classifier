import os
import tensorflow as tf
import keras
from keras.models import load_model
from  preprocess_images import load_dataset, load_history, process_single_img
from constants import *
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


"""
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
inputs: y_test - the true labels
        y_pred - predicted labels from the model
        classes - list of classes
        normalize - set to true if you want the output of the confusion matrix to be normalized
        cmap - the color scheme of the output
"""
def plot_confusion_matrix(y_test, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    #create figure
    plt.figure()

    #get confusion matrix with the actual labels and the predicted labels from scikit learn
    cm = confusion_matrix(y_test, y_pred)

    #print with 2 decimals of precision
    np.set_printoptions(precision=2)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print the confusion matrix
    print(cm)

    #set up the plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    #show the plot
    plt.show()

'''
given a model, input images, and labels, predict classes and also show the precision and recall based 
classification report
'''
def predict_and_report(model, test_x, test_y, class_names):
    predicted_classes = model.predict(test_x)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
    correct = np.where(predicted_classes==test_y)[0]
    incorrect = np.where(predicted_classes!=test_y)[0]
    print(classification_report(test_y, predicted_classes, target_names=class_names))
    print("num_correct: {}, num_incorrect: {}".format(len(correct), len(incorrect)))
    return predicted_classes

'''
plot the history of the training model
'''
def display_test_cvdata_curve(history):
    epochs = range(len(history['acc']))
    plt.plot(epochs, history['acc'], 'bo', label='Training accuracy')
    plt.plot(epochs, history['val_acc'], 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, history['loss'], 'bo', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

'''
predict the output of a single image
'''
def predict_single_img(image_path, width, height, model, classes):
    img = process_single_img(image_path,width, height)
    prediction = model.predict(np.array([img]))
    print(np.argmax(prediction))
    pred=np.argmax(prediction)
    pred_class = classes[pred]
    return pred_class


'''
display a single image given the image array and the predicted class
'''
def display_single_img(X,pred_class):
    plt.figure(figsize=[5,5])
    plt.plot()
    plt.imshow(X)
    plt.title("Class : {}".format(pred_class))
    plt.show()

'''
analyze a particular model; iee show the model training history, make predictions, predict on a 
single image, and show the confusion matrix
inputs: model_pth - string with the path to the model
        hist_pth - string with the path to the model training history
        test_set_pth - string with the path to the preprocessed test set
        width - the width of the input images
        height - the height of the input images
        samp_test_dir - string with path to directory to test model on
'''
def analyze_model(model_pth, hist_pth, test_set_pth, width,height, samp_test_dir=None):
    #1. load model and history
    model = load_model(model_pth)
    print("displaying:", model_pth)
    history = load_history(hist_pth)
    (test_X, test_Y, one_hot_test_Y, classes) = load_dataset(test_set_pth)
    model.summary()
    print(test_Y.shape)

    input()

    #display training history
    display_test_cvdata_curve(history)

    #make predictions
    predicted_classes = predict_and_report(model, test_X, test_Y, classes)
    input()

    #predict single:
    if(samp_test_dir!=None):
        predict_single_img(samp_test_dir,width,height,model,classes)

    #plot accuracy more specific to each class
    plot_confusion_matrix(test_Y, predicted_classes, classes, normalize=True)

# analyze_model("trained_models/waste_model_3d_in50x50.h5py", 
#     "model_history/waste_model_3d_in50x50.json",
#     "processed_datasets/test_data50x50.h5",50,50,samp_test_dir="test_classify/fruit116.jpg")

# analyze_model("trained_models/waste_model_3d_dropout_reg0.001_in50x50.h5py", 
#     "model_history/waste_model_3d_dropout_reg0.001_in50x50.json",
#     "processed_datasets/test_data50x50.h5",
#     50,50, samp_test_dir="test_classify/fruit116.jpg")

analyze_model("trained_models/waste_model_clumped_3d_dropout_reg0.001_in28x28.h5py", 
    "model_history/waste_model_clumped_3d_dropout_reg0.001_in28x28.json",
    "processed_datasets/clumped_test_data28x28x3.h5",
    28,28, samp_test_dir="test_classify/fruit116.jpg")

#for demo

#1. load model 1 and history
# model1 = load_model("trained_models/waste_model_3d_dropout_reg0.001_in50x50.h5py")
# print("displaying:", "trained_models/waste_model_3d_dropout_reg0.001_in50x50.h5py")
# history1 = load_history("model_history/waste_model_3d_dropout_reg0.001_in50x50.json")
# (test_X, test_Y, one_hot_test_Y, classes) = load_dataset("processed_datasets/test_data50x50.h5")
# model1.summary()

# input()

# #display training history
# display_test_cvdata_curve(history1)

# #make predictions
# predicted_classes = predict_and_report(model1, test_X, test_Y, classes)
# input()

# #predict single:
# predict_single_img("test_classify/fruit116.jpg",50,50,model1,classes)

# #plot accuracy more specific to each class
# plot_confusion_matrix(test_Y, predicted_classes, classes, normalize=True)

