import preprocess_images
from constants import IMG_WIDTH,IMG_HEIGHT

#tf and tf keras
import tensorflow as tf
import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


#imports needed to build model
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2


(train_X, train_Y, class_names) = preprocess_images.load_dataset_from_file('train_data.npy')
(test_X, test_Y, class_names_test) = preprocess_images.load_dataset_from_file('test_data.npy')


#convert 28X28 img to matrix of size 28x28x1
train_X = train_X.reshape(-1, IMG_WIDTH,IMG_HEIGHT, 3)
test_X = test_X.reshape(-1, IMG_WIDTH,IMG_HEIGHT, 3)

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)


#Split train data set into train, cross validation, and test set
train_X,valid_X,train_label,valid_label = \
    train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print("train,valid shape", train_X.shape, valid_X.shape, train_label.shape,valid_label.shape)
print(test_X.shape, test_Y_one_hot.shape, test_Y.shape)



#select sizes and number of epochs
batch_size = 50
epochs = 50
num_classes = len(class_names)
reg_lambda = 0.001
dropout = True
rgb = True


# # build model
waste_model=Sequential()
waste_model.add(Conv2D(32, kernel_size=(3,3), activation='linear', 
    input_shape=(IMG_WIDTH,IMG_HEIGHT,3),padding='same'))
waste_model.add(LeakyReLU(alpha=0.1))
waste_model.add(MaxPooling2D((2,2),padding='same'))
waste_model.add(Dropout(0.25))
waste_model.add(Conv2D(128,(3,3),activation='linear',padding='same'))
waste_model.add(LeakyReLU(alpha=0.1))
waste_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
waste_model.add(Dropout(0.25))
waste_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
waste_model.add(LeakyReLU(alpha=0.1))                  
waste_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
waste_model.add(Dropout(0.4))
waste_model.add(Flatten())
waste_model.add(Dense(128, activation='linear', activity_regularizer=l2(reg_lambda)))
waste_model.add(LeakyReLU(alpha=0.1))  
waste_model.add(Dropout(0.5))                
waste_model.add(Dense(num_classes, activation='softmax'))


# waste_model=Sequential()
# waste_model.add(Conv2D(32, kernel_size=(3,3), activation='linear', 
#     input_shape=(IMG_WIDTH,IMG_HEIGHT,1),padding='same'))
# waste_model.add(LeakyReLU(alpha=0.1))
# waste_model.add(MaxPooling2D((2,2),padding='same'))
# waste_model.add(Conv2D(128,(3,3),activation='linear',padding='same'))
# waste_model.add(LeakyReLU(alpha=0.1))
# waste_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# waste_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
# waste_model.add(LeakyReLU(alpha=0.1))                  
# waste_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# waste_model.add(Flatten())
# waste_model.add(Dense(128, activation='linear'))
# waste_model.add(LeakyReLU(alpha=0.1))            
# waste_model.add(Dense(num_classes, activation='softmax'))


#compile model
waste_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
waste_model.summary()

#train model
waste_train = waste_model.fit(train_X, train_label, batch_size=batch_size,
    epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

#save trained model for later
model_name = "waste_model2"+("_3d" if rgb else "") + ("_dropout" if dropout else "") + \
    ("_reg"+str(reg_lambda) if reg_lambda!=0 else "")
print("saved to:",model_name)
waste_model.save(model_name + ".h5py")

#evaluate model accuracy and loss
test_eval = waste_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

accuracy = waste_train.history['acc']
val_accuracy = waste_train.history['val_acc']
loss = waste_train.history['loss']
val_loss = waste_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#make predictions
predicted_classes = waste_model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

correct = np.where(predicted_classes==test_Y)[0]
incorrect = np.where(predicted_classes!=test_Y)[0]


#do a classification report
from sklearn.metrics import classification_report
target_names = ["Class {}".format(class_names_test[i]) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))
print("num_correct: {}, num_incorrect: {}".format(len(correct), len(incorrect)))