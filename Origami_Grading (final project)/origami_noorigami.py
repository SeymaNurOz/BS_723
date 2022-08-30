# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 13:51:17 2022

@author: seyma
"""

# resize image and force a new shape
from PIL import Image
from PIL import ImageFile
import splitfolders
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

import tensorflow as tf

import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import matplotlib.pyplot as plt
import numpy as np
import itertools






"""RESIZING THE IMAGES , change file dirs!!"""
      
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# yourpath = r'C:\Users\seyma\Desktop\BS723 ML\final-revised\origami_classification_data\oriwiki'

# for root, dirs, files in os.walk(yourpath, topdown=False):
#     for name in files:
#         x=os.path.join(root, name)
#         image = Image.open(x)
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
#         # report the size of the image
#         # print(image.size)
#         # # resize image and ignore original aspect ratio
#         img_resized = image.resize((100,100))
#         # print(img_resized.size)
#         img_resized.save(r"C:\Users\seyma\Desktop\BS723 ML\final-revised\resized-STEP1\ori"+"\RESIZED"+name)

"""SPLITING DATA AS TRAIN, VALIDATION AND TEST"""
"""oranlar nasıl olsunnn,test olsun mu"""

# splitfolders.ratio('resized-STEP1', output="splitted_STEP1", seed=14, ratio=(.7, 0.2,0.1)) 


"""a"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/predict")
mode = ap.parse_args().mode

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=False):
    """
    arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    """
     
    if cmap is None:
        cmap = plt.get_cmap('Oranges')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else 3*cm.max() / 4
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylim(len(target_names)-0.5, -0.5)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    # plt.savefig(title + '.png', dpi=500, bbox_inches = 'tight')
    plt.show()

# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1))#,len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1))#,len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

# Define data generators
train_dir = r"C:\Users\seyma\Desktop\BS723 ML\final-revised\splitted_STEP1\train"
val_dir = r"C:\Users\seyma\Desktop\BS723 ML\final-revised\splitted_STEP1\val"
test_dir = r"C:\Users\seyma\Desktop\BS723 ML\final-revised\splitted_STEP1\test"


"""CHANGE THIS VALUES ACCORDING TO THE NUMBER IN THE CASE"""
num_train=11189
num_val=3196
num_test=1600

batch_size = 64
num_epoch = 20

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(100,100),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(100,100),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical')

###BURAYI KONTROL ET
test_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=(100,100),
        batch_size=64,
        color_mode="rgb",
        class_mode='categorical')


# Create the model
model = Sequential()
##input shape cahenges according to img size and color mode
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100,100,3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
    #Dense(num of groups,..)

# mode="train"

"""TRAINING"""
if mode == "train":
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)
    
    model.save_weights('model_ph2_20.h5')
    plot_model_history(model_info)

# mode="train2"

# if mode == "train2":
#     model.load_weights('model_ph2_40.h5')
#     model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
#     model_info = model.fit_generator(
#             train_generator,
#             steps_per_epoch=num_train // batch_size,
#             epochs=num_epoch,
#             validation_data=validation_generator,
#             validation_steps=num_val // batch_size)
    
#     model.save_weights('model_ph1_80.h5')
#     plot_model_history(model_info)


mode="predict"	

"""PREDICTION"""

if mode=="predict":
    model.load_weights('model_ph2_20.h5')
    
    origami_dict = {0: "NO-ORIGAMI", 1:"ORIGAMI"}
    
    predictions = model.predict(test_generator)
    # print(validation_generator)
    # print(predictions)
    count=0
    for pred in predictions:
        maxindex = int(np.argmax(pred))
        predicted_type=origami_dict[maxindex]
        count+=1
    print(count)
    
  

    """TEST DATA"""
    
    X_train, y_train = next(test_generator)
    # print(y_train)
    # print(predictions)
    Y_pred=[]
    Y_real=[]
    for i in range(len(X_train)):
        act=y_train[i]
        act_tt=origami_dict[int(np.argmax(act))]
        Y_real.append(act_tt)
        pred= predictions[i]
        maxindex = int(np.argmax(pred))
        predicted_tt=origami_dict[maxindex]
        Y_pred.append(predicted_tt)
        plt.imshow(X_train[i])
        plt.title("Predicted:"+predicted_tt+"\n"+"Actual:"+act_tt)
        # plt.savefig( r"C:\Users\seyma\Desktop\BS723 ML\final-revised\images_ph1\\"+ str(i)+ '.png', dpi=1000, bbox_inches = 'tight')
        # break
    # print(Y_pred)
    # print(Y_real)
    

    """confussion matrix"""



    

# simply call the confusion_matrix function to build a confusion matrix

    cm = confusion_matrix(Y_real,Y_pred)
    # print(cm)
    target_names = ( "NO-0RIGAMI","ORIGAMI")
    plot_confusion_matrix(cm, target_names)    
        
    
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(Y_real,Y_pred)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Y_real,Y_pred,average="weighted")
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(Y_real,Y_pred,average="weighted")
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_real,Y_pred,average="weighted")
    print('F1 score: %f' % f1)



    
 
 



