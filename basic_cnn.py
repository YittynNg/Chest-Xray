# import the necessary modules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import seaborn as sns

from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import glob
import math


# initialize the initial learning rate, number of epochs to train for, and batch size
INIT_LR = 1e-3 # initial learning rate
EPOCHS = 30 # epochs
BS = 16 # batch size

# Defining paths
TRAIN_PATH = "./dataset/train"
VAL_PATH = "./dataset/test"
len_train = len(glob.glob(TRAIN_PATH + "/*/*"))
print("Training data in total: ",len_train)
len_val = len(glob.glob(VAL_PATH + "/*/*"))
print("Test data in total: ",len_val)

# Moulding train images and augment images (TBT: shear_range, zoom_range)
train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range=15, horizontal_flip = True)
test_dataset = ImageDataGenerator(rescale=1./255)

# Reshaping test and validation images 
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size = (224,224),
    batch_size = BS,
    class_mode = 'categorical')
validation_generator = test_dataset.flow_from_directory(
    VAL_PATH,
    target_size = (224,224),
    batch_size = BS,
    class_mode = 'categorical')

# Build training model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation='softmax'))

# Compile model
print("\n[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss=CategoricalCrossentropy(from_logits=True),optimizer=opt,metrics=['accuracy'])

# Getting parameters
model.summary()

# Training the model
print("\n[INFO] training head...")
compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / BS))
steps_per_epoch = compute_steps_per_epoch(len_train)
val_steps = compute_steps_per_epoch(len_val)

hist = model.fit_generator(
    train_generator,
    steps_per_epoch = steps_per_epoch,
    epochs = EPOCHS,
    validation_data = validation_generator,
    validation_steps = val_steps
)

# Getting summary
summary=hist.history
print(summary)

# Evaluate model
print("\n[INFO] evaluating network...")
print("Training set evaluation: ", model.evaluate_generator(train_generator))
print("Testing set evaluation: ", model.evaluate_generator(validation_generator))

# Confusion matrix
print("\nClass indices: ", train_generator.class_indices)
y_actual, y_test = [],[]

for i in os.listdir(VAL_PATH + "/COVID19"):
    img=image.load_img(VAL_PATH + "/COVID19/" + i,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    pred=model.predict_classes(img)
    y_test.append(pred[0])
    y_actual.append(0)

for i in os.listdir(VAL_PATH + "/NORMAL"):
    img=image.load_img(VAL_PATH + "/NORMAL/" + i,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    pred=model.predict_classes(img)
    y_test.append(pred[0])
    y_actual.append(1)

for i in os.listdir(VAL_PATH + "/PNEUMONIA"):
    img=image.load_img(VAL_PATH + "/PNEUMONIA/" + i,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    pred=model.predict_classes(img)
    y_test.append(pred[0])
    y_actual.append(2)

y_actual=np.array(y_actual)
y_test=np.array(y_test)
cn=confusion_matrix(y_actual,y_test)  #row: actual; column: predict
sns.heatmap(ccn, annot=True, fmt="d") #0: Covid ; 1: Normal ; 2: Pneumonia


# print recall, precision, f1 score
recall = recall_score(y_actual,y_test,average=None)
precision = precision_score(y_actual,y_test,average=None)
f1_score = f1_score(y_actual,y_test,average=None)

print("\nRecall: ", recall)
print("Overall recall: ", np.mean(recall))
print("Precision: ", precision)
print("Overall precision: ", np.mean(precision))
print("f1 score: ", f1_score)
print("Overall precision: ", np.mean(f1_score))

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")

plt.figure()
plt.plot(np.arange(0, N), hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), hist.history["val_loss"], label="val_loss")
plt.title("Training Loss on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("./plot/plot_loss.png")

plt.figure()
plt.plot(np.arange(0, N), hist.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), hist.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("./plot/plot_accuracy.png")

# Save model
print("[INFO] saving COVID-19 detector model...")
model.save("./model/model_covid.h5")