#!/usr/bin/env python
# coding: utf-8

# ## **Framework attention deep learning network with transfer learning**

# In[481]:


#Import the necessary libraries:
get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')
import random
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dense, Input, Add,  Activation,add,LayerNormalization
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization,Concatenate
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Dropout, Multiply
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, concatenate, UpSampling3D
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, Conv2DTranspose
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanIoU

import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score


import warnings

K.clear_session()
warnings.filterwarnings("ignore")


# ## **Importing and preprocessing dataset**

# In[556]:


#Load Data and Display

H,W,CH=[128,128,1]
image_dir = 'SAR/train/images/'
mask_dir = 'SAR/train/masks/'

def to_rgb_then_grayscale(image):
    image = tf.image.rgb_to_grayscale(image)
    return image


# In[557]:


#Loading------------------------------
Images = []
for file in tqdm(sorted(os.listdir(image_dir))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(image_dir + file, target_size=(H,W))
        image = tf.keras.preprocessing.image.img_to_array(to_rgb_then_grayscale(image)).astype('float32') / 255.
        Images.append(image)

Images = np.array(Images)

Masks = []
for file in tqdm(sorted(os.listdir(mask_dir))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(mask_dir +  file, target_size=(H,W))
        image = tf.keras.preprocessing.image.img_to_array(to_rgb_then_grayscale(image)).astype('float32') / 255.
        Masks.append(image)

Masks = np.array(Masks)


# In[558]:


x = Images
y = Masks
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train[0].shape)
print(y_train[0].shape)


# ## **Plot some of SAR images**

# In[559]:


#Display---------------------

figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
for i in range(9):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[index],cmap='gray')
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[560]:


#Display---------------------

figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
for i in range(9):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(y_train[index],cmap='gray')
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[561]:


for i in range(4):
    idx = np.random.randint(0,80)
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.title('Image', color = 'black', fontsize = 12)
    plt.imshow(x_train[idx],cmap='gray')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title('Mask', color = 'black', fontsize = 12)
    plt.imshow(y_train[idx],cmap='gray')
    plt.axis('off')


# ## **Building CNN model architecture**

# In[563]:


def conv2d_layer(input_t, num_filters, kernel_size=3, batch_norm = True):
    x = keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, padding='SAME', kernel_initializer='he_normal')(input_t)
    
    if batch_norm:
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        
    x = keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, padding='SAME', kernel_initializer='he_normal')(x)
    if batch_norm:
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
    
    return x
     


# In[564]:


def UNET_Model(input_t, base_filters = 32, batch_norm = True):
    c1 = conv2d_layer(input_t, base_filters, kernel_size=3, batch_norm = batch_norm)
    p1 = keras.layers.MaxPooling2D()(c1)
    p1 = keras.layers.Dropout(0.2)(p1)
    
    c2 = conv2d_layer(p1, base_filters*2)
    p2 = keras.layers.MaxPooling2D()(c2)
    p2 = keras.layers.Dropout(0.2)(p2)
    
    c3 = conv2d_layer(p2, base_filters*4)
    p3 = keras.layers.MaxPooling2D()(c3)
    p3 = keras.layers.Dropout(0.2)(p3)
    
    c4 = conv2d_layer(p3, base_filters*8)
    p4 = keras.layers.MaxPooling2D()(c4)
    p4 = keras.layers.Dropout(0.2)(p4)
    
    c5 = conv2d_layer(p4, base_filters*16)    
    
    u6 = keras.layers.Conv2DTranspose(base_filters * 8, (3, 3), strides=(2,2),  padding = 'same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    u6 = keras.layers.Dropout(0.2)(u6)
    c6 = conv2d_layer(u6, base_filters * 8, kernel_size = 3, batch_norm = batch_norm)
    
    u7 = keras.layers.Conv2DTranspose(base_filters * 4, (3, 3), strides=(2,2), padding = 'same')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    u7 = keras.layers.Dropout(0.2)(u7)
    c7 = conv2d_layer(u7, base_filters * 4, kernel_size = 3, batch_norm = batch_norm)
    
    u8 = keras.layers.Conv2DTranspose(base_filters * 2, (3, 3), strides=(2,2), padding = 'same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    u8 = keras.layers.Dropout(0.2)(u8)
    c8 = conv2d_layer(u8, base_filters * 2, kernel_size = 3, batch_norm = batch_norm)
    
    u9 = keras.layers.Conv2DTranspose(base_filters, (3, 3), strides=(2,2), padding = 'same')(c8)
    u9 = keras.layers.concatenate([u9, c1])
    u9 = keras.layers.Dropout(0.2)(u9)
    c9 = conv2d_layer(u9, base_filters, kernel_size = 3, batch_norm = batch_norm)
    
    output = keras.layers.Conv2D(1,1, activation='sigmoid')(c9)
    model = keras.models.Model(inputs=[input_t], outputs=[output])
    
    return model
     


# # **Create model**

# In[567]:


#Parameters
input_layer = keras.layers.Input((H, W, CH))
model = UNET_Model(input_layer, base_filters=32, batch_norm=True)


# ## **Loss functions**

# In[568]:


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


# In[569]:


#iou metric
smooth =100
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

#jacard_coef
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


# ## **Model compilation**

# In[570]:


#Model compile-------
model.compile( optimizer='adam',loss=[bce_jaccard_loss,dice_loss,'binary_crossentropy'],
              metrics=['accuracy', jacard_coef, iou]) #'sparse_categorical_crossentropy'
model.summary()


# In[571]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True)


# ## **Model Training**

# In[ ]:


#Model Training
nepochs=10
nbatch_size=48
history = model.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True,
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# In[532]:


df_result = pd.DataFrame(history.history)
df_result


# ## **Performance evaluation**

# In[533]:


# Plotting loss change over epochs---------------
nrange=nepochs
x = [i for i in range(nrange)]
plt.plot(x,history.history['loss'])
plt.title('change in loss over epochs')
plt.legend(['training_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
plt.tight_layout()

# Plotting accuracy change over epochs---------------------
x = [i for i in range(nrange)]
plt.plot(x,history.history['accuracy'])
plt.title('change in training accuracy coefitient over epochs')
plt.legend(['training accuracy'])
plt.xlabel('epochs')
plt.ylabel('training accuracy')
plt.show()
plt.tight_layout()

# Plotting jacard  accuracy change over epochs---------------------
x = [i for i in range(nrange)]
plt.plot(x,history.history['jacard_coef'])
plt.title('change in jacard_coef coefitient over epochs')
plt.legend(['jacard_coef'])
plt.xlabel('epochs')
plt.ylabel('jacard_coef')
plt.show()
plt.tight_layout()

# Plotting iou accuracy change over epochs---------------------
x = [i for i in range(nrange)]
plt.plot(x,history.history['iou'])
plt.title('change in iou coefitient over epochs')
plt.legend(['iou'])
plt.xlabel('epochs')
plt.ylabel('iou')
plt.show()
plt.tight_layout()


# ## **Model Evaluation**

# In[534]:


# predict test images
predict_y = model.predict(x_test)

plt.figure(figsize=(15,15))
for i in range(0,9,3):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
    plt.title('High image')
    
    plt.subplot(4,3,i+2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i])
    plt.title('Low image')
    
    plt.subplot(4,3,i+3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(predict_y[i])
    plt.title('Output by model')

plt.show()


# ## **Model predictions**

# In[535]:


# Creating predictions on our test set-----------------
predictions = model.predict(x_test)


# In[536]:


# create predictes mask--------------
def create_mask(predictions,input_shape=(W,H,1)):
    mask = np.zeros(input_shape)
    mask[predictions>0.5] = 1
    return mask


# In[537]:


# Ploting results for one image----------------

def plot_results_for_one_sample(sample_index):    
    mask =predictions[sample_index] #create_mask(predictions[sample_index])   for gray-scale
    fig = plt.figure(figsize=(20,20))
    #image-------------------
    fig.add_subplot(1,4,1)
    plt.title('Input image')
    plt.imshow(x_test[sample_index])
    plt.axis('off')
    plt.grid(None)
    #mask-----------
    fig.add_subplot(1,4,2)
    plt.title('Real mask')
    plt.imshow(y_test[sample_index],cmap='gray')
    plt.axis('off')
    plt.grid(None)
    #Predicted mask------------
    fig.add_subplot(1,4,3)
    plt.title('Predicted mask')  
    plt.imshow(mask,cmap='gray')
    plt.axis('off')
    plt.grid(None)
    #Segment---------------
    fig.add_subplot(1,4,4)
    plt.title("Segment image")
    plt.imshow(x_test[sample_index]*mask)
    plt.grid(None)
    plt.axis('off')  
    fig.tight_layout()    
plt.show()


# In[538]:


#Show predicted result---------------
plot_results_for_one_sample(1)


# In[539]:


#Show predicted result---------------
plot_results_for_one_sample(6)


# In[540]:


#Show predicted result---------------
plot_results_for_one_sample(10)


# In[541]:


#Show predicted result---------------
plot_results_for_one_sample(14)


# In[542]:


#Show predicted result---------------
plot_results_for_one_sample(20)


# In[543]:


#Show predicted result---------------
plot_results_for_one_sample(25)


# In[544]:


#Show predicted result---------------
plot_results_for_one_sample(30)


# In[ ]:




