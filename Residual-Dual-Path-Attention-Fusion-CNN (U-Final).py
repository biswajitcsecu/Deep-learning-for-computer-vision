#!/usr/bin/env python
# coding: utf-8

# ## **Framework Residual Dual-Path Attention-Fusion CNN(RDAPF)**

# In[312]:


#Import the necessary libraries:
get_ipython().run_line_magic('matplotlib', 'inline')
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
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Add,  Activation
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization,Concatenate
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
import warnings


K.clear_session()
warnings.filterwarnings("ignore")


# ## **Importing and preprocessing dataset**

# In[313]:


#Load Data and Display

H,W,CH=[128,128,3]
image_dir = 'SNCR/train/images/'
mask_dir = 'SNCR/train/masks/'

def to_rgb_then_grayscale(image):
    image = tf.image.rgb_to_grayscale(image)
    return image

def to_grayscale_then_rgb(image):
    image = tf.image.grayscale_to_rgb(image)
    return image


# In[314]:


#Loading------------------------------
Images = []
for file in tqdm(sorted(os.listdir(image_dir))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(image_dir + file, target_size=(H,W))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        Images.append(image)

Images = np.array(Images)

Masks = []
for file in tqdm(sorted(os.listdir(mask_dir))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(mask_dir +  file, target_size=(H,W))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        Masks.append(image)

Masks = np.array(Masks)


# In[315]:


x = Images
y = Masks
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train[0].shape)
print(y_train[0].shape)


# ## **Plot some of SAR images**

# In[316]:


#Display---------------------

figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
for i in range(9):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[index])
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[317]:


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


# ## **Building CNN model architecture**

# In[318]:


def rdapf_cnn(input_shape, num_classes):
    # Input layer
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, padding='same', activation='relu')(inputs)
    conv1 = Conv2D(64, 3, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, padding='same', activation='relu')(pool1)
    conv2 = Conv2D(128, 3, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, padding='same', activation='relu')(pool2)
    conv3 = Conv2D(256, 3, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, padding='same', activation='relu')(pool3)
    conv4 = Conv2D(512, 3, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Middle
    conv5 = Conv2D(1024, 3, padding='same', activation='relu')(pool4)
    conv5 = Conv2D(1024, 3, padding='same', activation='relu')(conv5)

    # Decoder
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(512, 3, padding='same', activation='relu')(up6)
    conv6 = Conv2D(512, 3, padding='same', activation='relu')(conv6)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(256, 3, padding='same', activation='relu')(up7)
    conv7 = Conv2D(256, 3, padding='same', activation='relu')(conv7)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(128, 3, padding='same', activation='relu')(up8)
    conv8 = Conv2D(128, 3, padding='same', activation='relu')(conv8)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1])
    conv9 = Conv2D(64, 3, padding='same', activation='relu')(up9)
    conv9 = Conv2D(64, 3, padding='same', activation='relu')(conv9)

    # Output layer
    output = Conv2D(num_classes, 1, activation='softmax')(conv9)

    # Create the model
    model = Model(inputs=inputs, outputs=output)

    return model


# # **Create model**

# In[324]:


# feature pyramid network (FPN)
# Input shape of your images
input_shape = (H, W, CH) 
num_classes = 3  
# Number of segmentation classes
model = rdapf_cnn(input_shape, num_classes)


# ## **Loss functions**

# In[325]:


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


# ## **Model compilation**

# In[326]:


#Model compile-------
model.compile(loss=['mse'], optimizer='adam', metrics=["acc"])
model.summary()


# In[327]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True)


# ## **Model Training**

# In[328]:


#Model Training
nepochs=1
nbatch_size=16
history = model.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True,
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# ## **Performance evaluation**

# In[333]:


#Plot history loss
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(0, 101, 25))
plt.show()

#Plot history Accuracy
plt.figure(figsize=(12,8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, 101, 25))
plt.show()


# ## **Model Evaluation**

# In[334]:


#Model evaluation
print("Input ----------------------------Ground Truth-------------------------------------Predicted Value")
for i in (range(6)):    
    r = random.randint(0, len(Images)-1)
    x, y = Masks[r],Images[r]
    x = x * 255.0
    
    x = np.clip(x, 0, 255).astype(np.uint8)
    y = y * 255.0
    y = np.clip(y, 0, 255).astype(np.uint8)
    
    x_inp=x.reshape(1,H,W,CH)
    result = model.predict(x_inp)
    result = result.reshape(H,W,CH)    
    result = np.clip(result, 0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(12,10))
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    ax = fig.add_subplot(1, 3, 1)
    plt.axis("off")
    ax.imshow(x)

    ax = fig.add_subplot(1, 3, 2)
    plt.axis("off")
    ax.imshow(y)
    
    ax = fig.add_subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(result)
plt.grid('off')    
plt.show()
print("--------------Done!----------------")
 


# In[ ]:




