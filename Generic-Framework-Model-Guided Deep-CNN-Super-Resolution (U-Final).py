#!/usr/bin/env python
# coding: utf-8

# In[65]:


#Import the necessary libraries:
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Add,  Activation, Dropout, add
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization,Concatenate
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
import warnings


K.clear_session()
warnings.filterwarnings("ignore")


# ## **Importing and preprocessing dataset**

# In[2]:


H,W,CH=[128,128,3]
base_dir = "SAR/"


# In[3]:


#Load Data and Display
def load_data(path):
    Low = []
    for file in tqdm(sorted(os.listdir(path+'low'))):
        if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
            image = tf.keras.preprocessing.image.load_img(path+'low/' + file, target_size=(H,W))
            image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
            Low.append(image)
    Low = np.array(Low)
    
    High = []
    for file in tqdm(sorted(os.listdir(path+'high'))):
        if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
            image = tf.keras.preprocessing.image.load_img(path+'high/' +  file, target_size=(H,W))
            image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
            High.append(image)
    High = np.array(High)
    
    return Low, High


# In[4]:


x_train, y_train =  load_data(base_dir+'train/')
x_test, y_test = load_data(base_dir+'val/')
print(x_train[0].shape)
print(y_train[0].shape)


# ## **Plot some of SAR images**

# In[49]:


#Display---------------------

figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
figure.suptitle('Low resolution Samples')
for i in range(9):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[index])
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[50]:


#Display---------------------

figure, axes = plt.subplots(8,2, figsize=(10,10))
figure.suptitle('Low resolution Samples')
indexes=[]
for i in range(9):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(y_train[index],cmap='gray')
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# ## **Model-Guided Deep CNNs**

# In[107]:


def MGDNet():
    # Define the generator model architecture
    input_shape = (H, W, CH)
    inputs = tf.keras.Input(shape=input_shape)
    
    # Extract low-resolution features
    conv1 = Conv2D(64, kernel_size=9, activation='relu', padding='same')(inputs)
    conv2 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv1)
    
    # Residual blocks
    residual = conv2
    for _ in tqdm(range(32)):
        residual = Conv2D(64, kernel_size=3, activation='relu', padding='same')(residual)
    conv3 = Conv2D(64, kernel_size=3, padding='same')(residual)
    outputs = tf.keras.layers.add([conv2, conv3])
    
    # Upsampling
    upsample = UpSampling2D(size=1)(outputs)
    outputs = Conv2D(3, kernel_size=9, padding='same')(upsample)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


# In[110]:


#MGDNet
model=MGDNet() #good


#  ## **Loss functions**

# In[111]:


# Define the perceptual loss
def perceptual_loss(y_true, y_pred):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    model = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)
    return tf.keras.losses.mean_squared_error(model(y_true), model(y_pred))


# In[112]:


#Loss functions
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


#   ## **Model compilation**

# In[113]:


#Model compile-------
model.compile(loss=[dice_loss,'mse'], optimizer='adam', metrics=["acc",dice_coeff])
model.summary()


# In[114]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True)


# ## **Model Training**

# In[ ]:


#Model Training
nepochs=1
nbatch_size=16
history = model.fit(x=x_train, y=y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True,
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


#  ## **Performance evaluation**

# In[ ]:


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


# In[ ]:


#Model evaluation
print("Input ----------------------------Ground Truth-------------------------------------Predicted Value")
for i in (range(6)):    
    idx = random.randint(0, len(x_train)-1)
    x, y = x_train[idx],y_train[idx]        
    x_inp=x.reshape(1,H,W,CH)
    result = model.predict(x_inp)
    result = result.reshape(H,W,CH) 

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


# predict/clean test images
predict_y = model.predict(x_test, batch_size=16)

plt.figure(figsize=(15,25))
for i in range(0,8,2):
    plt.subplot(4,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap='gray')
    
    plt.subplot(4,2,i+2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(predict_y[i], cmap='gray')

plt.show()


# In[ ]:


predict_y = model.predict(x_test)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,6))
ax1.imshow(x_test[11])
ax1.title.set_text("low-res image ")
ax2.imshow(y_test[11])
ax2.title.set_text("high-res image ")
ax3.imshow(predict_y[11])
ax3.title.set_text("model's output")


# In[ ]:




