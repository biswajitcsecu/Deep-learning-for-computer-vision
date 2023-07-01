#!/usr/bin/env python
# coding: utf-8

# ## **Framework pooling-based feature pyramid network**

# In[214]:


#Import the necessary libraries:
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import random
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Add,  Activation
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization,Concatenate
from tensorflow.keras.layers import Reshape, Conv2DTranspose, LeakyReLU,Lambda
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
import warnings


K.clear_session()
warnings.filterwarnings("ignore")


# ## **Importing and preprocessing dataset**

# In[215]:


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


# In[216]:


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


# In[217]:


x = Images
y = Masks
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train[0].shape)
print(y_train[0].shape)


# ## **Plot some of SAR images**

# In[218]:


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


# In[219]:


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

# In[220]:


##Multi-feature fusion cross neural network for salient object detection
def mffcn():
    # Input layer
    input_layer = Input(shape=(H, W, CH))

    # Convolutional block 1
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)

    # Convolutional block 2
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2)

    # Convolutional block 3
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv3)

    # Convolutional block 4
    conv4 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    conv4 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(conv4)

    # Fusion of features from previous layers
    fusion = Concatenate()([conv3, conv4])
    fusion = Conv2D(filters=256, kernel_size=(1, 1), activation='relu')(fusion)

    # Up-sampling
    upsample1 = UpSampling2D(size=(1, 1))(fusion)
    upsample2 = UpSampling2D(size=(1, 1))(fusion)

    # Final output layers
    output1 = Conv2D(filters=3, kernel_size=(1, 1), activation='sigmoid', name='output1')(upsample1)
    output2 = Conv2D(filters=3, kernel_size=(1, 1), activation='sigmoid', name='output2')(upsample2)

    # Define the model with input and output layers
    model = Model(inputs=input_layer, outputs=[output1, output2])

    return model



# # **Create model**

# In[221]:


# feature pyramid network (FPN)
model= mffcn( )


# ## **Loss functions**

# In[222]:


def perceptual_loss(y_true, y_pred):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)
    loss_model.trainable = False
    loss = tf.reduce_mean(tf.square(loss_model(y_true) - loss_model(y_pred)))
    return loss


# In[223]:


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

# In[224]:


#Model compile-------
model.compile(loss=[dice_loss,'mse',perceptual_loss], optimizer='adam', metrics=[ "acc", dice_coeff])
model.summary()


# In[225]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True)


# ## **Model Training**

# In[226]:


#Model Training
nepochs=1
nbatch_size=16
history = model.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True,
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# ## **Performance evaluation**

# In[238]:


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
plt.plot(history.history['output1_dice_coeff'])
plt.plot(history.history['val_output1_loss'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, 101, 25))
plt.show()


# ## **Model Evaluation**

# In[255]:


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
    result =tf.squeeze(result[0]) 
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




