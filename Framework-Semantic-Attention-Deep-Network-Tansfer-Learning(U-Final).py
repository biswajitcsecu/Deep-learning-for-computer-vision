#!/usr/bin/env python
# coding: utf-8

# ## **Framework attention deep learning network with transfer learning**

# In[179]:


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
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dense, Input, Add,  Activation,add
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization,Concatenate
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Dropout, Multiply
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, concatenate, UpSampling3D
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
import warnings

K.clear_session()
warnings.filterwarnings("ignore")


# ## **Importing and preprocessing dataset**

# In[180]:


#Load Data and Display

H,W,CH=[128,128,3]
image_dir = 'CaFFe/train/images/'
mask_dir = 'CaFFe/train/masks/'

def to_rgb_then_grayscale(image):
    image = tf.image.rgb_to_grayscale(image)
    return image

def to_grayscale_then_rgb(image):
    image = tf.image.grayscale_to_rgb(image)
    return image


# In[181]:


#Loading------------------------------
Images = []
for file in tqdm(sorted(os.listdir(image_dir))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(image_dir + file, target_size=(H,W))
        image = tf.keras.preprocessing.image.img_to_array((image)).astype('float32') / 255.
        Images.append(image)

Images = np.array(Images)

Masks = []
for file in tqdm(sorted(os.listdir(mask_dir))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(mask_dir +  file, target_size=(H,W))
        image = tf.keras.preprocessing.image.img_to_array(to_rgb_then_grayscale(image)).astype('float32') / 255.
        Masks.append(image)

Masks = np.array(Masks)


# In[182]:


x = Images
y = Masks
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train[0].shape)
print(y_train[0].shape)


# ## **Plot some of SAR images**

# In[183]:


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


# In[184]:


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


# In[185]:


for i in range(4):
    idx = np.random.randint(0,80)
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.title('High Resolution Imge', color = 'black', fontsize = 12)
    plt.imshow(x_train[idx],cmap='gray')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title('low Resolution Image ', color = 'black', fontsize = 12)
    plt.imshow(y_train[idx],cmap='gray')
    plt.axis('off')


# ## **Building CNN model architecture**

# In[187]:


def build_unet(num_classes, input_shape):
    # Load the pre-trained VGG16 model
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the weights of the VGG16 model
    for layer in vgg16.layers:
        layer.trainable = False
    
    # Encoder (downsampling)
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bridge (bottleneck)
    bridge = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    bridge = Conv2D(512, 3, activation='relu', padding='same')(bridge)
    
    # Decoder (upsampling)
    upconv3 = UpSampling2D(size=(2, 2))(bridge)
    upconv3 = Conv2D(256, 2, activation='relu', padding='same')(upconv3)
    merge3 = concatenate([conv3, upconv3], axis=3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(merge3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
    
    upconv2 = UpSampling2D(size=(2, 2))(conv4)
    upconv2 = Conv2D(128, 2, activation='relu', padding='same')(upconv2)
    merge2 = concatenate([conv2, upconv2], axis=3)
    conv5 = Conv2D(128, 3, activation='relu', padding='same')(merge2)
    conv5 = Conv2D(128, 3, activation='relu', padding='same')(conv5)
    
    upconv1 = UpSampling2D(size=(2, 2))(conv5)
    upconv1 = Conv2D(64, 2, activation='relu', padding='same')(upconv1)
    merge1 = concatenate([conv1, upconv1], axis=3)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(merge1)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(conv6)
    
    # Output layer
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv6)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


# # **Create model**

# In[196]:


# Number of semantic classes
num_classes = 3 
# Input shape of your images
input_shape = (H, W, CH)  

model = build_unet(num_classes, input_shape)


# ## **Loss functions**

# In[197]:


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

# In[198]:


#Model compile-------
model.compile( optimizer='adam',loss=['sparse_categorical_crossentropy'], metrics=['accuracy',])
model.summary()


# In[199]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True)


# ## **Model Training**

# In[200]:


#Model Training
nepochs=1
nbatch_size=32
history = model.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True,
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# ## **Performance evaluation**

# In[201]:


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
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, 101, 25))
plt.show()


# ## **Model Evaluation**

# In[202]:


## **Model prediction**
predict_y = model.predict(x_test)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,6))
ax1.imshow(x_test[11])
ax1.title.set_text("low-res image ")
ax2.imshow(y_test[11])
ax2.title.set_text("high-res image ")
ax3.imshow(predict_y[11])
ax3.title.set_text("model's output")
plt.show()


# In[204]:


# predict test images
predict_y = model.predict(x_test)

plt.figure(figsize=(15,25))
for i in range(0,9,3):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap='gray')
    plt.title('High image')
    
    plt.subplot(4,3,i+2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap='gray')
    plt.title('Low image')
    
    plt.subplot(4,3,i+3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(predict_y[i], cmap='gray')
    plt.title('Output by model')

plt.show()


# In[ ]:





# In[ ]:




