#!/usr/bin/env python
# coding: utf-8

# ## **Geometric similarity deep learning  network super resolution**

# In[7]:


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
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Input, Add,  Activation
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import warnings


K.clear_session()
warnings.filterwarnings("ignore")


# ## **Importing and preprocessing dataset**

# In[3]:


#Load Data and Display

H,W,CH=[128,128,3]
higher = 'SAR/high/'
lower = 'SAR/low/'

#Loading------------------------------
clean = []
for file in tqdm(sorted(os.listdir(higher))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(higher + '/' + file, target_size=(H,W))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        clean.append(image)

clean = np.array(clean)
blurry = []
for file in tqdm(sorted(os.listdir(lower))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(lower + '/' + file, target_size=(H,W))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        blurry.append(image)

blurry = np.array(blurry)


# In[4]:


x = clean
y = blurry
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train[0].shape)
print(y_train[0].shape)


# ## **Plot some of SAR images**

# In[5]:


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


# ## **Building CNN model architecture**

# In[11]:


def residual_block(x, filters):
    """Residual block with two convolutional layers."""
    x_copy = x

    x = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = Add()([x, x_copy])
    return x

def GSN_model():
    """Geometric Similarity Network (GSN) model for image super-resolution."""
    input_img = Input(shape=(H, W, CH))  # Input image shape (None, None, 3)

    # Preprocessing
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_img)

    # Residual blocks
    for _ in range(16):
        x = residual_block(x, 64)

    # Postprocessing
    x = Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Add()([x, input_img])

    # Upscaling
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    # Model
    model = Model(inputs=input_img, outputs=x)
    return model


# In[12]:


# Define the loss function
def vae_loss(y_true, y_pred):
    mse_loss = MeanSquaredError()(y_true, y_pred)
    kl_loss = -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
    return mse_loss + kl_loss


# ## **Model compilation**

# In[13]:


#Model compile-------
# Build the GSN model
model = GSN_model()
model.compile(loss=['mse', vae_loss], optimizer='adam',metrics=["acc"])
model.summary()


# In[14]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True)


# ## **Model Training**

# In[15]:


#Model Training
nepochs=2
nbatch_size=32
history = model.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True,
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# ## **Performance evaluation**

# In[16]:


#Plot history loss
plt.figure(figsize=(10,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(0, 101, 25))
plt.show()

#Plot history Accuracy
plt.figure(figsize=(10,8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, 101, 25))
plt.show()


# ## **Model Evaluation**

# In[17]:


#Model evaluation
print("Input ----------------------------Ground Truth-------------------------------------Predicted Value")
for i in (range(6)):    
    r = random.randint(0, len(clean)-1)
    x, y = blurry[r],clean[r]
    x = x * 255.0
    
    x = np.clip(x, 0, 255).astype(np.uint8)
    y = y * 255.0
    y = np.clip(y, 0, 255).astype(np.uint8)
    
    x_inp=x.reshape(1,H,W,CH)
    result = model.predict(x_inp)
    result = result.reshape(H,W,CH)    
    #result = result*255.0
    #result = np.clip(result, 0, 255).astype(np.uint8)

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
    plt.imshow(result,cmap='gray')
plt.grid('off')    
plt.show()
print("--------------Done!----------------")
 


# In[ ]:




