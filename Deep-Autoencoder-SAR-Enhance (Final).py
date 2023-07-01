#!/usr/bin/env python
# coding: utf-8

# ## **Deep autoencoder to SAR image enhancement**

# In[82]:


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
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

K.clear_session()


# ## **Load Data and Display**

# In[83]:


H,W,CH=[128,128,3]
higher = 'SAR/high/'
lower = 'SAR/low/'


# In[84]:


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


# In[85]:


x = clean
y = blurry
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train[0].shape)
print(y_train[0].shape)


# In[87]:


#Display---------------------
plt.style.use('dark_background')
figure, axes = plt.subplots(8,2, sharex=True, sharey=True, figsize=(12,10))
indexes=[]
for i in range(9):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[index])
    plt.axis("off")

plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.tight_layout()
plt.show()


# In[ ]:





# In[88]:


def build_autoencoder():
    input_img = Input(shape=(H, W, CH))

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    encoded = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = Conv2DTranspose(128, (3, 3), strides=(1, 1), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Autoencoder
    autoencoder = Model(input_img, decoded)
    return autoencoder


# In[90]:


autoencoder = build_autoencoder()
optimizer = Adam(lr=learning_rate)
autoencoder.compile(optimizer=optimizer, loss='mean_squared_error',metrics=[MeanSquaredError(), "acc", MeanAbsoluteError()])
autoencoder.summary()


# In[92]:


tf.keras.utils.plot_model(autoencoder, 'Model.png', show_shapes=True)


# In[ ]:


# Assuming you have loaded your training data into X_train
learning_rate = 0.001
nbatch_size = 32
nepochs = 20

early_stopping = EarlyStopping(patience=10, verbose=1)

history = autoencoder.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True,callbacks=[early_stopping],
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# ## **Performance evaluation**

# In[ ]:


#Plot history loss
plt.style.use('dark_background')
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(0, 101, 25))
plt.show()

#Plot history Accuracy
plt.style.use('dark_background')
plt.figure(figsize=(12,8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, 101, 25))
plt.show()


# ## **Evaluation**

# In[ ]:


#Evaluation
plt.style.use('dark_background')
print("\n\n Input --------------------\n Ground Truth\n-------------------------\n Predicted Value")

for i in (range(6)):    
    r = random.randint(0, len(clean)-1)
    x, y = blurry[r],clean[r]
    x_inp=x.reshape(1,H,W,CH)
    result = autoencoder.predict(x_inp)
    result = result.reshape(H,W,CH)

    fig = plt.figure(figsize=(12,10))
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(x)

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(y)
    
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(result)
plt.grid('off')    
plt.show()
print("--------------Done!----------------")
 
 


# In[ ]:




