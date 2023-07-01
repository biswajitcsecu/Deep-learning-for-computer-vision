#!/usr/bin/env python
# coding: utf-8

# ## **Deep learning  for SAR images using residual CNNs**

# In[27]:


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
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Add,  Activation
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import plot_model
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

# In[21]:


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

# In[22]:


# Define the residual block
def residual_block(input_tensor, filters):
    x = Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = Add()([x, input_tensor])
    output_tensor = Activation('relu')(x)
    return output_tensor

# Define the deep learning model
def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # Initial convolutional layer
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = Activation('relu')(x)

    # Stack of residual blocks
    for _ in range(16):
        x = residual_block(x, 64)

    # Final convolutional layer
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)

    # Output residual image
    output = Conv2D(3, (3, 3), padding='same')(x)

    # Combine input and output to create the final model
    model = Model(inputs=inputs, outputs=output)

    return model


# ## **Model compilation**

# In[28]:


#Model compile-------
input_shape=[H,W,CH]
model=build_model(input_shape)
model.compile(loss='mse', optimizer='adam',metrics=["acc"])
model.summary()


# In[29]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True)


# ## **Model Training**

# In[30]:


#Model Training
nepochs=5
nbatch_size=32
history = model.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True,
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# ## **Performance evaluation**

# In[34]:


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

# In[44]:


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




