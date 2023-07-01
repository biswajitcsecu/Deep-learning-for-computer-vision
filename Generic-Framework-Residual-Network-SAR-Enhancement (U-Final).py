#!/usr/bin/env python
# coding: utf-8

# ## **Framework  Enhancement SAR Dense Unet**

# In[3]:


#Import the necessary libraries:
get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')
import random
import cv2
import os
import gc
import numpy as np
import pandas as pd
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Input, Add,  Activation,add,LayerNormalization
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization,Concatenate,SeparableConv2D
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Dropout, Multiply
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, concatenate, UpSampling3D
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, Conv2DTranspose
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet201, DenseNet121
from tensorflow.keras.models import Model,  Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanIoU

import segmentation_models as sm
from segmentation_models.metrics import iou_score

import warnings

K.clear_session()
warnings.filterwarnings("ignore")


# ## **Data Reading and Train test split**

# In[4]:


#Data Directory
H,W,CH=[128,128,3]
high_dir = 'SAR/train/high/'
low_dir = 'SAR/train/low/'


# In[5]:


#Load Data------------------------------
High = []
for file in tqdm(sorted(os.listdir(high_dir))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(high_dir + file, target_size=(H,W,3))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        High.append(image)

High = np.array(High)

Low = []
for file in tqdm(sorted(os.listdir(low_dir))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(low_dir +  file, target_size=(H,W,3))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        Low.append(image)

Low = np.array(Low)


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(High, Low, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
gc.collect()


# ## **Visualization the image and masks**

# In[8]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
for i in tqdm(range(9)):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[index],cmap='gray')
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[9]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
for i in tqdm(range(9)):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(y_train[index],cmap='gray')
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[10]:


#Display test data
for i in tqdm(range(4)):
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


# ## **Building  Network architecture**

# In[74]:


#Define model
def CNN():
    input_img = Input(shape=(H, W, CH))
    l1 = Conv2D(64, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(input_img)
    l2 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l1)

    l3 = MaxPooling2D(padding='same')(l2)
    #l3 = Dropout(0.3)(l3)
    l4 = Conv2D(128, (3, 3),  padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l3)
    l5 = Conv2D(128, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l4)

    l6 = MaxPooling2D(padding='same')(l5)
    l7 = Conv2D(256, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l6)
    l7 = Conv2D(256, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l7)
    
    l8 = UpSampling2D()(l7)

    l9 = Conv2D(128, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l8)
    l10 = Conv2D(128, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l9)

    l11 = add([l5, l10])
    l12 = UpSampling2D()(l11)
    l13 = Conv2D(64, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l12)
    l14 = Conv2D(64, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l13)

    l15 = add([l14, l2])

    decoded = Conv2D(3, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l15)

    model = Model(input_img, decoded)
    
    return model


# # **Model build**

# In[75]:


# Test the model
input_shape=[H,W,CH]
model = CNN()


# ## **Loss functions**

# In[80]:


#iou metric
smooth =100
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    iouv = (intersection + smooth) / (sum_ - intersection + smooth)
    return iouv



# ## **Model compilation**

# In[77]:


#Model compile-------
estopping =  ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)    
model.compile(optimizer,loss=['mse'],metrics=['accuracy',iou]) 
model.summary()


# In[78]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True)


# ## **Model Training**

# In[79]:


#Model Training
nepochs=2
nbatch_size=32
history = model.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True, callbacks=[estopping],
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# In[81]:


df_result = pd.DataFrame(history.history)
df_result


# ## **Performance evaluation**

# In[82]:


# Plotting loss change over epochs---------------
nrange=nepochs
x = [i for i in tqdm(range(nrange))]
plt.plot(x,history.history['loss'])
plt.title('change in loss over epochs')
plt.legend(['training_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
plt.tight_layout()

# Plotting accuracy change over epochs---------------------
x = [i for i in tqdm(range(nrange))]
plt.plot(x,history.history['accuracy'])
plt.title('change in training accuracy coefitient over epochs')
plt.legend(['training accuracy'])
plt.xlabel('epochs')
plt.ylabel('training accuracy')
plt.show()
plt.tight_layout()

# Plotting iou accuracy change over epochs---------------------
x = [i for i in  tqdm(range(nrange))]
plt.plot(x,history.history['iou'])
plt.title('change in iou coefitient over epochs')
plt.legend(['iou'])
plt.xlabel('epochs')
plt.ylabel('iou')
plt.show()
plt.tight_layout()


# ## **Model Evaluation**

# In[83]:


# predict test images
predict_y = model.predict(x_test)

plt.figure(figsize=(15,15))
for i in  tqdm(range(0,9,3)):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
    plt.title('High image')
    
    plt.subplot(4,3,i+2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_test[i])
    plt.title('Lowimage')
    
    plt.subplot(4,3,i+3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(predict_y[i])
    plt.title('Output by model')

plt.show()


# ## **Model predictions**

# In[84]:


# Creating predictions on our test set-----------------
predictions = model.predict(x_test)


# In[96]:


# Ploting results for one image----------------

def plot_results_for_one_sample(sample_index):    
    pdimg =predictions[sample_index] 
    fig = plt.figure(figsize=(20,20))
    #image-------------------
    fig.add_subplot(1,4,1)
    plt.title('High image')
    plt.imshow(x_test[sample_index])
    plt.axis('off')
    plt.grid(None)
    #mask-----------
    fig.add_subplot(1,4,2)
    plt.title('Low image')
    plt.imshow(y_test[sample_index],cmap='gray')
    plt.axis('off')
    plt.grid(None)
    #Predicted mask------------
    fig.add_subplot(1,4,3)
    plt.title('Predicted image')  
    plt.imshow(pdimg,cmap='gray')
    plt.axis('off')
    plt.grid(None)
    #Segment---------------
    fig.add_subplot(1,4,4)
    plt.title("Enhanced image")
    plt.imshow(x_test[sample_index])
    plt.grid(None)
    plt.axis('off')  
    fig.tight_layout()    
plt.show()


# In[97]:


#Show predicted result---------------
plot_results_for_one_sample(1)


# In[98]:


#Show predicted result---------------
plot_results_for_one_sample(6)


# In[99]:


#Show predicted result---------------
plot_results_for_one_sample(10)


# In[100]:


#Show predicted result---------------
plot_results_for_one_sample(14)


# In[101]:


#Show predicted result---------------
plot_results_for_one_sample(20)


# In[102]:


#Show predicted result---------------
plot_results_for_one_sample(25)


# In[103]:


#Show predicted result---------------
plot_results_for_one_sample(30)


# ## **Make predictions**

# In[104]:


# Predict on train, val and test
preds_train = model.predict(x_train[:int(x_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(x_train[int(x_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(x_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train).astype(np.uint8)
preds_val_t = (preds_val).astype(np.uint8)
preds_test_t = (preds_test).astype(np.uint8)


# In[105]:


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 10))
ax[0].imshow(x_train[ix])
ax[0].axis('off')
ax[0].set_title('High Image')

ax[1].imshow(y_train[ix])
ax[1].axis('off')
ax[1].set_title('Low image')

ax[2].imshow(np.squeeze(preds_train_t[ix]))
ax[2].axis('off')
ax[2].set_title('Predicted image')

plt.show()


# In[ ]:




