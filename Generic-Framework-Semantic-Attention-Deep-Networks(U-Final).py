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

# In[334]:


#Load Data and Display

H,W,CH=[128,128,3]
image_dir = 'SAR/train/images/'
mask_dir = 'SAR/train/masks/'

def to_rgb_then_grayscale(image):
    image = tf.image.rgb_to_grayscale(image)
    return image


# In[335]:


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
        image = tf.keras.preprocessing.image.img_to_array((image)).astype('float32') / 255.
        Masks.append(image)

Masks = np.array(Masks)


# In[336]:


x = Images
y = Masks
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train[0].shape)
print(y_train[0].shape)


# ## **Plot some of SAR images**

# In[337]:


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


# In[338]:


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


# In[339]:


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

# In[367]:


#(1) EGDN
#(2) TGNM


# ## **Edge  Guidance Deep Network (EGDN)**

# In[401]:


#Edge  Guidance Deep Network (EGDN)
def EGDN(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Contracting path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Expanding path
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(256, 2, activation='relu', padding='same')(up5)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(128, 2, activation='relu', padding='same')(up6)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(64, 2, activation='relu', padding='same')(up7)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # Output layer
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv7)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model


# ## **Transformer Attention Guidance Deep Network Model (TAGDN)**

# In[424]:


#TAGDN----------
def TAGDN(input_shape, num_classes):
    # Encoder backbone
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    # Decoder
    inputs = base_model.input
    encoder_output = base_model.output
    
    # Self-attention mechanism
    attention = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(encoder_output)
    attention = LayerNormalization()(attention)
    attention = Dropout(0.5)(attention)
    attention = Add()([encoder_output, attention])

    # Classification head
    x = Conv2D(filters=num_classes, kernel_size=1, activation='softmax')(attention)
    outputs = tf.keras.layers.UpSampling2D(size=(input_shape[0] // 4, input_shape[1] // 4))(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# ## **Transformer Based Bilateral Attention Guidance Deep Network**

# In[454]:


#BABNet
# Define the Transformer block
def transformer_block(inputs, filters, kernel_size, strides, padding):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

# Define the Bilateral Attention block
def bilateral_attention_block(inputs, filters):
    x = layers.Conv2D(filters, 1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

# Define the model architecture
def BABNet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    enc_conv1 = transformer_block(inputs, 64, 3, 1, 'same')
    enc_pool1 = layers.MaxPooling2D(pool_size=(2, 2))(enc_conv1)
    enc_conv2 = transformer_block(enc_pool1, 128, 3, 1, 'same')
    enc_pool2 = layers.MaxPooling2D(pool_size=(2, 2))(enc_conv2)
    enc_conv3 = transformer_block(enc_pool2, 256, 3, 1, 'same')
    enc_pool3 = layers.MaxPooling2D(pool_size=(2, 2))(enc_conv3)
    
    # Bottleneck
    bottleneck_conv = transformer_block(enc_pool3, 512, 3, 1, 'same')
    
    # Decoder
    dec_upsamp1 = layers.UpSampling2D(size=(2, 2))(bottleneck_conv)
    dec_conv1 = transformer_block(dec_upsamp1, 256, 3, 1, 'same')
    dec_att1 = bilateral_attention_block(dec_conv1, 256)
    dec_concat1 = layers.Concatenate()([dec_conv1, dec_att1])
    
    dec_upsamp2 = layers.UpSampling2D(size=(2, 2))(dec_concat1)
    dec_conv2 = transformer_block(dec_upsamp2, 128, 3, 1, 'same')
    dec_att2 = bilateral_attention_block(dec_conv2, 128)
    dec_concat2 = layers.Concatenate()([dec_conv2, dec_att2])
    
    dec_upsamp3 = layers.UpSampling2D(size=(2, 2))(dec_concat2)
    dec_conv3 = transformer_block(dec_upsamp3, 64, 3, 1, 'same')
    dec_att3 = bilateral_attention_block(dec_conv3, 64)
    dec_concat3 = layers.Concatenate()([dec_conv3, dec_att3])
    
    # Output
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(dec_concat3)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model


# ## **Spatial Adaptive Convolution Based Content-Aware Network (SAC-CAN)**

# In[462]:


#SACCAN

def spatial_adaptive_conv(x, filters, kernel_size):
    # Spatial Adaptive Convolution
    x_shape = x.get_shape().as_list()
    pooled_features = MaxPooling2D(pool_size=(2, 2))(x)
    upsampled_features = UpSampling2D(size=(2, 2))(pooled_features)
    residual = Conv2D(filters, kernel_size, padding='same')(upsampled_features)
    spatial_attention = Conv2D(x_shape[-1], kernel_size, activation='sigmoid', padding='same')(residual)
    return spatial_attention * x

def content_aware_block(x, filters, kernel_size):
    conv1 = Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    conv2 = Conv2D(filters, kernel_size, activation='relu', padding='same')(conv1)
    return conv2

def SACCAN(input_shape, num_classes):
    inputs = Input(input_shape)
    
    # Encoding Path
    conv1 = content_aware_block(inputs, 64, 3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = content_aware_block(pool1, 128, 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = content_aware_block(pool2, 256, 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = content_aware_block(pool3, 512, 3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Decoding Path
    up5 = Concatenate()([UpSampling2D(size=(2, 2))(conv4), conv3])
    conv5 = content_aware_block(up5, 256, 3)
    
    up6 = Concatenate()([UpSampling2D(size=(2, 2))(conv5), conv2])
    conv6 = content_aware_block(up6, 128, 3)
    
    up7 = Concatenate()([UpSampling2D(size=(2, 2))(conv6), conv1])
    conv7 = content_aware_block(up7, 64, 3)
    
    # Final convolutional layer
    output = Conv2D(num_classes, 1, activation='softmax')(conv7)
    
    model = Model(inputs=inputs, outputs=output)
    
    return model


# ## **Adaptive Feature Selection Network (AFSN)**

# In[479]:


#AFSN
def AFSN(input_shape, num_classes):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Adaptive Feature Selection
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up5 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4)
    up5 = concatenate([up5, conv3], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv2], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv1], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # Output
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv7)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model


# ## **Adaptive Feature Selection with Local Relationship Upsampling Network (LRSN)**

# In[551]:


#LRSN:Adaptive Feature Selection with Local Relationship Upsampling Network (LRSN)
def adaptive_feature_selection(input_tensor, num_filters):
    conv = Conv2D(num_filters, 3, activation='relu', padding='same')(input_tensor)
    conv = Conv2D(num_filters, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(num_filters, 3, activation='relu', padding='same')(conv)
    return conv

def local_relationship_upsampling(input_tensor, skip_tensor, num_filters):
    upsample = UpSampling2D(size=(2, 2))(input_tensor)
    conv = Conv2D(num_filters, 3, activation='relu', padding='same')(upsample)
    concat = Concatenate()([conv, skip_tensor])
    conv = Conv2D(num_filters, 3, activation='relu', padding='same')(concat)
    conv = Conv2D(num_filters, 3, activation='relu', padding='same')(conv)
    return conv

def LRSN(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = adaptive_feature_selection(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = adaptive_feature_selection(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = adaptive_feature_selection(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Decoder
    up4 = local_relationship_upsampling(pool3, conv3, 256)
    up5 = local_relationship_upsampling(up4, conv2, 128)
    up6 = local_relationship_upsampling(up5, conv1, 64)

    # Output
    outputs = Conv2D(num_classes, 1, activation='softmax')(up6)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# # **Create model**

# In[552]:


#Parameters
input_shape=(H,W,CH)
num_classes=3

# EGDN (Pass)
#model=EGDN(input_shape, num_classes)

#TAGDN (Pass)
#model=TAGDN(input_shape, num_classes)

#BABNet (Pass)
#model=BABNet(input_shape, num_classes)

#SACCAN (Good)
#model = SACCAN(input_shape, num_classes)

#AFSN (Pass)
#model = AFSN(input_shape, num_classes)

#LRSN (Pass)
#model = LRSN(input_shape, num_classes)


# ## **Loss functions**

# In[553]:


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


# In[554]:


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

# In[555]:


#Model compile-------
model.compile( optimizer='adam',loss=[bce_jaccard_loss,dice_loss,'sparse_categorical_crossentropy'],
              metrics=['accuracy', jacard_coef, iou])
model.summary()


# In[530]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True)


# ## **Model Training**

# In[531]:


#Model Training
nepochs=1
nbatch_size=16
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




