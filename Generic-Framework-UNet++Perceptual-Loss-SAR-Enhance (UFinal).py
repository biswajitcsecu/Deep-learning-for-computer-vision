#!/usr/bin/env python
# coding: utf-8

# ## **Framework  Enhancement SAR Dense Unet**

# In[142]:


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
from tensorflow.keras import layers, regularizers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Input, Add,  Activation,add,LayerNormalization
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization,Concatenate,SeparableConv2D
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Dropout, Multiply
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, concatenate, UpSampling3D
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, Conv2DTranspose
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import MobileNetV2
from keras.applications.densenet import DenseNet201, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import mean_squared_error
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanIoU

import segmentation_models as sm
from segmentation_models.metrics import iou_score

import warnings

K.clear_session()
warnings.filterwarnings("ignore")


# ## **Data Reading and Train test split**

# In[143]:


#Data Directory
H,W,CH=[128,128,3]
high_dir = 'SAR/train/high/'
low_dir = 'SAR/train/low/'


# In[144]:


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


# In[146]:


x_train, x_test, y_train, y_test = train_test_split(High, Low, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
gc.collect()


# ## **Visualization the image and masks**

# In[147]:


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


# In[148]:


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


# In[149]:


#Display test data
for i in tqdm(range(4)):
    idx = np.random.randint(0,80)
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.title('High Image', color = 'black', fontsize = 12)
    plt.imshow(x_train[idx],cmap='gray')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title('Low Image', color = 'black', fontsize = 12)
    plt.imshow(y_train[idx],cmap='gray')
    plt.axis('off')


# ## **Building  Network architecture**

# In[207]:


def conv_block(inputs, filters, dropout_rate):
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv = Dropout(dropout_rate)(conv)
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv)
    return conv

def build_unet(input_shape, filters, dropout_rate):
    inputs = Input(input_shape)

    # Downward path
    conv1 = conv_block(inputs, filters, dropout_rate)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = conv_block(pool1, filters * 2, dropout_rate)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = conv_block(pool2, filters * 4, dropout_rate)
    pool3 = MaxPooling2D((2, 2))(conv3)
    conv4 = conv_block(pool3, filters * 8, dropout_rate)
    pool4 = MaxPooling2D((2, 2))(conv4)

    # Bottom path
    conv5 = conv_block(pool4, filters * 16, dropout_rate)

    # Upward path
    up6 = Conv2DTranspose(filters * 8, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = conv_block(up6, filters * 8, dropout_rate)
    up7 = Conv2DTranspose(filters * 4, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = conv_block(up7, filters * 4, dropout_rate)
    up8 = Conv2DTranspose(filters * 2, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = conv_block(up8, filters * 2, dropout_rate)
    up9 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1])
    conv9 = conv_block(up9, filters, dropout_rate)

    # Output
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# # **Model build**

# In[208]:


# Test the model
input_shape=[H,W,CH]
filters = 64  # Number of filters in the first layer
dropout_rate = 0.5  # Dropout rate for regularization

# Create the model
model = build_unet(input_shape, filters, dropout_rate)
    


# ## **Loss function & Metric**

# In[209]:


# Load pre-trained VGG16 model
vgg16 = VGG16(weights='imagenet', include_top=False)

# Specify the layers from which we will extract features
content_layers = ['block3_conv3']
style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']

# Create a new model that outputs feature maps from the specified layers
content_outputs = [vgg16.get_layer(layer).output for layer in content_layers]
style_outputs = [vgg16.get_layer(layer).output for layer in style_layers]
feature_extractor = Model(inputs=vgg16.input, outputs=content_outputs + style_outputs)

# Define perceptual loss function
def perceptual_loss(y_true, y_pred):
    # Extract features from target and generated outputs
    target_features = feature_extractor(y_true)
    generated_features = feature_extractor(y_pred)

    # Compute mean squared error between target and generated features
    loss = 0
    for target_feature, generated_feature in zip(target_features, generated_features):
        loss += K.mean(mean_squared_error(target_feature, generated_feature))

    return loss

class CharbonnierLoss(tf.keras.losses.Loss):
    def __init__(self, epsilon: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = tf.convert_to_tensor(epsilon)

    def call(self, y_true, y_pred):
        squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(
            tf.sqrt(squared_difference + tf.square(self.epsilon))
        )


# In[210]:


#metrics
class PSNRMetric(tf.keras.metrics.Metric):
    def __init__(self, max_val: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_val = max_val
        self.psnr = tf.keras.metrics.Mean(name="psnr")

    def update_state(self, y_true, y_pred, *args, **kwargs):
        psnr = tf.image.psnr(y_true, y_pred, max_val=self.max_val)
        self.psnr.update_state(psnr, *args, **kwargs)

    def result(self):
        return self.psnr.result()

    def reset_state(self):
        self.psnr.reset_state()


class SSIMMetric(tf.keras.metrics.Metric):
    def __init__(self, max_val: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_val = max_val
        self.ssim = tf.keras.metrics.Mean(name="ssim")

    def update_state(self, y_true, y_pred, *args, **kwargs):
        ssim = tf.image.ssim(y_true, y_pred, max_val=self.max_val)
        self.ssim.update_state(ssim, *args, **kwargs)

    def result(self):
        return self.ssim.result()

    def reset_state(self):
        self.ssim.reset_state()        
    


# ## **Model compilation**

# In[211]:


#Model compile-------
Closs = CharbonnierLoss(epsilon=1e-3)
psnr = PSNRMetric(max_val=1.0)
ssim = SSIMMetric(max_val=1.0)    
estopping =  ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)    
model.compile(optimizer,loss=[perceptual_loss,Closs],metrics=['accuracy',psnr,ssim]) 
model.summary()


# In[212]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True)


# ## **Model Training**

# In[226]:


#Model Training
nepochs=2
nbatch_size=32
history = model.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True, callbacks=[estopping],
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# In[229]:


df_result = pd.DataFrame(history.history)
df_result


# ## **Performance evaluation**

# In[230]:


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
plt.plot(x,history.history['ssim_metric_6'])
plt.title('change in ssim coefitient over epochs')
plt.legend(['ssim_metric'])
plt.xlabel('epochs')
plt.ylabel('ssim_metric')
plt.show()
plt.tight_layout()


# ## **Model Evaluation**

# In[231]:


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

# In[232]:


# Creating predictions on our test set-----------------
predictions = model.predict(x_test)


# In[233]:


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


# In[234]:


#Show predicted result---------------
plot_results_for_one_sample(1)


# In[235]:


#Show predicted result---------------
plot_results_for_one_sample(6)


# In[236]:


#Show predicted result---------------
plot_results_for_one_sample(10)


# In[237]:


#Show predicted result---------------
plot_results_for_one_sample(14)


# In[238]:


#Show predicted result---------------
plot_results_for_one_sample(20)


# In[239]:


#Show predicted result---------------
plot_results_for_one_sample(25)


# In[240]:


#Show predicted result---------------
plot_results_for_one_sample(30)


# ## ---------------------------**End**----------------------------------- ##

# In[ ]:




