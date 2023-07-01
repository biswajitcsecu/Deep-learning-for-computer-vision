#!/usr/bin/env python
# coding: utf-8

# ## **Framework  Enhancement SAR Dense Unet**

# In[62]:


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

# In[109]:


#Data Directory
H,W,CH=[128,128,3]
high_dir = 'SAR/train/high/'
low_dir = 'SAR/train/low/'


# In[110]:


#Load Data------------------------------
High = []
for file in tqdm(sorted(os.listdir(high_dir))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(high_dir + file, target_size=(H,W,CH))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        High.append(image)

High = np.array(High)

Low = []
for file in tqdm(sorted(os.listdir(low_dir))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(low_dir +  file, target_size=(H,W,CH))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        Low.append(image)

Low = np.array(Low)


# In[117]:


x_train, x_test, y_train, y_test = train_test_split(High, Low, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
gc.collect()


# ## **Visualization the image and masks**

# In[118]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
for idx in tqdm(range(9)):
    plt.subplot(3,3,idx+1)
    plt.imshow(x_train[idx],cmap='gray')
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[119]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
for idx in tqdm(range(9)):
    plt.subplot(3,3,idx+1)
    plt.imshow(y_train[idx],cmap='gray')
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[120]:


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

# In[121]:


kernal_init = tf.keras.initializers.random_normal(stddev=0.008, seed = 101)      
regularizer = tf.keras.regularizers.L2(1e-4)
bias_init = tf.constant_initializer()


# In[122]:


def generic_model(input_shape):
    
    inputs = tf.keras.Input(shape = input_shape)                                          
    #layers    
    conv = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                  activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(inputs)
    conv = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init,
                  activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv)
    
    #layers                                   
    conv_up = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = kernal_init,
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv)
    conv_up = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = kernal_init,
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv_up)
                                    
    #layers                               
    conv1_1 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv_up)
    conv1_2 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv1_1)
    
    conv1_3 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', 
                     kernel_initializer = tf.keras.initializers.glorot_normal(seed = 101),
                     bias_initializer = bias_init, kernel_regularizer = regularizer)(conv1_2)
    
    conc1 = tf.add(conv1_3, conv1_1)
    conv1 = tf.keras.activations.relu(conc1)
    
    #layers

    conv2_1 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init,
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv1)
    conv2_2 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv2_1)
    
    conv2_3 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', 
                     kernel_initializer = tf.keras.initializers.glorot_normal(seed = 101),
                     bias_initializer = bias_init, kernel_regularizer = regularizer)(conv2_2)
    
    conc2 = tf.add(conv2_3, conv2_1)
    conv2 = tf.keras.activations.relu(conc2)
    
    #layers
    conv3_1 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init,
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv2)
    conv3_2 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv3_1)
    conv3_3 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv3_2)
    conv3_4 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv3_3)
    
    conv3_5 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', 
                     kernel_initializer = tf.keras.initializers.glorot_normal(seed = 101),
                     bias_initializer = bias_init, kernel_regularizer = regularizer)(conv3_4)
    
    conc3 = tf.add(conv3_5, conv3_1)
    conv3 = tf.keras.activations.relu(conc3)
    
    #layers
    conv4_1 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv3)
    conv4_2 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init,
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv4_1)
    conv4_3 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv4_2)
    conv4_4 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv4_3)
    
    conv4_5 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', 
                     kernel_initializer = tf.keras.initializers.glorot_normal(seed = 101),
                     bias_initializer = bias_init, kernel_regularizer = regularizer)(conv4_4)
    
    conc4 = tf.add(conv4_5, conv4_1)
    conv4 = tf.keras.activations.relu(conc4)
    
    #layers
    conv5_1 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv4)
    conv5_2 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv5_1)
    conv5_3 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv5_2)
    conv5_4 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv5_3)
    
    conv5_5 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same',
                     kernel_initializer = tf.keras.initializers.glorot_normal(seed = 101),
                     bias_initializer = bias_init, kernel_regularizer = regularizer)(conv5_4)
    
    conc5 = tf.add(conv5_5, conv5_1)
    conv5 = tf.keras.activations.relu(conc5)
    
    #layers
    conv6_1 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv5)
    conv6_2 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv6_1)
    conv6_3 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv6_2)
    conv6_4 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                     activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(conv6_3)
    
    conv6_5 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', 
                     kernel_initializer = tf.keras.initializers.glorot_normal(seed = 101),
                     bias_initializer = bias_init, kernel_regularizer = regularizer)(conv6_4)
    
    conc6 = tf.add(conv6_5, conv6_1)
    conv6 = tf.keras.activations.relu(conc6)

    #deconvolution                                        
    deconv = Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2, padding = 'same',
                             kernel_initializer = tf.keras.initializers.glorot_normal(seed = 101),
                             kernel_regularizer = regularizer)(conv5)
    deconv = Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2, padding = 'same', 
                             kernel_initializer = tf.keras.initializers.glorot_normal(seed = 101),
                             kernel_regularizer = regularizer)(deconv)

    conv = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = kernal_init, 
                  activation = 'relu',bias_initializer = bias_init, kernel_regularizer = regularizer)(deconv)
    
    conv = Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = 'same', 
                  kernel_initializer = tf.keras.initializers.glorot_normal(seed = 101),
                  bias_initializer = bias_init, kernel_regularizer = regularizer)(conv)
    conc = tf.add(conv, inputs)
    gman_net_output = tf.keras.activations.relu(conc)
    
     # Parallel Network 
    
    conv = Conv2D(filters = 64, kernel_size = 3, dilation_rate = 4, padding = 'same', kernel_initializer = kernal_init, 
                  activation = 'relu',kernel_regularizer = regularizer)(inputs)
    conv = Conv2D(filters = 64, kernel_size = 3, dilation_rate = 2, padding = 'same', kernel_initializer = kernal_init, 
                  activation = 'relu',kernel_regularizer = regularizer)(conv)
    conv = Conv2D(filters = 64, kernel_size = 3, dilation_rate = 2, padding = 'same', kernel_initializer = kernal_init, 
                  activation = 'relu',kernel_regularizer = regularizer)(conv)
    conv = Conv2D(filters = 64, kernel_size = 3, dilation_rate = 1, padding = 'same', kernel_initializer = kernal_init, 
                  activation = 'relu',kernel_regularizer = regularizer)(conv)
    conv = Conv2D(filters = 64, kernel_size = 3, dilation_rate = 1, padding = 'same', kernel_initializer = kernal_init, 
                  activation = 'relu',kernel_regularizer = regularizer)(conv)
    conv = Conv2D(filters = 64, kernel_size = 3, dilation_rate = 1, padding = 'same', kernel_initializer = kernal_init, 
                  activation = 'relu',kernel_regularizer = regularizer)(conv)
    
    deconv = Conv2DTranspose(filters = 64, kernel_size = 3, dilation_rate = 4, padding = 'same',
                             kernel_initializer = tf.keras.initializers.glorot_normal(seed = 101),
                           activation = 'relu', kernel_regularizer = regularizer)(conv)
    
    conv = Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = 'same', 
                  kernel_initializer = tf.keras.initializers.glorot_normal(seed = 101),
                 kernel_regularizer = regularizer)(deconv)
    
    conc = tf.add(conv, inputs)
    parallel_output = tf.keras.activations.relu(conc)
    
    output = tf.add(gman_net_output, parallel_output)
    mod = Model(inputs = inputs, outputs = output)
    

    # Create the model
    model = models.Model(inputs=inputs, outputs=output)

    return mod


# # **Model build**

# In[123]:


# Test the model
input_shape=[H,W,CH]
model = generic_model(input_shape)


# ## **Loss Functions**

# In[124]:


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

#CharbonnierLoss
class CharbonnierLoss(tf.keras.losses.Loss):
    def __init__(self, epsilon: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = tf.convert_to_tensor(epsilon)

    def call(self, y_true, y_pred):
        squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(tf.sqrt(squared_difference + tf.square(self.epsilon)))


# ## **Metrics**

# In[125]:


#PSNRMetric
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
        
#SSIM Metric
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

# In[126]:


#Model compile-------
Closs = CharbonnierLoss(epsilon=1e-3)
psnr = PSNRMetric(max_val=1.0)
ssim = SSIMMetric(max_val=1.0)    
estopping =  ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-07)    
model.compile(optimizer,loss=[perceptual_loss,Closs],metrics=['accuracy',psnr,ssim]) 
model.summary()


# In[127]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True)


# ## **Model Training**

# In[128]:


#Model Training
nepochs=1
nbatch_size=32
history = model.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True, callbacks=[estopping],
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# In[129]:


df_result = pd.DataFrame(history.history)
df_result


# ## **Performance evaluation**

# In[133]:


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
plt.plot(x,history.history['ssim_metric_5'])
plt.title('change in ssim coefitient over epochs')
plt.legend(['ssim_metric'])
plt.xlabel('epochs')
plt.ylabel('ssim_metric')
plt.show()
plt.tight_layout()


# ## **Model Evaluation**

# In[131]:


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

# In[132]:


# Creating predictions on our test set-----------------
predictions = model.predict(x_test)


# In[134]:


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


# In[135]:


#Show predicted result---------------
plot_results_for_one_sample(1)


# In[136]:


#Show predicted result---------------
plot_results_for_one_sample(6)


# In[137]:


#Show predicted result---------------
plot_results_for_one_sample(10)


# In[138]:


#Show predicted result---------------
plot_results_for_one_sample(14)


# In[139]:


#Show predicted result---------------
plot_results_for_one_sample(20)


# In[140]:


#Show predicted result---------------
plot_results_for_one_sample(25)


# In[141]:


#Show predicted result---------------
plot_results_for_one_sample(30)


# ## ---------------------------**End**----------------------------------- ##

# In[ ]:





# In[ ]:




