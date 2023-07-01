#!/usr/bin/env python
# coding: utf-8

# In[200]:


#Library-------------
import os
import cv2
import random
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import seaborn as sns
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
import glob
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import plot_model
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate

import warnings
warnings.filterwarnings('ignore')


# # Load Dataset
# 

# In[118]:


# data
H,W,CH=[128,128,3]
image_path = "Polyp/train/images/"
mask_path = "Polyp/train/masks/"
# di sort
list_image = np.sort(next(os.walk(image_path), (None, None, []))[2])
list_mask = np.sort(next(os.walk(mask_path), (None, None, []))[2])
len(list_image), len(list_mask)


# In[120]:


train_images_dir = sorted(glob.glob(image_path+'/*.png'))
train_masks_dir = sorted(glob.glob(mask_path+'/*.png'))


# # Read Images and Annotations

# In[121]:


def colored_segmentation_image(seg, colors, n_classes):
    seg_img = np.zeros_like(seg)
    
    for c in range(n_classes):
        seg_img[:, :, 0] += ((seg[:, :, 0] == c)* (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg[:, :, 0] == c)* (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg[:, :, 0] == c)* (colors[c][2])).astype('uint8')
        return seg_img    

  


# In[122]:


# initiate fix colors list
class_colors = [(128,0,0), (170,110,40), (128,128,0), (0,128,128),(0,0,128), (230,25,75), (245,130,48), (255,255,25),
              (210,245,60), (240,50,230), (128,128,128), (220,190,255),(255,215,180), (70,140,240), (0,130,200)]


# In[124]:


# membuat list of images
train_images = []
for img in train_images_dir:
    n = cv2.imread(img)
    train_images.append(n)
     

train_masks = []
for img in train_masks_dir:
    n = cv2.imread(img)
    train_masks.append(n)   


# # Plot Sample (Train Images)

# In[125]:


fig = plt.figure(figsize = (20,12))

for index in range(8):
    ax = fig.add_subplot(2,4,index+1)
    ax.set_title("Sample train image {}".format(index+1))
    ax.imshow(train_images[index], cmap='gray')   
    


# # Plot sample (annotations images)

# In[126]:


#Plot sample of train annotations
fig = plt.figure(figsize = (20,12))

for index in range(8):
    ax = fig.add_subplot(2,4,index+1)
    ax.set_title("Sample train annotations {}".format(index+1))
    ax.imshow(colored_segmentation_image(train_masks[index],n_classes=1, colors=class_colors))     


# In[201]:


#image and mask list
train_images = []
for img in train_images_dir:
    n = cv2.imread(img)
    n = cv2.resize(n, (H,W))
    n = asarray(n)
    n = n.astype('float32')
    n /= 255.0
    train_images.append(n)
    
train_masks = []
for img in train_masks_dir:
    n = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    n = cv2.resize(n, (H,W))
    n = asarray(n)
    n = n.astype('float32')
    n /= 255.0
    
    train_masks.append(n)
  


# In[202]:


print("There are {} images in train images".format(len(os.listdir(image_path))))
print("\nThere are {} images in train annotations".format(len(os.listdir(mask_path))))


# In[203]:


print("shape of one sample image in train images dataset: {}".format(train_images[0].shape))
print("\nshape of one sample image in train annotations dataset: {}".format(train_masks[0].shape))


# # Split Dataset

# In[204]:


train_images = np.array(train_images)
train_masks = np.array(train_masks) 
train_masks = np.expand_dims(train_masks, axis=-1)

x_train, x_test, y_train, y_test = train_test_split(train_images,train_masks,test_size = 0.2,random_state = 48)


# In[205]:


x_train = np.array(x_train)
x_test = np.array(x_test)    

y_train = np.array(y_train).squeeze()
y_test = np.array(y_test)

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("x_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))
     


# # Using Pretrained Model

# In[206]:


from keras.applications.vgg16 import VGG16  
from keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel
from keras.applications.inception_v3 import InceptionV3 as PTModel
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.models import Model
from keras.layers import BatchNormalization


# In[207]:


height = x_train.shape[1]
width = x_train.shape[2]
channel = x_train.shape[3]


def UNET():
    inputs = tf.keras.Input((width, height, channel))
    bn = BatchNormalization()(inputs)
    bn = Dropout(0.5)(bn)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(bn)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])


# In[208]:


model1 = UNET()
model1.summary()


# # U-Net Model

# In[209]:


def CNNModel():
    
    inputs = tf.keras.Input((width, height, channel))
    bn = BatchNormalization()(inputs)
    bn = Dropout(0.5)(bn)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, (1,1, activation = 'softmax')(conv9)
    
    
    model = Model(inputs=[inputs], outputs=conv10)   
    
    return model


# # ResUNet

# In[232]:


#ResNet model

k_size = 3

def bn_act(x, act=True):
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.Activation('relu')(x)
    return x

def conv_block(x, filters, kernel_size=3, padding='same', strides=1):
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=3, padding='same', strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size, padding, strides)
    shortcut = Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=3, padding='same', strides=1):
    res = conv_block(x, filters, k_size, padding, strides)
    res = conv_block(res, filters, k_size, padding, 1)
    shortcut = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = UpSampling2D((2,2))(x)
    c = Concatenate()([u, xskip])
    return c

def ResUNet():    
    #layer rank
    f = [16, 32, 64, 128, 256]
    
    ## Encoder
    e0 = Input((width, height, channel))
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    
    model = tf.keras.models.Model(inputs, outputs)
    
    return model


# # Model build

# In[223]:


#Model instance
#model= CNNModel();
#model=  UNET()
model= ResUNet()


model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()


# In[224]:


dot_img_file = 'model_unet1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


# # Callbacks

# In[225]:


callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('modelMUnet.h5', verbose=1, save_best_only=True, save_weights_only=True),
    CSVLogger("dataResUnet.csv")
]


# # Train model

# In[226]:


nbatch_size=16
nepochs=1
results = model.fit(x_train, y_train, batch_size=nbatch_size, epochs=nepochs, 
                    callbacks=callbacks,  max_queue_size=32,
                    validation_data=(x_test, y_test), workers=4,
                    verbose=1, use_multiprocessing=True
                   )  


# # Performence plot

# In[227]:


#Performence plot
acc = results.history['accuracy']
val_acc = results.history['val_accuracy']

loss = results.history['loss']
val_loss = results.history['val_loss']

epochs = range(len(acc))

fig = plt.figure(figsize = (15,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot(epochs, acc, label='Training Accuracy')
ax1.plot(epochs, val_acc, label='Validation Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.legend(loc=0)

ax2.plot(epochs, loss, label='Training Loss')
ax2.plot(epochs, val_loss, label='Validation Loss')
ax2.set_title('Training and Validation Loss')
ax2.legend(loc=0)


# In[228]:


#Load model
model.save('modelMUnet.h5') 
load = tf.keras.models.load_model('modelMUnet.h5')     


# # Evaluation

# In[229]:


y_pred = load.predict(x_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_pred_argmax.shape


# In[230]:


metrics = load.evaluate(x_test, y_pred)
print("{}: {}".format(load.metrics_names[0], metrics[0]))
print("{}: {}".format(load.metrics_names[1], metrics[1]))
     


#  # Plot Prediction Image

# In[231]:


num = random.randint(0,len(x_test))

plt.figure(figsize=(20,20))

plt.subplot(231)
plt.title('Original Test Image', fontsize=20, pad=20)
plt.imshow(x_test[num])

plt.subplot(232)
plt.title('Label Test Image', fontsize=20, pad=20)
plt.imshow(y_test[num].squeeze(), cmap='jet')

plt.subplot(233)
plt.title('Prediction Test Image', fontsize=20, pad=20)
plt.imshow(y_pred_argmax[num], cmap='jet')

plt.show()
     


# In[ ]:





# In[ ]:




