#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Import Required Packages
import numpy as np
import pandas as pd
import imageio
import random
import os
import cv2
from glob import glob
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Input, Conv2D, MaxPooling2D,Activation, Add, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, concatenate, Multiply, Concatenate
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import warnings


K.clear_session()
warnings.filterwarnings("ignore")


# ## **Data Preparation**

# In[2]:


#path of images and mask directories
image_path='Kvasir/train/images/'
mask_path='Kvasir/train/masks/'

#load paths of images and masks files
image_paths=[i for i in glob(image_path+'*.jpg')]
mask_paths =[i for i in glob(mask_path+'*.jpg')]

#Create lists of image and mask paths
number_of_images, number_of_masks = len(image_paths), len(mask_paths)

print(f"1. There are {number_of_images} images and {number_of_masks} masks in our dataset")
print(f"2. An example of an image path is: \n {image_paths[0]}")
print(f"3. An example of a mask path is: \n {mask_paths[0]}")


# ## **Visulization of  masks and images**

# In[3]:


#display image and  mask
number_of_samples = len(image_paths)

for i in tqdm(range(3)):
    N = random.randint(0, number_of_samples - 1)

    img = imageio.imread(image_paths[N])
    mask = imageio.imread(mask_paths[N])
    mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0], img.shape[1])
    fig, arr = plt.subplots(1, 3, figsize=(20, 8))
    arr[0].imshow(img)
    arr[0].set_title('Image')
    arr[0].axis("off")
    arr[1].imshow(mask)
    arr[1].set_title('Segmentation')
    arr[1].axis("off")    
    arr[2].imshow(mask, cmap='Paired')
    arr[2].set_title('Image Overlay')
    arr[2].axis("off")
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()    


# ## **Split the images and masks into training, validation, and test sets**

# In[6]:


# First split the image paths into training and validation sets
train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths,
                                                                                        train_size=0.8, random_state=0)

# Keep part of the validation set as test set
validation_image_paths, test_image_paths, validation_mask_paths, test_mask_paths = train_test_split(val_image_paths, 
                           val_mask_paths, train_size = 0.80, random_state=0)

print(f'There are {len(train_image_paths)} images in the Training Set')
print(f'There are {len(validation_image_paths)} images in the Validation Set')
print(f'There are {len(test_image_paths)} images in the Test Set')
print(f'-------------------done-------------------------------')


# ## **Create a data pipeline**

# In[7]:


#Read imagand mask
def read_image(image_path, mask_path):    
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    #image = tf.image.convert_image_dtype(image, tf.float32)/255.
    image = tf.image.resize(image, (128, 128), method='nearest')

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.convert_image_dtype(mask,  tf.float32)
    mask = tf.image.resize(mask, (128, 128), method='nearest')
    
    return image, mask


# In[8]:


#Create a data generator function to read and load images and masks in batches

def data_generator(image_paths, mask_paths, buffer_size, batch_size):    
    image_list = tf.constant(image_paths) 
    mask_list = tf.constant(mask_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(buffer_size).batch(batch_size)
    
    return dataset


# In[9]:


#Create data pipelines for the training, validation and test sets using both functions
nbatch_size = 16
buffer_size = 500

train_dataset = data_generator(train_image_paths, train_mask_paths, buffer_size, nbatch_size)
validation_dataset = data_generator(validation_image_paths, validation_mask_paths, buffer_size, nbatch_size)
test_dataset = data_generator(test_image_paths, test_mask_paths, buffer_size, nbatch_size)


# ## **Model Architecture**

# In[10]:


def attention_module(input_feature, reduction_ratio=16):
    channel = input_feature.shape[-1]
    shared_layer_one = Conv2D(channel // reduction_ratio, (1, 1), padding='same', activation='relu')(input_feature)
    shared_layer_two = Conv2D(channel, (1, 1), padding='same')(shared_layer_one)

    attention = Activation('sigmoid')(shared_layer_two)
    scaled_feature = Multiply()([input_feature, attention])

    return scaled_feature


def AGCFFNet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
    conv4 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv5)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv6)

    conv7 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool3)
    conv8 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv7)

    # Attention-guided Contextual Feature Fusion
    attention1 = attention_module(conv8)
    up1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(attention1)

    concat1 = Concatenate()([up1, conv6])
    attention2 = attention_module(concat1)
    up2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(attention2)

    concat2 = Concatenate()([up2, conv4])
    attention3 = attention_module(concat2)
    up3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(attention3)

    concat3 = Concatenate()([up3, conv2])
    attention4 = attention_module(concat3)

    # Decoder
    up4 = UpSampling2D(size=(1, 1), interpolation='bilinear')(attention4)
    conv9 = Conv2D(64, (3, 3), padding='same', activation='relu')(up4)
    conv10 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv9)

    output = Conv2D(4, (1, 1), activation='sigmoid')(conv10)

    model = Model(inputs=inputs, outputs=output)

    return model


# ## **Buildup Model**

# In[11]:


H, W,CH = [128,128,3]
input_shape=(H, W, CH)
model = AGCFFNet(input_shape)


# ## **Loss functions**

# In[ ]:


# Define the perception loss
def perception_loss(y_true, y_pred):
    return MeanSquaredError()(tf.image.sobel_edges(y_true), tf.image.sobel_edges(y_pred))

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


# ## **Metrics**

# In[ ]:


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


# ## **Model Compilation**

# In[ ]:


opt = keras.optimizers.Adam(learning_rate=0.0001)
loss1=[perception_loss,'sparse_categorical_crossentropy',dice_loss]
metric1 = ['accuracy',iou,jacard_coef]
model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#binary_crossentropy,sparse_categorical_crossentropy,categorical_crossentropy
model.summary()


# In[13]:


tf.keras.utils.plot_model(model, 'model.png', show_shapes=True)


# ## **Train Model**

# In[16]:


#train parameters
callback = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=1e-1, patience=5, verbose=1, min_lr = 2e-6)
nepochs = 1
nbatch_size = 4


# In[17]:


#train
history = model.fit(train_dataset,validation_data = validation_dataset, 
                    epochs = nepochs,verbose=1,callbacks = [callback, reduce_lr], 
                    batch_size = nbatch_size,shuffle = True,
                    max_queue_size=10,workers=1,use_multiprocessing=True,
                   )


# # **Model performances**

# In[18]:


#Model performances
acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# ## **Compute Model Accuracy**

# In[19]:


train_loss, train_accuracy = model.evaluate(train_dataset, batch_size = 32)
validation_loss, validation_accuracy = model.evaluate(validation_dataset, batch_size = 32)
test_loss, test_accuracy = model.evaluate(test_dataset, batch_size = 32)


# In[20]:


print(f'Model Accuracy on the Training Dataset: {round(train_accuracy * 100, 2)}%')
print(f'Model Accuracy on the Validation Dataset: {round(validation_accuracy * 100, 2)}%')
print(f'Model Accuracy on the Test Dataset: {round(test_accuracy * 100, 2)}%')


# ## **Model Evaluation**

# In[21]:


#Evaluate Predicted Segmentations
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)    
    return pred_mask[0]


# In[22]:


#display: an input image, its true mask, and its predicted mask

def display(display_list):
    plt.figure(figsize=(16, 16))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# In[30]:


#show predictions results

def show_predictions(dataset, num):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,create_mask(model.predict(sample_image[tf.newaxis, ...]))], cmap='Paired')


# In[31]:


#Predict and compare masks of images in the training set

show_predictions(train_dataset, 8)


# In[35]:


#Predict and compare masks of images in the validation set

show_predictions(validation_dataset,9)


# In[36]:


#Predict and compare masks of images in the test set

show_predictions(test_dataset, 26)


# ## **Model evaluation**

# In[62]:


print("Input ----------------------------Ground Truth-------------------------------------Predicted Value")

for idx in tqdm(range(6)): 
    for image, mask in test_dataset.take(idx):
        pred_mask = model.predict(image)
        x,y =image[0], mask[0],
        pred_mask = model.predict(image)
        result =  create_mask(pred_mask)        
        #Plot result
        fig = plt.figure(figsize=(12,10))
        fig.subplots_adjust(hspace=0.1, wspace=0.2)
        #Image
        ax = fig.add_subplot(1, 3, 1)
        plt.axis("off")
        ax.imshow(x, cmap='Paired')
        #Mask
        ax = fig.add_subplot(1, 3, 2)
        plt.axis("off")
        ax.imshow(y, cmap='Paired')
        #result
        ax = fig.add_subplot(1, 3, 3)
        plt.axis("off")
        plt.imshow(result, cmap='Paired') 
        plt.grid('off')    
        plt.show()
        
print("-------------------------------------------------------Done!---------------------------------------------------------")


# In[ ]:





# In[ ]:




