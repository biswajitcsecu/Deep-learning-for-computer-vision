#!/usr/bin/env python
# coding: utf-8

# ## **Image SR with GANs**

# In[3]:


import random
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from numpy import linalg as LA
from numpy.random import randint
from tqdm.notebook import tqdm, tnrange,trange
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keract
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.mobilenet import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, Activation, UpSampling2D, Concatenate
from tensorflow.keras.layers import Dense, Input, Add, add, Activation,add,LayerNormalization,AvgPool2D
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization,Concatenate,SeparableConv2D, Lambda
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D,  Multiply
from tensorflow.keras.layers import Input, Reshape, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dropout, Lambda, Flatten,LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,Callback ,ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D,  MaxPool2D, UpSampling2D, concatenate, Activation
from tensorflow.keras.layers import Layer, Reshape, Conv2DTranspose, Multiply, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.losses import MeanSquaredError
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

import warnings

warnings.filterwarnings('ignore')
K.clear_session()


# ## **Data load and splitting**

# In[4]:


batch_size = 8
good = 'CUFED/train/high/'
bad = 'CUFED/train/low/'
dataset_split =300


# In[5]:


clean = []
for file in tqdm(sorted(os.listdir(good)[0 : dataset_split])):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(good + '/' + file, target_size=(64,64),color_mode='rgb')
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        clean.append(image)
clean = np.array(clean)

blurry = []
for file in tqdm(sorted(os.listdir(bad)[0 : dataset_split])):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(bad + '/' + file, target_size=(128,128),color_mode='rgb')
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        blurry.append(image)

blurry = np.array(blurry)


# In[6]:


#Slice datasets
x = clean
y = blurry
train_x, test_x, train_y, test_y = train_test_split(np.array(x), np.array(y), test_size=0.1)

# Construct tf.data.Dataset object
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.batch(batch_size)

print(train_x.shape)
print(train_y.shape)


#  ## **Visualization the rgb image and gray**

# In[7]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(8,8))
indexes=[]
for i in tqdm(range(9)):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(train_x[index])
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[8]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(8,8))
indexes=[]
for i in tqdm(range(9)):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(train_y[index])
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# ## **GAN: Generator**

# In[22]:


def get_generator_model():    
    # Input low-resolution image
    input_shape = (64, 64, 3)
    inputs = Input(shape=input_shape)

    # First convolutional layer
    x = Conv2D(64, kernel_size=9, padding='same')(inputs)
    x = Activation('relu')(x)
    
    # Residual blocks
    for _ in tqdm(range(16)):
        x = residual_block(x)
    
    # Second convolutional layer
    x = Conv2D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([inputs, x])
    
    # Upsampling
    x = upsample(x, 2)
    
    # Third convolutional layer
    x = Conv2D(3, kernel_size=9, padding='same')(x)
    outputs = Activation('tanh')(x)
    
    # Create the model
    generator = Model(inputs=inputs, outputs=outputs)
    
    return generator


# ## **GAN:Discriminator**

# In[23]:


# Define a residual block
def residual_block(inputs):
    x = Conv2D(64, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([inputs, x])
    return x

# Define an upsampling block
def upsample(inputs, scale):
    x = Conv2D(256, kernel_size=3, padding='same')(inputs)
    x = UpSampling2D(size=scale)(x)
    x = Activation('relu')(x)
    return x

def get_discriminator_model():    
    input_shape = (128, 128, 3)
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(inputs)
    x = Activation('relu')(x)
    
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(1, kernel_size=1, strides=1, padding='same')(x)
    outputs = Activation('sigmoid')(x)
    
    discriminator = Model(inputs=inputs, outputs=outputs)

    
    return discriminator


# ### **Loss Functions**

# In[24]:


cross_entropy = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) - tf.random.uniform(shape=real_output.shape, maxval=0.1), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) + tf.random.uniform(shape=fake_output.shape, maxval=0.1), fake_output)
    total_loss = real_loss + fake_loss
    
    return total_loss

def generator_loss(fake_output, real_y):
    real_y = tf.cast(real_y, 'float32')
    
    return mse(fake_output, real_y)


# ## **Build model**

# In[25]:


generator_optimizer = tf.keras.optimizers.Adam(0.0005)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0005)


# In[26]:


generator = get_generator_model()
generator.summary()

tf.keras.utils.plot_model(generator, 'Model.png', show_shapes=True, dpi=75)


# In[27]:


discriminator = get_discriminator_model()
discriminator.summary()
tf.keras.utils.plot_model(discriminator, 'Model.png', show_shapes=True, dpi=75)


# ## **Training GAN**

# In[28]:


@tf.function
def train_step(input_x, real_y):    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(input_x, training=True)
        real_output = discriminator(real_y, training=True)
        generated_output = discriminator(generated_images, training=True)
        
        # L2 Loss 
        gen_loss = generator_loss( generated_images, real_y )
        disc_loss = discriminator_loss( real_output, generated_output )

    # Compute the gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Optimize with Adam
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    print("Loss HR, Loss LR, Loss GAN")
    print(gen_loss, disc_loss, loss_gan=(gen_loss+disc_loss))


# ## **Model Training**

# In[ ]:


#Train loop
num_epochs = 5

for idx in tqdm(range(num_epochs)):
    print('epochs:', idx)
    for (x_input, y_target) in dataset:        
        train_step(x_input, y_target)
        print( '<=----------------------------------!done!----------------------------------=>')


# ## **Model predictions**

# In[ ]:


#Results---------
y = generator(test_x)

for i in tqdm(range(len(test_x))):
    plt.figure(figsize=(8,8))
    or_image = plt.subplot(3,3,1)
    or_image.set_title('Grayscale Input', color='red', fontsize=10)
    plt.axis('off')
    plt.imshow(np.clip(test_x[i],0,1))
    
    in_image = plt.subplot(3,3,2)    
    image = Image.fromarray((y[i] * 255).astype('uint8'))
    image = np.asarray(np.clip(y[i],0,1))
    in_image.set_title('Colorized Output',  color='green', fontsize=10)
    plt.axis('off')
    plt.imshow(image)
    
    out_image = plt.subplot(3,3,3)
    image = Image.fromarray((test_y[i] * 255).astype('uint8'))
    plt.axis('off')
    out_image.set_title('Ground Truth',  color='blue', fontsize=10)
    plt.imshow( image )    
    
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()  
    


# ## **Model predictions**

# In[ ]:


# Creating predictions on our test set-----------------

predictions = generator(test_x)


# In[ ]:


# Ploting results for one image----------------

def plot_results_for_one_sample(sample_index):    
    pdimg =predictions[sample_index] 
    fig = plt.figure(figsize=(10,10))
    # Gray image-------------------
    fig.add_subplot(1,3,1)
    plt.title('Gray image')
    org=test_x[sample_index]
    np.clip((org),0,1 ).astype('uint8')
    plt.imshow(org, cmap='gray')
    plt.axis('off')
    plt.grid(None)
    
    #RGB image----------
    fig.add_subplot(1,3,2)
    plt.title('RGB image')
    rgborg= test_y[sample_index]
    np.clip((rgborg),0,1 ).astype('uint8')
    plt.imshow(rgborg)
    plt.axis('off')
    plt.grid(None)
    
    #Predicted image------------
    fig.add_subplot(1,3,3)
    plt.title('Predicted image')  
    plt.imshow(np.clip(pdimg,0,1))
    plt.axis('off')
    plt.grid(None)

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show() 


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(1)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(5)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(10)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(15)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(20)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(25)


# ## **Model predictions**

# In[ ]:


# Generate deblurred images from test set
test_images = generator.predict(test_x)

# Plotting example results
n = 10
plt.figure(figsize=(25, 5))
for i in tqdm(range(n)):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(np.clip(test_x[i],0,1))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, i+n+1)
    plt.imshow(np.clip(test_images[i],0,1))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()     


# In[ ]:





# In[ ]:




