#!/usr/bin/env python
# coding: utf-8

# ## **Multi-stage Feature-Fusion Dense Attentive Network (MFFDAN)-image resolution**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import random
import cv2
import os
import numpy as np
import pandas as pd
from numpy.random import randint
from tqdm.notebook import tqdm, tnrange,trange
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, Add, add, Activation,add,LayerNormalization,AvgPool2D
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization,Concatenate,SeparableConv2D, Lambda
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D,  Multiply
from tensorflow.keras.layers import Input, Reshape, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dropout, Lambda, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,Callback ,ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D,  MaxPool2D, UpSampling2D, concatenate, Activation
from tensorflow.keras.layers import Layer, Reshape, Conv2DTranspose, Multiply, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.losses import MeanSquaredError
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

K.clear_session()


# ## **Load Data and Visualization**

# In[ ]:


H,W,CH=[128,128,3]
good = 'ISRU/train/high/'
bad = 'ISRU/train/low/'


# In[ ]:


clean = []
for file in tqdm(sorted(os.listdir(good))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(good + '/' + file, target_size=(H,W),color_mode='rgb')
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        clean.append(image)

clean = np.array(clean)
blurry = []
for file in tqdm(sorted(os.listdir(bad))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(bad + '/' + file, target_size=(H,W),color_mode='rgb')
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        blurry.append(image)

blurry = np.array(blurry)


# In[ ]:


x = clean
y = blurry

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.shape)
print(y_train.shape)


# In[ ]:


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


# In[ ]:


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


# ## **ImageDataGenerator**

# In[ ]:


# Create an instance of the ImageDataGenerator for data augmentation
data_gen = ImageDataGenerator(
    rotation_range=10,   
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    zoom_range=0.1,  
    horizontal_flip=True    
)

data_gen.fit(x_train)
train_generator = data_gen.flow(x_train, y_train, batch_size=32)

# Visualize images
images, labels = train_generator.next()
num_images = len(images)
num_rows = 4
num_cols = num_images // num_rows

# Create a figure and subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 12))
if num_rows > 1:
    axes = axes.flatten()
    
for i, (image, label) in enumerate(zip(images, labels)):
    ax = axes[i]    
    ax.imshow(image,cmap='gray')
    ax.axis('off')    
    label_idx = label.argmax()
    ax.set_title(f"Label: {label_idx}")

plt.tight_layout()
plt.show()


# ## **Defining CNN Model**

# In[ ]:


def dense_block(input_layer, num_filters, num_layers):
    dense_output = input_layer
    for _ in range(num_layers):
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(dense_output)
        dense_output = Concatenate()([dense_output, x])
    return dense_output

def multi_stage_feature_fusion_dense_network(input_shape, num_filters=64, num_blocks=3, num_layers=3):
    input_layer = Input(shape=input_shape)
    
    # Initial convolutional layer
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(input_layer)
    
    # Dense blocks
    dense_outputs = []
    for _ in range(num_blocks):
        dense_output = dense_block(x, num_filters, num_layers)
        dense_outputs.append(dense_output)
        x = Concatenate()(dense_outputs)
    
    # Fusion layer
    fusion_layer = Conv2D(num_filters, kernel_size=1, padding='same', activation='relu')(x)
    
    # Upsampling layer
    output_layer = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(fusion_layer)
    output_layer = Conv2D(3, kernel_size=3, padding='same', activation='linear')(output_layer)  # Assuming RGB images
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model


# ## **Buildup Model**

# In[ ]:


input_shape = (H, W, CH) 
# Create the UNet model

model = multi_stage_feature_fusion_dense_network(input_shape)


# ## **Loss function**

# In[ ]:


# Perceptual loss
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(H, W, CH))
feature_extractor = tf.keras.Model(inputs=vgg_model.input,outputs=vgg_model.get_layer('block4_conv2').output)

def perceptual_loss(y_true, y_pred):
    target_features = feature_extractor(y_true)
    generated_features = feature_extractor(y_pred)    
    loss = tf.keras.losses.MeanSquaredError()(target_features, generated_features)    
    return loss

# Define loss function
def pixel_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

#PSNR
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303

#SSIM
def SSIM(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))


# ## **Model compilation and train**

# In[ ]:


K.clear_session()

model.compile(optimizer=Adam(learning_rate=1e-4),loss=[perceptual_loss,'mse'], metrics=[PSNR,'accuracy',SSIM])

model.summary()


# In[ ]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True,dpi=75)


# ## **Model Callback**

# In[ ]:


#Specify Callback
def show_image(image, title=None):
    image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)    
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

class ShowProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        id = np.random.randint(len(x_train))
        himage = x_train[id]
        limage = y_train[id]
        pred_img = self.model(tf.expand_dims(himage,axis=0))[0]
        
        plt.figure(figsize=(10,8))
        plt.subplot(1,3,1)
        show_image(himage, title="High Image")
        
        plt.subplot(1,3,2)
        show_image(limage, title="Low Image")
        
        plt.subplot(1,3,3)
        show_image(pred_img, title="Predicted Image")
            
        plt.tight_layout()
        plt.show()


# ## **Model Train**

# In[ ]:


nepochs=10
nbatch_size=6
cbs = [ShowProgress()]
SPE=len(x_train)//nbatch_size

history = model.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True, steps_per_epoch=SPE,callbacks=cbs,
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )
#train_generator


# In[ ]:


df_result = pd.DataFrame(history.history)
df_result.sort_values('val_loss', ascending=True, inplace = True)
df_result
#model.save_weights("UNetVGG19_b32_ep_50.h5")


# ## **Performance evaluation**

# In[ ]:


#Plot history loss
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(0, 101, 25))
plt.show()


# In[ ]:


#Plot history Accuracy
plt.figure(figsize=(12,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, 101, 25))
plt.show()


# ## **Model Evaluation**

# In[ ]:


print("\n\n Input --------------------\n Ground Truth\n-------------------------\n Predicted Value")
for i in tqdm(range(6)):    
    r = random.randint(0, len(clean)-1)
    x, y = blurry[r],clean[r]
    x_inp=x.reshape(1,H,W,CH)
    result = model.predict(x_inp)
    result = result.reshape(H,W,CH)

    fig = plt.figure(figsize=(12,10))
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(x)

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(y)
    
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(result)
    
plt.show()
print("--------------Done!----------------")


# ## **Model Evaluation**

# In[ ]:


# predict test images
predict_y = model.predict(x_test)

plt.figure(figsize=(15,15))
for i in  tqdm(range(0,9,3)):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i])
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


# ## **Model Evaluation**

# In[ ]:


#Display
def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

for i in tqdm(range(20)):
    idx = randint(len(x_test))
    image = x_test[idx]
    mask = y_test[idx]
    pred_mask = model.predict(tf.expand_dims(image,axis=0))[0]
        
    plt.figure(figsize=(10,8))
    plt.subplot(1,4,1)
    show_image(image, title="Higher Image")
        
    plt.subplot(1,4,2)
    show_image(mask, title="Lower Image")
        
    plt.subplot(1,4,3)
    show_image(pred_mask, title="Predicted Image")  
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()    



# ## **Prediction Visualization**

# In[ ]:


def PSNR(y_true,y_pred):
    mse=tf.reduce_mean( (y_true - y_pred) ** 2 )
    return 20 * log10(1/ (mse ** 0.5))

def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def pixel_MSE(y_true,y_pred):
    return tf.reduce_mean( (y_true - y_pred) ** 2 )

#Display

def plot_images(high,low,predicted):
    plt.figure(figsize=(12,9))
    plt.subplot(1,3,1)
    plt.title('High Image', color = 'green', fontsize = 10)
    plt.axis('off')
    plt.imshow(high)
    plt.subplot(1,3,2)
    plt.title('Low Image ', color = 'black', fontsize = 10)
    plt.axis('off')
    plt.imshow(low)
    plt.subplot(1,3,3)
    plt.title('Predicted Image ', color = 'Red', fontsize = 10)
    plt.imshow(predicted)
    plt.axis('off')
    
    plt.show()

for i in tqdm(range(16,25)):    
    predicted = np.clip(model.predict(x_test[i].reshape(1,H, W,CH)),0.0,1.0).reshape(H, W,CH)
    plot_images(x_test[i],y_test[i],predicted)
    print('PSNR',PSNR(x_test[i],predicted),'dB')
    


# ## **Benchmark test**

# In[ ]:


file="SRB/30.jpg"
img = tf.keras.preprocessing.image.load_img(file, target_size=(H,W),color_mode='rgb')
arr=np.array(img)
pred=model.predict(arr.reshape(1,H,W,CH))
pred = pred * 255.0
pred = np.clip(pred, 0, 255).astype(np.uint8)
cv2.cvtColor(np.array(pred), cv2.COLOR_RGB2BGR)

plt.figure(figsize=(9,6))
plt.subplot(1,2,1)
plt.title('Low Image', color = 'green', fontsize = 10)
plt.imshow(img)
plt.axis('off')
plt.subplot(1,2,2)
plt.title('Predicted Image ', color = 'black', fontsize = 10)
plt.axis('off')
plt.imshow(pred[0])
plt.show()


# ## **------------------------------------Done----------------------------------------**

# In[ ]:




