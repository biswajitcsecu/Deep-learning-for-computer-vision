#!/usr/bin/env python
# coding: utf-8

# In[47]:


#Import Required Packages
import cv2
import os
import random
import glob
import pandas as pd
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, tnrange
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet201, DenseNet121
from tensorflow.keras import models
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, Add,  Activation,add,LayerNormalization,AvgPool2D
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization,Concatenate,SeparableConv2D
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D,  Multiply
from tensorflow.keras.layers import Input, Reshape, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,Callback ,ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D,  MaxPool2D, UpSampling2D, concatenate, Activation
from tensorflow.keras.layers import Layer, Input, Add, Multiply, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import backend as K

random.seed(23)

import warnings

K.clear_session()
warnings.filterwarnings("ignore")


# ## **Data Loading**

# In[48]:


#Getting the image path 
image_path = "UltraSound/train/images/*.png"
mask_path = "UltraSound/train/masks/*.png"

## size images 
H,W,CH=(128,128,3)

image_names = sorted(glob.glob(image_path), key=lambda x: x.split('.')[0])
mask_names = sorted(glob.glob(mask_path), key=lambda x: x.split('.')[0])


# ## **Data Reading, Train and Test split**

# In[49]:


#appending then into the list 
images =[]
masks = []

for image in image_names:
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (H, W), interpolation = cv2.INTER_AREA)
    images.append(img)
    
images = np.array(images)/255.

for mask in mask_names:
    msk = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    msk = cv2.resize(msk, (H, W), interpolation=cv2.INTER_NEAREST)
    masks.append(msk)
    
masks = np.array(masks)/255.


# In[50]:


## splitting the image into train and test 
X=images
Y=masks

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


# ## **Train, Test split size**

# In[51]:


x_train = np.reshape(x_train,(len(x_train),H,W,3))
y_train = np.reshape(y_train,(len(y_train),H,W,1))
print("Shape of training images:", x_train.shape, y_train.shape)

x_test = np.reshape(x_test,(len(x_test),H,W,3))
y_test = np.reshape(y_test,(len(y_test),H,W,1))
print("Shape of test images:",x_test.shape,y_test.shape)


# ## **Data Visualization**

# In[52]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(10,10))
idx=[]
for i in tqdm(range(9)):
    idx=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[idx])
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[53]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(10,10))
idx=[]
for i in tqdm(range(9)):
    idx=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(y_train[idx])
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# ## **Visualization the image and  masks**

# In[54]:


#show image
def show_image(image, title=None, cmap='Paired', alpha=1):
    plt.imshow(image, cmap=cmap, alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    
plt.figure(figsize=(8,15))
for i in range(8):
    plt.subplot(4,2,i+1)
    if (i+1)%2!=0:
        idx = np.random.randint(len(x_train))
        image = x_train[idx]
        mask = y_train[idx]
        show_image(image)
    elif (i+1)%2==0:
        show_image(mask)
        


# ## **Augmentation**

# In[55]:


def data_generator(x_train, y_train, batch_size):
    data_gen_args = dict(width_shift_range = 0.1,
            height_shift_range = 0.1,
            rotation_range = 10,
            zoom_range = 0.1)
    
    image_generator = ImageDataGenerator(**data_gen_args).flow(x_train, x_train, batch_size, seed = 42)
    mask_generator = ImageDataGenerator(**data_gen_args).flow(y_train, y_train, batch_size, seed = 42)
    while True:
        x_batch, _ = image_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch
        
def augmentedImageChecker(x_train, y_train, seed):
    raw_batch, mask_batch = next(data_generator(x_train, y_train, seed))
    fix, ax = plt.subplots(5,2, figsize=(8,20))
    for i in range(5):
        ax[i,0].set_title('Train Image')
        ax[i,0].axis('off') 
        ax[i,0].imshow(raw_batch[i,:,:,0])
        ax[i,1].set_title('Train Mask')
        ax[i,1].imshow(mask_batch[i,:,:,0],cmap='gray')
        ax[i,1].axis('off')    
    plt.show()    
    


# In[56]:


augmentedImageChecker(x_train, y_train, 5)


# ## **CNN Model**

# In[57]:


#DenseUnet

def DenseBlock(channels,inputs):
    conv1_1 = Conv2D(channels, (1, 1),activation=None, padding='same')(inputs)
    conv1_1=BatchActivate(conv1_1)
    conv1_2 = Conv2D(channels//4, (3, 3), activation=None, padding='same')(conv1_1)
    conv1_2 = BatchActivate(conv1_2)

    conv2=concatenate([inputs,conv1_2])
    conv2_1 = Conv2D(channels, (1, 1), activation=None, padding='same')(conv2)
    conv2_1 = BatchActivate(conv2_1)
    conv2_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same')(conv2_1)
    conv2_2 = BatchActivate(conv2_2)

    conv3 = concatenate([inputs, conv1_2,conv2_2])
    conv3_1 = Conv2D(channels, (1, 1), activation=None, padding='same')(conv3)
    conv3_1 = BatchActivate(conv3_1)
    conv3_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same')(conv3_1)
    conv3_2 = BatchActivate(conv3_2)

    conv4 = concatenate([inputs, conv1_2, conv2_2,conv3_2])
    conv4_1 = Conv2D(channels, (1, 1), activation=None, padding='same')(conv4)
    conv4_1 = BatchActivate(conv4_1)
    conv4_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same')(conv4_1)
    conv4_2 = BatchActivate(conv4_2)
    result=concatenate([inputs,conv1_2, conv2_2,conv3_2,conv4_2])
    return result

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def DenseUNet(input_shape):
    filters=16
    keep_prob=0.9
    block_size=8

    inputs = Input(shape=input_shape)

    conv1 = Conv2D(filters * 1, (3, 3), activation=None, padding="same")(inputs)
    conv1 = BatchActivate(conv1)
    conv1 = DenseBlock(filters * 1, conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = DenseBlock(filters * 2, pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = DenseBlock(filters * 4, pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    convm = DenseBlock(filters * 8, pool3)

    deconv3 = Conv2DTranspose(filters * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(filters * 4, (1, 1), activation=None, padding="same")(uconv3)
    uconv3 = BatchActivate(uconv3)
    uconv3 = DenseBlock(filters * 4, uconv3)


    deconv2 = Conv2DTranspose(filters * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(filters * 2, (1, 1), activation=None, padding="same")(uconv2)
    uconv2 = BatchActivate(uconv2)
    uconv2 = DenseBlock(filters * 2, uconv2)

    deconv1 = Conv2DTranspose(filters * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(filters * 1, (1, 1), activation=None, padding="same")(uconv1)
    uconv1 = BatchActivate(uconv1)
    uconv1 = DenseBlock(filters * 1, uconv1)

    outputs = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    outputs = Activation('sigmoid')(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    
    return model     


# ## **Create model**

# In[58]:


K.clear_session()
input_shape = (H, W, CH)
model = DenseUNet(input_shape)


# ## **Loss functions**

# In[59]:


#dice_loss
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

#iou metric
smooth =100
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

#jacard_coef
def jaccard_coef(y_true,y_pred,smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f*y_pred_f)
    return (intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)-intersection+smooth)

def jaccard_loss(y_true,y_pred,smooth=1):
    return -jaccard_coef(y_true,y_pred,smooth)


# ## **Model compilation**

# In[60]:


#Model compile-------
metrics = ["accuracy",jaccard_loss, iou,
           tf.keras.metrics.AUC(), 
           tf.keras.metrics.SensitivityAtSpecificity(0.5), 
           tf.keras.metrics.SpecificityAtSensitivity(0.5)]

model.compile(optimizer=Adam(learning_rate=1e-4),loss=['binary_crossentropy', dice_loss],metrics=metrics) 
#model.compile( optimizer='adam',loss=['dice_loss,binary_crossentropy'],metrics=['accuracy']) 
#'sparse_categorical_crossentropy'=class>2
#'binary_crossentropy' class=2
model.summary()


# In[61]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True,dpi=75)


# ## **Callback**

# In[62]:


#Specify Callback
def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

class ShowProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        id = np.random.randint(len(x_train))
        image = x_train[id]
        mask = y_train[id]
        pred_mask = self.model(tf.expand_dims(image,axis=0))[0]
        
        plt.figure(figsize=(8,6))
        plt.subplot(1,3,1)
        show_image(image, title="Image")
        
        plt.subplot(1,3,2)
        show_image(mask, title="Mask")
        
        plt.subplot(1,3,3)
        show_image(pred_mask, title="Predicted Mask")
            
        plt.tight_layout()
        plt.show()


# ## **Model Training**

# In[64]:


nepochs=50
nbatch_size=12
cbs = [ShowProgress()]
SPE=len(x_train)//nbatch_size


# In[65]:


#Fit Model
history = model.fit(data_generator(x_train, y_train,nbatch_size), epochs=nepochs, validation_data=(x_test, y_test),
                    verbose=1,shuffle=True, steps_per_epoch=SPE,callbacks=cbs,
                    max_queue_size=8,workers=24,use_multiprocessing=True,
                   )
#batch_size=nbatch_size,


# In[ ]:


df_result = pd.DataFrame(history.history)
df_result.sort_values('val_loss', ascending=True, inplace = True)
df_result


# ## **Performance evaluation**

# In[ ]:


plt.figure(figsize = (12,8))
plt.title("Learning curve")
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();

plt.figure(figsize = (12,8))
plt.title("Learning curve")
plt.plot(history.history["accuracy"], label="Accuracy")
plt.plot(history.history["val_accuracy"], label="val_Accuracy")
plt.plot(np.argmax(history.history["val_accuracy"]), np.max(history.history["val_accuracy"]), marker="x", color="r",
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend();

plt.figure(figsize = (12,8))
plt.title("Learning curve")
plt.plot(history.history["jaccard_loss"], label="Jaccard_loss")
plt.plot(history.history["val_jaccard_loss"], label="val_Jaccard_loss")
plt.plot(np.argmax(history.history["val_jaccard_loss"]), np.max(history.history["val_jaccard_loss"]), marker="x", color="r",
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("Jaccard Loss")
plt.legend();


plt.figure(figsize = (12,8))
plt.title("Learning curve")
plt.plot(history.history["iou"], label="IOU")
plt.plot(history.history["val_iou"], label="val_iou")
plt.plot(np.argmax(history.history["val_iou"]), np.max(history.history["val_iou"]), marker="x", color="r",
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("IOU")
plt.legend();


# ## **Model Evaluation**

# In[ ]:


# predict test images
predict_y = model.predict(x_test)

plt.figure(figsize=(15,15))
for i in range(0,9,3):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
    plt.title('Image')
    
    plt.subplot(4,3,i+2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_test[i])
    plt.title('Mask')
    
    plt.subplot(4,3,i+3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(predict_y[i])
    plt.title('Output by model')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# ## **Model predictions**

# In[ ]:


figure, axes = plt.subplots(3,3, figsize=(20,20))
for i in tqdm(range(0,3)):
    rand_num = random.randint(0,50)
    original_img = x_test[rand_num]
    axes[i,0].imshow(original_img)
    axes[i,0].title.set_text('Original Image')
    
    original_mask = y_test[rand_num]
    axes[i,1].imshow(original_mask)
    axes[i,1].title.set_text('Original Mask')
    
    original_img = np.expand_dims(original_img, axis=0)
    predicted_mask = model.predict(original_img).reshape(H,W)
    axes[i,2].imshow(predicted_mask)
    axes[i,2].title.set_text('Predicted Mask')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()    


# ## **Predict & Evaluate Model**

# In[ ]:


# Predict on train, val and test
preds_train = model.predict(x_train[:int(x_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(x_train[int(x_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(x_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Perform a sanity check on some random training samples
ix = random.randint(0, 20)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 10))
ax[0].imshow(x_train[ix])
ax[0].axis('off')
ax[0].set_title('Image')

ax[1].imshow(y_train[ix])
ax[1].axis('off')
ax[1].set_title('Mask')

ax[2].imshow(np.squeeze(preds_train_t[ix]), cmap='cividis')
ax[2].axis('off')
ax[2].set_title('Predicted mask')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# ## **Evaluation**

# In[ ]:


def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

for i in tqdm(range(20)):
    idx = randint(len(x_test))
    image = x_test[idx]
    mask = y_test[idx]
    pred_mask = model.predict(tf.expand_dims(image,axis=0))[0]
    post_process = (pred_mask[:,:,0] > 0.5).astype('uint8')
        
    plt.figure(figsize=(10,8))
    plt.subplot(1,4,1)
    show_image(image, title="Original Image")
        
    plt.subplot(1,4,2)
    show_image(mask, title="Original Mask")
        
    plt.subplot(1,4,3)
    show_image(pred_mask, title="Predicted Mask")
    
    plt.subplot(1,4,4)
    show_image(post_process, title="Post=Processed Mask")      
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()  


# ## **Segmentation predictions**

# In[ ]:


# Creating predictions on our test set-----------------
predictions = model.predict(x_test)
# create predictes mask--------------
def create_mask(predictions,input_shape=(W,H,1)):
    mask = np.zeros(input_shape)
    #mask[predictions>0.5] = 1
    mask = (predictions > 0.5).astype('uint8')
    return mask


# In[ ]:


# Ploting results for one image----------------
def plot_results_for_one_sample(sample_index):    
    mask =create_mask(predictions[sample_index]) #create_mask(predictions[sample_index])   for gray-scale
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
    plt.imshow(mask, cmap='Paired')
    plt.axis('off')
    plt.grid(None)
    #Segment---------------
    fig.add_subplot(1,4,4)
    plt.title("Segment image")
    plt.imshow(x_test[sample_index]*mask)
    plt.grid(None)
    plt.axis('off')  
    fig.tight_layout()    

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()    


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(1)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(6)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(10)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(14)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(20)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(25)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(30)


# ## **Model inference on datasets**

# In[ ]:


model.evaluate(x_test, y_test, verbose=1)
# Predict on train, val and test
preds_train = model.predict(x_train, verbose=1)
preds_test = model.predict(x_test, verbose=1)

# Threshold predictions
preds_train_th = (preds_train > 0.5).astype(np.uint8)
preds_test_th = (preds_test > 0.5).astype(np.uint8)


# In[ ]:


#plot results
def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='g',alpha=1, linewidths=3, levels=[0.5])
    ax[0].set_title(' Image')
    ax[0].set_axis_off()

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title(' Mask Image')
    ax[1].set_axis_off()

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='g',alpha=1, linewidths=3,levels=[0.5])
    ax[2].set_title(' Image Predicted')
    ax[2].set_axis_off()
    
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='b',alpha=1,  linewidths=3, levels=[0.5])
    ax[3].set_title(' Mask Image Predicted binary');
    ax[3].set_axis_off()    

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()     


# ## **Predictions on training set**
# 

# In[ ]:


plot_sample(x_train, y_train, preds_train, preds_train_th, ix=5)


# In[ ]:


plot_sample(x_train, y_train, preds_train, preds_train_th)


# In[ ]:


plot_sample(x_train, y_train, preds_train, preds_train_th)


# In[ ]:


plot_sample(x_train, y_train, preds_train, preds_train_th)


# In[ ]:


plot_sample(x_train, y_train, preds_train, preds_train_th)


# ## **------------------------------Done-----------------------------------------**

# In[ ]:





# In[ ]:




