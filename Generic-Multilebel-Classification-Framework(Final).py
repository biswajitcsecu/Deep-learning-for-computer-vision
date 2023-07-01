#!/usr/bin/env python
# coding: utf-8

# In[67]:


## Import required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import io
import cv2
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Add, MaxPooling2D, \
                                    Conv2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report
from scikitplot.metrics import plot_roc


import warnings
warnings.filterwarnings('ignore')

tf.keras.backend.clear_session()
print(tf.__version__)


# In[68]:


path = 'RSSCN/'
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
labels = ["Field", "Forest", "Grass", "Industry","Parking","Resident","RiverLake"]


# In[73]:


# Hyperparameters Settings
EPOCHS = 100
BATCH_SIZE = 48
LEARNING_RATE = 0.0001
H,W,CH=[128,128,3]


# ## **Visualize Images In Different Classes**
# 

# In[74]:


#Visualize Images In Different Classes

plt.style.use("dark_background")
fig, axs = plt.subplots(len(labels), 5, figsize = (12,15))
class_len = {}
for i, c in enumerate(labels):
    class_path = os.path.join(path, c)
    all_images = os.listdir(class_path)
    sample_images = random.sample(all_images, 5)
    class_len[c] = len(all_images)
    
    for j, image in enumerate(sample_images):
        img_path = os.path.join(class_path, image)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axs[i, j].imshow(img)
        axs[i, j].set(xlabel = c, xticks = [], yticks = [])        

fig.tight_layout()


# In[75]:


fig, ax = plt.subplots()
ax.pie(
    class_len.values(),
    labels = class_len.keys(),
    autopct = "%1.1f%%",
    colors = ["pink", "teal", "orange","red","green","blue","yellow"]
)
fig.show()


# ## **Data Preprocessing & Data Augmentation**
# 

# In[77]:


# Data Generator
datagen_train = ImageDataGenerator(
    rescale = 1./255,     
    validation_split = 0.2,
    rotation_range = 15, 
#     horizontal_flip = True,
    shear_range = 0.05,
    width_shift_range = 0.15,
    height_shift_range = 0.15, 
    zoom_range = 0.1
)

datagen_val = ImageDataGenerator(
    rescale = 1./255, 
    validation_split = 0.2 
)    

train_generator = datagen_train.flow_from_directory(
    directory = path,
    target_size=(H, W),
    classes = labels,
    seed = SEED,
    batch_size = BATCH_SIZE, 
    shuffle = True,
    interpolation="bilinear",
    follow_links=False,
    subset = 'training'
)

valid_generator = datagen_val.flow_from_directory(
    directory = path,
    target_size=(H, W),
    classes = labels,
    seed = SEED,
    batch_size = BATCH_SIZE, 
    shuffle = True,
    interpolation="bilinear",
    follow_links=False,
    subset = 'validation'
)

valid_generator.class_indices


# ## **Display Augmented Images**
# 

# In[78]:


# Get Second Batch of Augmented Images
next(train_generator) 
batch = next(train_generator)  
print(batch[0].shape) 


# In[79]:


# Show processed image and their corresponding label
display_len = 30
fig, axs = plt.subplots(3, 10, figsize = (30, 10))

j = 0
for n in range(display_len):
    i = n%3
    if j == 10: j=0
    axs[i, j].imshow(batch[0][n])
    label = labels[np.argmax(batch[1][n])] 
    axs[i, j].set(xlabel = label, xticks = [], yticks = [])
    j += 1


# ## **Load Pretrained InceptionV3**
# 

# In[80]:


base_model = InceptionV3(
    include_top = False, 
    weights = 'imagenet', 
    input_tensor = Input((H, W, CH)),
)


# In[81]:


base_model.trainable = True
base_model.summary()


# ## **Create Transfer Learning Model**
# 

# In[82]:


# Add classification head to the model
C=len(labels)
head_model_start = base_model.output
head_model = GlobalAveragePooling2D()(head_model_start)
head_model = Flatten()(head_model) 
head_model = Dense(256, activation = "relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(128, activation = "relu")(head_model)
head_model = Dropout(0.3)(head_model)
head_model = Dense(C, activation = "softmax")(head_model) # 3 output classes

cnn_model = Model(inputs = base_model.input, outputs = head_model)
cnn_model.summary()


# ## *Compile the model*

# In[83]:


# Compile the model
metrics_list = ["accuracy"]
metrics_list += [Recall(class_id = i) for i in range(len(labels))] 
metrics_list += [Precision(class_id = i) for i in range(len(labels))]

op = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, epsilon=1e-07)

cnn_model.compile(
    loss = "categorical_crossentropy",
    optimizer = op,
    metrics = metrics_list
)


# ## **Model Training**
# 

# In[ ]:


#Model Training

history = cnn_model.fit_generator(
    train_generator, shuffle=True,
    validation_data = valid_generator,
    steps_per_epoch = BATCH_SIZE,max_queue_size=10,
    epochs = EPOCHS, workers=1,
    use_multiprocessing=True,
)


# ## **Model Evaluation**

# In[ ]:


# Plot loss per epoch

train_loss = history.history["loss"]
valid_loss = history.history["val_loss"]

epochs = range(len(train_loss)) 

plt.plot(epochs, train_loss)
plt.plot(epochs, valid_loss)
plt.legend(["Training Loss", "Validation Loss"])
plt.title("Loss")
plt.show()


# In[ ]:


# Plot accuracy per epoch

train_acc = history.history["accuracy"]
valid_acc = history.history["val_accuracy"]

epochs = range(len(train_acc)) 

plt.plot(epochs, train_acc)
plt.plot(epochs, valid_acc)
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.title("Accuracy")
plt.show()


# In[ ]:


# Plot recall per epoch

fig, axs = plt.subplots(1, 3, figsize = (15, 5))

train_rec_0 = history.history["recall"]
valid_rec_0 = history.history["val_recall"]
train_rec_1 = history.history["recall_1"]
valid_rec_1 = history.history["val_recall_1"]
train_rec_2 = history.history["recall_2"]
valid_rec_2 = history.history["val_recall_2"]

epochs = range(len(train_rec_0)) 

axs[0].plot(epochs, train_rec_0)
axs[0].plot(epochs, valid_rec_0)
axs[0].legend(["Training Recall", "Validation Recall"])
axs[0].set_title("Recall for class 0")

axs[1].plot(epochs, train_rec_1)
axs[1].plot(epochs, valid_rec_1)
axs[1].legend(["Training Recall", "Validation Recall"])
axs[1].set_title("Recall for class 1")

axs[2].plot(epochs, train_rec_2)
axs[2].plot(epochs, valid_rec_2)
axs[2].legend(["Training Recall", "Validation Recall"])
axs[2].set_title("Recall for class 2")

fig.tight_layout()


# In[ ]:


# Plot precision per epoch

fig, axs = plt.subplots(1, 3, figsize = (15, 5))

train_pre_0 = history.history["precision"]
valid_pre_0 = history.history["val_precision"]
train_pre_1 = history.history["precision_1"]
valid_pre_1 = history.history["val_precision_1"]
train_pre_2 = history.history["precision_2"]
valid_pre_2 = history.history["val_precision_2"]

epochs = range(len(train_pre_0)) 

axs[0].plot(epochs, train_pre_0)
axs[0].plot(epochs, valid_pre_0)
axs[0].legend(["Training Precision", "Validation Precision"])
axs[0].set_title("Precision for class 0")

axs[1].plot(epochs, train_pre_1)
axs[1].plot(epochs, valid_pre_1)
axs[1].legend(["Training Precision", "Validation Precision"])
axs[1].set_title("Precision for class 1")

axs[2].plot(epochs, train_pre_2)
axs[2].plot(epochs, valid_pre_2)
axs[2].legend(["Training Precision", "Validation Precision"])
axs[2].set_title("Precision for class 2")

fig.tight_layout()


# In[ ]:


# Sample random images from Validation Set for each class
valid_batch = next(valid_generator)
valid_class_0 = []
valid_class_1 = []
valid_class_2 = []
for i in range(100):
    rand = random.randint(0, BATCH_SIZE-1)
    image_onehot = valid_batch[1][rand]
#     print(image_onehot)
    if(image_onehot[0] == 1 and len(valid_class_0) < 5):
        valid_class_0.append(rand)
    elif(image_onehot[1] == 1 and len(valid_class_1) < 5):
        valid_class_1.append(rand)
    elif(image_onehot[2] == 1 and len(valid_class_2) < 5):
        valid_class_2.append(rand)
    
    if(len(valid_class_0) + len(valid_class_1) + len(valid_class_2) >= 15):
        break


# ## **Activation Maps (Feature Maps)**
# 

# In[ ]:


print(len(cnn_model.layers))
for i in range(len(cnn_model.layers)):
    layer = cnn_model.layers[i]
    if 'conv' not in layer.name:
        continue
    print(i, layer.name, layer.output.shape)


# In[ ]:


conv_list = [7, 11, 30, 50, 299] 
outputs = [cnn_model.layers[i].output for i in conv_list]
model = Model(inputs=cnn_model.inputs, outputs=outputs)

channels = 8
fig, axs = plt.subplots(len(conv_list), channels, figsize = (32, 16))

img = valid_batch[0][valid_class_0[3]]
img = np.expand_dims(img, axis=0)
feature_maps = model.predict(img)
for index, fmap in enumerate(feature_maps):
    for j in range(channels):
        # plot filter channel
        axs[index, j].imshow(fmap[0, :, :, j])
        info = "Layer " + str(conv_list[index]) + " Channel " + str(j+1) 
        axs[index, j].set(xlabel = info, xticks = [], yticks = [])
    
# show the figure
plt.show()  


# ## **Compute Saliency Map**
# 

# In[ ]:


valid_images = []
valid_images.append(valid_batch[0][valid_class_0[4]])
valid_images.append(valid_batch[0][valid_class_1[4]])
valid_images.append(valid_batch[0][valid_class_2[2]])


# In[ ]:


layers = [layer.output for layer in cnn_model.layers]

for i, _img in enumerate(valid_images):
    img = _img.reshape((1, *_img.shape))
    images = tf.Variable(img, dtype=float)

    with tf.GradientTape() as tape:
        tape.reset()
        pred = cnn_model(images, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]        

    grads = tape.gradient(loss, images)

    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]

    ## normalize to range between 0 and 1
    arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
    
    fig, axs = plt.subplots(1, 2, figsize=(14,5))
    axs[0].imshow(_img)
    im = axs[1].imshow(grad_eval, cmap="jet", alpha=0.8)
    axs[0].set(xlabel = labels[i], xticks = [], yticks = [])
    axs[1].set(xlabel = "Saliency Map", xticks = [], yticks = [])
    fig.colorbar(im)
    
    plt.show()


# ## *Confusion Matrix*

# In[ ]:


# Use the validation set for testing
valid_generator = datagen_val.flow_from_directory(
    directory = path,
    classes = labels,
    seed = SEED,
    batch_size = BATCH_SIZE, 
    shuffle = False,
    subset = 'validation'
)

pred = cnn_model.predict_generator(valid_generator) 
y_pred = np.round(pred) 
y_pred = np.argmax(y_pred, axis=1) 

# Obtain actual labels
y_true = valid_generator.classes
    
# Now plot matrix
cm = confusion_matrix(y_true, y_pred, labels = [0, 1, 2])
sns.heatmap(
    cm, 
    cmap="Greens",
    annot = True, 
    fmt = "d"
)
plt.show()


# In[ ]:


# ROC curve
fig, ax = plt.subplots(figsize=(10,6))
plot_roc(y_true, pred, ax=ax)


# In[ ]:


# precision, recall, f1-score, support
print(classification_report(y_true, y_pred))


# In[ ]:





# In[ ]:




