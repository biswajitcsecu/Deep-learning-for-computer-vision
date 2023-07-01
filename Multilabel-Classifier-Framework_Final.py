#!/usr/bin/env python
# coding: utf-8

# # Import  modules

# In[2]:


import os
import shutil
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score
import keract

import tensorflow as tf
import matplotlib as mpl
from keras import backend
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


import warnings
warnings.filterwarnings('ignore')


# # Read Data

# In[3]:


# Creating batches of data

labels = ['Healthy', 'Powdery','Rust']
train_path = 'Plant/train'
valid_path = 'Plant/valid'
test_path = 'Plant/test'


# In[4]:


#param
H=128
W=128
CH=3
nbatch_size=16

#Batch generation 
train_batches = ImageDataGenerator(preprocessing_function=
    tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=train_path, target_size=(H,W),
    batch_size=nbatch_size)

valid_batches = ImageDataGenerator(preprocessing_function=
    tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=valid_path, target_size=(H,W), 
    batch_size=nbatch_size)

test_batches = ImageDataGenerator(preprocessing_function=
     tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=test_path, target_size=(H,W), 
    batch_size=nbatch_size, shuffle=False)

nclasses=3;


# # Samples from Training Set 

# In[6]:


#Display trainings samples
fig, axes = plt.subplots(nrows = 2, ncols = 8, figsize = (25, 8))
fig.suptitle("Samples from Training Set Batch", fontsize = 16, fontdict = dict(weight = 'bold'))
for curr_axis, curr_image in zip(axes.flatten(), train_batches[0][0][:16]):
    curr_axis.imshow(tf.squeeze(curr_image), cmap = 'gray')
    curr_axis.axis(False)
plt.show()   


# # Samples from Validation Set

# In[7]:


#Display validation samples
fig, axes = plt.subplots(nrows = 2, ncols = 8, figsize = (25, 8))
fig.suptitle("Samples from Validation Set Batch", fontsize = 16, fontdict = dict(weight = 'bold'))
for curr_axis, curr_image in zip(axes.flatten(), valid_batches[0][0][:16]):
    curr_axis.imshow(tf.squeeze(curr_image), cmap = 'gray')
    curr_axis.axis(False)
plt.show()


# # Samples from Testing Set

# In[8]:


#Display test samples
fig, axes = plt.subplots(nrows = 2, ncols = 8, figsize = (25, 8))
fig.suptitle("Samples from Testing Set Batch", fontsize = 16, fontdict = dict(weight = 'bold'))
for curr_axis, curr_image in zip(axes.flatten(), test_batches[0][0][:16]):
    curr_axis.imshow(tf.squeeze(curr_image), cmap = 'gray')
    curr_axis.axis(False)
plt.show()    


# # Pre-trained Mobilenet

# In[9]:


#Loading pre-trained mobilenet  classifier
mobile = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False)
mobile.summary()


# In[10]:


#model config-------------
x = mobile.layers[-12].output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x) 
output = Dense(units=nclasses, activation='sigmoid')(x)


# In[11]:


# new model
model = Model(inputs=mobile.input, outputs=output)

for layer in model.layers[:-23]:
    layer.trainable = False

model.summary()   


# In[12]:


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[13]:


#Train model
nepochs=20

histmob =model.fit(x=train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=nepochs,
          verbose=1,
          use_multiprocessing=True
)


# # Display model performance

# In[14]:


# summarize history for accuracy
plt.plot(histmob.history['accuracy'])
plt.plot(histmob.history['val_accuracy'])
plt.title('Mobilenet Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(histmob.history['loss'])
plt.plot(histmob.history['val_loss'])
plt.title('Mobilenet Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# # Model prediction

# In[18]:


# 'Healthy' , 'Powdery' and 'Rust' images 

test_labels = test_batches.classes

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))

accuracy = accuracy_score(y_true=test_labels, y_pred=predictions.argmax(axis=1))

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
 


# In[19]:


# Pring accuracy of  model
print('Accuracy: ', accuracy)


# In[20]:


# Confusion Matrix 
test_batches.class_indices
cm_plot_labels = ['Healthy','Powdery','Rust' ] 
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


# In[21]:


#classification report
print(classification_report(y_true=test_labels, y_pred=predictions.argmax(axis=1)))


# In[22]:


# Prepare image for mobilenet prediction
def preprocess_image(file):   
    img = image.load_img(file, target_size=(H, W))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


from IPython.display import Image
img_path = 'Plant/sample/pl4.jpg'
Image(filename=img_path, width=256,height=256) 


# In[23]:


# Preprocess image and make prediction
preprocessed_image = preprocess_image(img_path)
predictions = model.predict(preprocessed_image)
predictions


# In[24]:


result = np.argmax(predictions)
labels[result]


# In[25]:


#Set of test data
img, l = next(iter(test_batches))

plt.figure(figsize=(30,10))
for i in range(8):
    actual_label = list(test_labels)[np.argmax(l[i])]
    predicted_label = np.argmax(model.predict(img[i][None,...]))
    plt.subplot(2,4,i+1)
    plt.axis('off')
    plt.imshow(tf.squeeze(img[i]))
    plt.title(f'actual label: {actual_label}, predicted label: {list(l)[predicted_label]}')


# # Activation silency map

# In[26]:


#Activation silency map
image = load_img(img_path, target_size= (H, W))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
y_hat = model.predict(image)


# In[27]:


#layers
layers=['conv1','conv1_bn','conv_pw_2','conv_pw_3','conv_pw_4']


# In[28]:


#salience map
activations= keract.get_activations(model, image, layer_names= layers, nodes_to_evaluate= None, output_format= 'simple', 
                                    auto_compile= True)
keract.display_activations(activations, cmap='viridis', save= False, directory= 'activations')


# In[ ]:




