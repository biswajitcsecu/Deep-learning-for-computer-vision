#!/usr/bin/env python
# coding: utf-8

# ## **Dual-channel graph convolutional neural network**

# In[11]:


import numpy as np
from keras.layers import Input, Conv2D, Dense, Flatten, Reshape
from keras.models import Model
from keras.layers import concatenate
from keras.models import Model
from keras.layers import Input, Dense, concatenate


# In[12]:


def dual_channel_gcn(input_shape):
    # Define the input tensor
    input_tensor = Input(shape=input_shape)

    # Channel 1
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_tensor)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)
    flatten1 = Flatten()(conv2)

    # Channel 2
    conv3 = Conv2D(32, kernel_size=(5, 5), activation='relu')(input_tensor)
    conv4 = Conv2D(64, kernel_size=(5, 5), activation='relu')(conv3)
    flatten2 = Flatten()(conv4)

    # Concatenate the features from both channels
    concatenated = concatenate([flatten1, flatten2])  

    # Dense layers for classification
    dense1 = Dense(128, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense1)

    # Create the model
    model = Model(inputs=input_tensor, outputs=output)
    return model



# In[13]:


# Create the model
input_shape = (128, 128, 1)  # Example input shape, adjust according to your needs
model = dual_channel_gcn(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
x_train = np.random.rand(1000, 128, 128, 1)  # Example training data, replace with your own
y_train = np.random.randint(2, size=(1000, 1))  # Example labels, replace with your own



# In[ ]:


model.fit(x_train, y_train, batch_size=16, epochs=10)


# In[ ]:




