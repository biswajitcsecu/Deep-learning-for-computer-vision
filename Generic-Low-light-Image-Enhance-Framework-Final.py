#!/usr/bin/env python
# coding: utf-8

# # **Generic Framework Low-light Image Enhance  (Final)**

# In[17]:


import os
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from plotly.offline import iplot, init_notebook_mode
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from keras.utils import plot_model
import warnings

warnings.filterwarnings('ignore')
K.clear_session()


# ## **TensorFlow Dataset**

# In[4]:


#image loading
IMAGE_SIZE = 128
BATCH_SIZE = 32
MAX_TRAIN_IMAGES = 1000

def load_data(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    image = image / 255.0
    return image

def data_generator(low_light_images):
    dataset = tf.data.Dataset.from_tensor_slices((low_light_images))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

train_low_light_images = sorted(glob("UFO/train/low/*"))[:MAX_TRAIN_IMAGES]
val_low_light_images = sorted(glob("UFO/train/low/*"))[MAX_TRAIN_IMAGES:]
test_low_light_images = sorted(glob("UFO/test/low/*"))

train_dataset = data_generator(train_low_light_images)
val_dataset = data_generator(val_low_light_images)

print("Train Dataset:", train_dataset)
print("Validation Dataset:", val_dataset)


# In[5]:


#plot images
plt.figure(figsize=(20, 20))
plt.title('Samples', fontweight='bold')
for images in train_dataset.take(1):
    for i in range(12):
        ax = plt.subplot(5, 4, i + 1)
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.imshow(images[i])
        plt.title('Low-light image', fontweight='bold')
        plt.axis("off")
        plt.tight_layout()
        
plt.show()


# ## **DCE Framework**

# In[6]:


#DCE-Net
def DCENET():
    input_img = keras.Input(shape=[None, None, 3])
    
    conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(input_img)
    
    conv2 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv1)
    
    conv3 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv2)
    
    conv4 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv3)
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    
    conv5 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(int_con1)
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    
    conv6 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(int_con2)
    
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same")(int_con3)
    
    return keras.Model(inputs=input_img, outputs=x_r)


# ## **Loss functions**

# In[7]:


#Color constancy loss
def color_constancy_loss(x):
    mean_rgb = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
    mr, mg, mb = mean_rgb[:, :, :, 0], mean_rgb[:, :, :, 1], mean_rgb[:, :, :, 2]
    d_rg = tf.square(mr - mg)
    d_rb = tf.square(mr - mb)
    d_gb = tf.square(mb - mg)
    return tf.sqrt(tf.square(d_rg) + tf.square(d_rb) + tf.square(d_gb))


# In[8]:


#Exposure loss
def exposure_loss(x, mean_val=0.6):
    x = tf.reduce_mean(x, axis=3, keepdims=True)
    mean = tf.nn.avg_pool2d(x, ksize=16, strides=16, padding="VALID")
    return tf.reduce_mean(tf.square(mean - mean_val))


# In[9]:


#Illumination smoothness loss
def illumination_smoothness_loss(x):
    batch_size = tf.shape(x)[0]
    h_x = tf.shape(x)[1]
    w_x = tf.shape(x)[2]
    count_h = (tf.shape(x)[2] - 1) * tf.shape(x)[3]
    count_w = tf.shape(x)[2] * (tf.shape(x)[3] - 1)
    h_tv = tf.reduce_sum(tf.square((x[:, 1:, :, :] - x[:, : h_x - 1, :, :])))
    w_tv = tf.reduce_sum(tf.square((x[:, :, 1:, :] - x[:, :, : w_x - 1, :])))
    batch_size = tf.cast(batch_size, dtype=tf.float32)
    count_h = tf.cast(count_h, dtype=tf.float32)
    count_w = tf.cast(count_w, dtype=tf.float32)
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


# In[10]:


#Spatial consistency loss


class SpatialConsistencyLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super(SpatialConsistencyLoss, self).__init__(reduction="none")

        self.left_kernel = tf.constant([[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32)
        self.right_kernel = tf.constant([[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]], dtype=tf.float32)
        self.up_kernel = tf.constant([[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32)
        self.down_kernel = tf.constant([[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]], dtype=tf.float32        )

    def call(self, y_true, y_pred):
        original_mean = tf.reduce_mean(y_true, 3, keepdims=True)
        enhanced_mean = tf.reduce_mean(y_pred, 3, keepdims=True)
        original_pool = tf.nn.avg_pool2d(original_mean, ksize=4, strides=4, padding="VALID")
        enhanced_pool = tf.nn.avg_pool2d(enhanced_mean, ksize=4, strides=4, padding="VALID")

        d_original_left = tf.nn.conv2d(original_pool, self.left_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_original_right = tf.nn.conv2d(original_pool, self.right_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_original_up = tf.nn.conv2d(original_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_original_down = tf.nn.conv2d(original_pool, self.down_kernel, strides=[1, 1, 1, 1], padding="SAME")

        d_enhanced_left = tf.nn.conv2d(enhanced_pool, self.left_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_enhanced_right = tf.nn.conv2d(enhanced_pool, self.right_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_enhanced_up = tf.nn.conv2d(enhanced_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_enhanced_down = tf.nn.conv2d(enhanced_pool, self.down_kernel, strides=[1, 1, 1, 1], padding="SAME")

        d_left = tf.square(d_original_left - d_enhanced_left)
        d_right = tf.square(d_original_right - d_enhanced_right)
        d_up = tf.square(d_original_up - d_enhanced_up)
        d_down = tf.square(d_original_down - d_enhanced_down)
        
        return d_left + d_right + d_up + d_down


# ## **Curve estimation**

# In[11]:


#Deep curve estimation model
class ZeroDCE(keras.Model):
    def __init__(self, **kwargs):
        super(ZeroDCE, self).__init__(**kwargs)
        self.dce_model = DCENET()

    def compile(self, learning_rate, **kwargs):
        super(ZeroDCE, self).compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.spatial_constancy_loss = SpatialConsistencyLoss(reduction="none")

    def get_enhanced_image(self, data, output):
        r1 = output[:, :, :, :3]
        r2 = output[:, :, :, 3:6]
        r3 = output[:, :, :, 6:9]
        r4 = output[:, :, :, 9:12]
        r5 = output[:, :, :, 12:15]
        r6 = output[:, :, :, 15:18]
        r7 = output[:, :, :, 18:21]
        r8 = output[:, :, :, 21:24]
        x = data + r1 * (tf.square(data) - data)
        x = x + r2 * (tf.square(x) - x)
        x = x + r3 * (tf.square(x) - x)
        enhanced_image = x + r4 * (tf.square(x) - x)
        x = enhanced_image + r5 * (tf.square(enhanced_image) - enhanced_image)
        x = x + r6 * (tf.square(x) - x)
        x = x + r7 * (tf.square(x) - x)
        enhanced_image = x + r8 * (tf.square(x) - x)
        return enhanced_image

    def call(self, data):
        dce_net_output = self.dce_model(data)
        return self.get_enhanced_image(data, dce_net_output)

    def compute_losses(self, data, output):
        enhanced_image = self.get_enhanced_image(data, output)
        loss_illumination = 200 * illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(self.spatial_constancy_loss(enhanced_image, data))
        loss_color_constancy = 5 * tf.reduce_mean(color_constancy_loss(enhanced_image))
        loss_exposure = 10 * tf.reduce_mean(exposure_loss(enhanced_image))
        total_loss = (
            loss_illumination
            + loss_spatial_constancy
            + loss_color_constancy
            + loss_exposure
        )
        return {
            "total_loss": total_loss,
            "illumination_smoothness_loss": loss_illumination,
            "spatial_constancy_loss": loss_spatial_constancy,
            "color_constancy_loss": loss_color_constancy,
            "exposure_loss": loss_exposure,
        }

    def train_step(self, data):
        with tf.GradientTape() as tape:
            output = self.dce_model(data)
            losses = self.compute_losses(data, output)
        gradients = tape.gradient(losses["total_loss"], self.dce_model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.dce_model.trainable_weights))
        return losses

    def test_step(self, data):
        output = self.dce_model(data)
        return self.compute_losses(data, output)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.dce_model.save_weights(filepath, overwrite=overwrite, save_format=save_format, options=options)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.dce_model.load_weights(filepath=filepath,by_name=by_name,skip_mismatch=skip_mismatch,options=options,)
        


# ## **Training**

# In[22]:


#Training
model = ZeroDCE()
model.compile(learning_rate=1e-4,metrics=['accuracy'])


# In[ ]:


tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)


# In[ ]:


nepochs=100
history = model.fit(train_dataset, validation_data=val_dataset, 
                             epochs=nepochs,shuffle=True,verbose=1, use_multiprocessing=True)


# In[ ]:


history.history


# ## **Performance plots**

# In[ ]:


#plot total loss
tloss = pd.DataFrame({'train': history.history['total_loss'],
                    'val': history.history['val_total_loss']})

fig = px.line(tloss, x=tloss.index, y=tloss.columns[0::], title='Training and Evaluation Total Loss every Epoch', 
              markers=True)
fig.show()


# In[ ]:


#plot illumination loss
loss = pd.DataFrame({'train': history.history['illumination_smoothness_loss'],
                    'val': history.history['val_illumination_smoothness_loss']})

fig = px.line(loss, x=loss.index, y=loss.columns[0::], title='Training and Evaluation Illumination loss every Epoch',
              markers=True)
fig.show()


# In[ ]:


sloss = pd.DataFrame({'train': history.history['spatial_constancy_loss'],
                    'val': history.history['val_spatial_constancy_loss']})

fig = px.line(sloss, x=sloss.index, y=sloss.columns[0::], title='Training and Evaluation Spatial  loss every Epoch',
              markers=True)
fig.show()


# In[ ]:


#plot color constancy loss
closs = pd.DataFrame({'train': history.history['color_constancy_loss'],
                    'val': history.history['val_color_constancy_loss']})

fig = px.line(closs, x=closs.index, y=closs.columns[0::], title='Training and Evaluation Color loss every Epoch',
              markers=True)
fig.show()


# In[ ]:


#plot exposure loss
eloss = pd.DataFrame({'train': history.history['exposure_loss'],
                    'val': history.history['val_exposure_loss']})

fig = px.line(eloss, x=eloss.index, y=eloss.columns[0::], title='Training and Evaluation Exposure loss every Epoch',
              markers=True)
fig.show()


# In[ ]:


model.save_weights("weights")


# ## **Model Evaluation**

# In[ ]:


#Inference
def plot_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()

def infer(original_image):
    image = keras.preprocessing.image.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    return output_image


# In[ ]:


#Prediction on test images
for val_image_file in test_low_light_images:
    original_image = Image.open(val_image_file)
    enhanced_image = infer(original_image)
    plot_results(
        [original_image, ImageOps.autocontrast(original_image), enhanced_image],
        ["Original", "Non-deep Enhanced", "Deep-Enhanced"],
        (20, 12),
    )


# In[ ]:




