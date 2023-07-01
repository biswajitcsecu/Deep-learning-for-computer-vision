#!/usr/bin/env python
# coding: utf-8

# ## **Image registration spatial distance of local descriptors**

# In[5]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


# Load the reference and target images
ref_image = cv2.imread("image1.jpg")
target_image = cv2.imread("image2.jpg")

# Convert images to grayscale
ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)


# In[7]:


# Initialize the SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# Detect keypoints and compute descriptors for reference and target images
ref_keypoints, ref_descriptors = sift.detectAndCompute(ref_gray, None)
target_keypoints, target_descriptors = sift.detectAndCompute(target_gray, None)

# Initialize the FlannBasedMatcher object
matcher = cv2.FlannBasedMatcher()

# Match descriptors using k-nearest neighbors
matches = matcher.knnMatch(ref_descriptors, target_descriptors, k=2)

# Filter good matches using Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract the matched keypoints' coordinates
ref_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
target_pts = np.float32([target_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Find the perspective transformation between the reference and target images
M, _ = cv2.findHomography(ref_pts, target_pts, cv2.RANSAC, 5.0)

# Warp the target image using the found perspective transformation
result_image = cv2.warpPerspective(target_image, M, (ref_image.shape[1], ref_image.shape[0]))


# In[8]:


# Plot the original image and the edge map
fig, axs = plt.subplots(1, 3, figsize=(12, 10))
axs[0].imshow(ref_image, cmap='gray')
axs[0].set_title('ref_image')
axs[0].axis('off')
axs[1].imshow(target_image, cmap='gray')
axs[1].set_title('target_image')
axs[1].axis('off')
axs[2].imshow(result_image, cmap='gray')
axs[2].set_title('Image-registration')
axs[2].axis('off')
plt.show()
 


# In[ ]:




