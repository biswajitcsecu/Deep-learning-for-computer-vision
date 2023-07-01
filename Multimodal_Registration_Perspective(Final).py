#!/usr/bin/env python
# coding: utf-8

# ## **Image-Registration-Perspective**

# In[24]:


from __future__ import print_function
from skimage import io
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from skimage.transform import pyramid_gaussian
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage import feature
from torch.autograd import Function
import cv2
from IPython.display import clear_output
import pandas as pd
from skimage.io import imsave
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## **Load images**
# 

# In[7]:


# fixed image
original_I = io.imread("images/a1.jpg").astype(np.float32) 
# moving image
original_J = io.imread("images/a2.jpg").astype(np.float32) 
height, width, channels = original_I.shape
I = original_I[int(1.5*height/9):int(7.5*height/9),int(1.5*width/9):int(7.5*width/9)]
J = original_J[int(1.5*height/9):int(7.5*height/9),int(1.5*width/9):int(7.5*width/9)]

fig=plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(I/255)
plt.title("Fixed Image")
plt.axis('off')
fig.add_subplot(1,2,2)
plt.imshow(J/255) 
plt.title("Moving Image")
plt.axis('off')
plt.show()


# ## **Gaussian Pyramid**
# 

# In[8]:


downscale = 2.0
ifplot=True
if np.ndim(I) == 3:
    nChannel=I.shape[2]
    pyramid_I = tuple(pyramid_gaussian(gaussian(I, sigma=1, multichannel=True), downscale=downscale, multichannel=True))
    pyramid_J = tuple(pyramid_gaussian(gaussian(J, sigma=1, multichannel=True), downscale=downscale, multichannel=True))
elif np.ndim(I) == 2:
    nChannel=1
    pyramid_I = tuple(pyramid_gaussian(gaussian(I, sigma=1, multichannel=False), downscale=downscale, multichannel=False))
    pyramid_J = tuple(pyramid_gaussian(gaussian(J, sigma=1, multichannel=False), downscale=downscale, multichannel=False))
else:
    print("Unknown rank for an image")

get_ipython().run_line_magic('matplotlib', 'inline')
fig=plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(pyramid_I[6]/255)
fig.add_subplot(1,2,2)
plt.imshow(pyramid_J[6]/255) 
plt.show()
fig=plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(pyramid_I[5]/255)
fig.add_subplot(1,2,2)
plt.imshow(pyramid_J[5]/255) 
plt.show()
fig=plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(pyramid_I[4]/255)
fig.add_subplot(1,2,2)
plt.imshow(pyramid_J[4]/255) 
plt.show()


# ## **HomographyNet**

# In[10]:


class HomographyNet(nn.Module):
    def __init__(self):
        super(HomographyNet, self).__init__()
        self.B = torch.zeros(8,3,3).to(device)
        self.B[0,0,2] = 1.0
        self.B[1,1,2] = 1.0
        self.B[2,0,1] = 1.0
        self.B[3,1,0] = 1.0
        self.B[4,0,0], self.B[4,1,1] = 1.0, -1.0
        self.B[5,1,1], self.B[5,2,2] = -1.0, 1.0
        self.B[6,2,0] = 1.0
        self.B[7,2,1] = 1.0

        self.v1 = torch.nn.Parameter(torch.zeros(8,1,1).to(device), requires_grad=True)
        self.vL = torch.nn.Parameter(torch.zeros(8,1,1).to(device), requires_grad=True)

    def forward(self, s):
        C = torch.sum(self.B*self.vL,0)
        if s==0:
            C += torch.sum(self.B*self.v1,0)
        A = torch.eye(3).to(device)
        H = A
        for i in torch.arange(1,10):
            A = torch.mm(A/i,C)
            H = H + A
        return H


# ## **MINE network**

# In[11]:


n_neurons = 100
class MINE(nn.Module): 
    def __init__(self):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(2*nChannel, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, 1)
        self.bsize = 1 

    def forward(self, x, ind):
        x = x.view(x.size()[0]*x.size()[1],x.size()[2])
        MI_lb=0.0
        for i in range(self.bsize):
            ind_perm = ind[torch.randperm(len(ind))]
            z1 = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x[ind,:])))))
            z2 = self.fc3(F.relu(self.fc2(F.relu(self.fc1(torch.cat((x[ind,0:nChannel],x[ind_perm,nChannel:2*nChannel]),1))))))
            MI_lb += torch.mean(z1) - torch.log(torch.mean(torch.exp(z2)))

        return MI_lb/self.bsize


# In[12]:


def PerspectiveTransform(I, H, xv, yv):
    # apply homography
    xvt = (xv*H[0,0]+yv*H[0,1]+H[0,2])/(xv*H[2,0]+yv*H[2,1]+H[2,2])
    yvt = (xv*H[1,0]+yv*H[1,1]+H[1,2])/(xv*H[2,0]+yv*H[2,1]+H[2,2])
    J = F.grid_sample(I,torch.stack([xvt,yvt],2).unsqueeze(0)).squeeze()
    return J

def multi_resolution_loss():
    loss=0.0
    for s in np.arange(L-1,-1,-1):
        if nChannel>1:
            Jw_ = PerspectiveTransform(J_lst[s].unsqueeze(0), homography_net(s), xy_lst[s][:,:,0], xy_lst[s][:,:,1]).squeeze()
            mi = mine_net(torch.cat([I_lst[s],Jw_],0).permute(1,2,0),ind_lst[s])
            loss = loss - (1./L)*mi
        else:
            Jw_ = PerspectiveTransform(J_lst[s].unsqueeze(0).unsqueeze(0), homography_net(s), 
                                       xy_lst[s][:,:,0], xy_lst[s][:,:,1]).squeeze()
            mi = mine_net(torch.stack([I_lst[s],Jw_],2),ind_lst[s])
            loss = loss - (1./L)*mi

    return loss


def histogram_mutual_information(image1, image2):
    hgram, x_edges, y_edges = np.histogram2d(image1.ravel(), image2.ravel(), bins=100)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


# In[14]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[15]:


L = 6 
sampling = 0.1

I_lst,J_lst,h_lst,w_lst,xy_lst,ind_lst=[],[],[],[],[],[]

for s in range(L):
    I_ = torch.tensor(cv2.normalize(pyramid_I[s].astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_32F)).to(device)
    J_ = torch.tensor(cv2.normalize(pyramid_J[s].astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_32F)).to(device)

    if nChannel>1:
        I_lst.append(I_.permute(2,0,1))
        J_lst.append(J_.permute(2,0,1))
        h_, w_ = I_lst[s].shape[1], I_lst[s].shape[2]

        ind_ = torch.randperm(int(h_*w_*sampling))
        ind_lst.append(ind_)
    else:
        I_lst.append(I_)
        J_lst.append(J_)
        h_, w_ = I_lst[s].shape[0], I_lst[s].shape[1]

        edges_grayscale = cv2.dilate(cv2.Canny(cv2.GaussianBlur(rgb2gray(pyramid_I[s]),(21,21),0).astype(np.uint8), 0, 10),
                                 np.ones((5,5),np.uint8),
                                 iterations = 1)
        ind_ = torch.randperm(int(h_*w_*sampling))
        ind_lst.append(ind_)
    h_lst.append(h_)
    w_lst.append(w_)

    y_, x_ = torch.meshgrid([torch.arange(0,h_).float().to(device), torch.arange(0,w_).float().to(device)])
    y_, x_ = 2.0*y_/(h_-1) - 1.0, 2.0*x_/(w_-1) - 1.0
    xy_ = torch.stack([x_,y_],2)
    xy_lst.append(xy_)


# ## **Homography parameters**

# In[16]:


homography_net = HomographyNet().to(device)
mine_net = MINE().to(device)

optimizer = optim.Adam([{'params': mine_net.parameters(), 'lr': 1e-3},
                    {'params': homography_net.vL, 'lr': 1e-3},
                    {'params': homography_net.v1, 'lr': 1e-4}], amsgrad=True)
mi_list = []
for itr in range(500):
    optimizer.zero_grad()
    loss = multi_resolution_loss()
    mi_list.append(-loss.item())
    loss.backward()
    optimizer.step()
    clear_output(wait=True)
    plt.plot(mi_list)
    plt.title("MI")
    plt.show()


# In[17]:


I_t = torch.tensor(I).to(device) 
J_t = torch.tensor(J).to(device) 
H = homography_net(0)
if nChannel>1:
    J_w = PerspectiveTransform(J_t.permute(2,0,1).unsqueeze(0), H, xy_lst[0][:,:,0], xy_lst[0][:,:,1]).squeeze().permute(1,2,0)
else:
    J_w = PerspectiveTransform(J_t.unsqueeze(0).unsqueeze(0), H , xy_lst[0][:,:,0], xy_lst[0][:,:,1]).squeeze()


# In[23]:


D = J_t - I_t
D_w = J_w - I_t
D = (D - torch.min(D))/(torch.max(D) - torch.min(D))
D_w = (D_w - torch.min(D_w))/(torch.max(D_w) - torch.min(D_w))

fig=plt.figure(figsize=(10,10))
fig.add_subplot(1,2,1)
plt.imshow(D.cpu().data)
plt.title("Difference image before registration")
fig.add_subplot(1,2,2)
plt.imshow(D_w.cpu().data)         
plt.title("Difference image after registration")
plt.show()


# ## **Apply transformation to original image**
# 

# In[19]:


normalized_affine_matrix = H.detach().cpu().numpy()

h = I.shape[0]
w = J.shape[1]

test_transform = [[2/w, 0, -1],[0, 2/h, -1], [0,0,1]]
unnormalized_affine_matrix = np.linalg.inv(np.matmul(np.matmul(np.linalg.inv(test_transform), normalized_affine_matrix), 
                                                     test_transform))

tx = -int(1.5*original_I.shape[0]/9)
ty = -int(1.5*original_I.shape[0]/9)
move_origin = np.array([[1,0,tx],[0,1,ty],[0,0,1]])

matrix_for_full_image = np.matmul(np.matmul(np.linalg.inv(move_origin), unnormalized_affine_matrix), move_origin)


# In[20]:


rows,cols,ch = original_J.shape
dst = cv2.warpAffine(original_J, matrix_for_full_image[:2], (cols,rows))


# In[22]:


D = original_J - original_I
D_w = dst - original_I
D = (D - np.min(D))/(np.max(D) - np.min(D))
D_w = (D_w - np.min(D_w))/(np.max(D_w) - np.min(D_w))


fig=plt.figure(figsize=(10,10))
fig.add_subplot(1,2,1)
plt.imshow(D)
plt.title("Difference image before registration")
fig.add_subplot(1,2,2)
plt.imshow(D_w)         
plt.title("Difference image after registration")
plt.show()


# In[ ]:




