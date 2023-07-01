#!/usr/bin/env python
# coding: utf-8

# ## **Multisensor-IR-Visual-Image-Fusion**

# In[ ]:


##Libraries
from __future__ import print_function
import torch
import numpy as numpy
import cv2
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm
import torch.nn as nn
import torchvision.transforms as transforms
import imageio
from torch.autograd import Variable
from collections import OrderedDict

import math
import time
import sys
import os
import random
import glob
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import PIL.ImageOps 
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## **Data Load andProcessing**

# In[27]:


DATA_DIR = "Data"
trn_dir = f'{DATA_DIR}/train'
tst_dir = f'{DATA_DIR}/test'


# In[28]:


#parameters
nsz = 64
batch_size = 16
num_epochs = 5
os.listdir(DATA_DIR)


# In[29]:


trn_fnames = glob(f'{trn_dir}/*/*.jpg')
trn_fnames[:3]


# In[30]:


img = plt.imread(trn_fnames[7])
plt.imshow(img,cmap='gray');
plt.axis('off')


# In[31]:


train_ds = datasets.ImageFolder(trn_dir)

tfms = transforms.Compose([
    transforms.Resize((nsz, nsz//2)),  # PIL Image
#     transforms.Grayscale(), 
    transforms.ToTensor(),        # Tensor
    transforms.Normalize([0.44 , 0.053, 0.062], [0.076, 0.079, 0.085])
])

train_ds = datasets.ImageFolder(trn_dir, transform=tfms)
test_ds = datasets.ImageFolder(tst_dir, transform=tfms)

len(train_ds), len(test_ds)


# In[32]:


train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,  shuffle=True, num_workers=8)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=8)


# In[33]:


inputs, targets = next(iter(train_dl))
out = torchvision.utils.make_grid(inputs, padding=3)
plt.figure(figsize=(16, 12))
plt.imshow(out[-1],cmap='gray')


# In[34]:


inputs, targets = next(iter(test_dl))
out = torchvision.utils.make_grid(inputs, padding=3)
plt.figure(figsize=(16, 12))
plt.imshow(out[-1],cmap='gray')


# ## **CNN Model**

# In[35]:


#define CNN model
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

        )
        
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.1, inplace=True),
        )
        
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)

        )   
        
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )   
        
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )                
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )              
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )   

        self.fc1 = nn.Linear(256*8*4*2*1, 2)        
        
    def forward(self, x, y, z):
        outx = self.conv1_1(x)
        outx = self.conv2_1(outx)
        outx = self.conv3_1(outx)
        outx = self.conv4(outx)
        outx = outx.view(outx.size(0), -1)
        
        outy = self.conv1_2(y)
        outy = self.conv2_2(outy)
        outy = self.conv3_2(outy)
        
        outz = self.conv1_3(z)
        outz = self.conv2_3(outz)
        outz = self.conv3_3(outz)
        
        oyz=torch.cat([outy,outz],1)
        
        oyz = self.conv5(oyz)
        oyz = oyz.view(oyz.size(0), -1)
                
        oo=torch.cat([outx,oyz],1)             
        out = self.fc1(oo)
           
        return out


# ## **Model train**

# In[36]:


#Create model
model = CNN()

use_gpu = torch.cuda.is_available()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

if use_gpu:    
    model = model.cuda()
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
model


# In[37]:


#Hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


# In[38]:


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


# In[39]:


#train fit
losses = []

for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_dl): 
        
        inputs = to_var(inputs)
#         inputs2 = to_var(inputs2)
#         inputs3 = to_var(inputs3)
        targets = to_var(targets)
        
        inputs1=inputs[:,0,:,:]
        inputs1=inputs1.resize(inputs1.shape[0],1,nsz, nsz//2)
        inputs2=inputs[:,1,:,:]
        inputs2=inputs1.resize(inputs2.shape[0],1,nsz, nsz//2)
        inputs3=inputs[:,2,:,:]
        inputs3=inputs1.resize(inputs3.shape[0],1,nsz, nsz//2)
        
        # forwad pass
        optimizer.zero_grad()
        outputs = model(inputs1,inputs2,inputs3)

        # loss
        loss = criterion(outputs, targets)
        losses += [loss.item()]

        # backward pass
        loss.backward()
        
        # update parameters
        optimizer.step()
        
        # report
        if (i + 1) % 50 == 0:
            print('Epoch[%2d/%2d],Step[%3d/%3d],Loss: %.4f'% (epoch + 1,num_epochs,i+1,len(train_ds)//batch_size,loss.item()))


# ## **Evaluation**

# In[45]:


#Visualizing model performance
plt.figure(figsize=(12, 4))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.title('Cross Entropy Loss');
plt.show()


# In[46]:


#Evaluate model performance
def evaluate_model(model, dataloader):
    # for batch normalization layers
    model.eval() 
    corrects = 0
    for inputs, targets in dataloader:
        inputs, targets = to_var(inputs, True), to_var(targets, True)
#         targets = to_var(targets)nsz, nsz//2
        
        inputs1=inputs[:,0,:,:]
        inputs1=inputs1.resize(inputs1.shape[0],1,nsz, nsz//2)
        inputs2=inputs[:,1,:,:]
        inputs2=inputs1.resize(inputs2.shape[0],1,nsz, nsz//2)
        inputs3=inputs[:,2,:,:]
        inputs3=inputs1.resize(inputs3.shape[0],1,nsz, nsz//2)
        
        outputs = model(inputs1,inputs2,inputs3)
        _, preds = torch.max(outputs.data, 1)
        corrects += (preds == targets.data).sum()
        
    zz=len(dataloader.dataset)
    
    print('accuracy: {:.2f}'.format(100. * corrects / len(dataloader.dataset)))
    print('corrects: {:.2f}'.format(corrects))
    print('Toatal: {:.2f}'.format(zz))   


# In[47]:


#Save model
evaluate_model(model, train_dl)
evaluate_model(model, test_dl)
torch.save(model.state_dict(), 'ECNN_wights.pth')


# In[48]:


#Model path
model_path='ECNN_wights.pth'
use_gpu=torch.cuda.is_available()


# In[49]:


print('CPU Mode Acitavted')
state_dict = torch.load(model_path,map_location='cpu')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] 
    new_state_dict[name] = v

# load params
model.load_state_dict(new_state_dict, strict=False)


# In[50]:


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


# In[51]:


#data path
original_path1= 'Data/test/VIS/VIS_107.jpg'  
original_path2= 'Data/test/IR/IR_107.jpg'


# In[52]:


#data augmentation
tfms1 = transforms.Compose([
    transforms.Resize((nsz, nsz//2)), 
    transforms.ToTensor(), 
    transforms.Normalize([0.45 ], [0.1])
])

tfms2 = transforms.Compose([
    transforms.Resize((nsz, nsz//2)),
    transforms.ToTensor(),
    transforms.Normalize([ 0.050], [ 0.09])
])
tfms3 = transforms.Compose([
    transforms.Resize((nsz, nsz//2)),
    transforms.ToTensor(),
    transforms.Normalize([0.06], [ 0.09])
])


# In[53]:


img1_org = Image.open(original_path1)
img2_org = Image.open(original_path2)
img1_org = np.asarray(img1_org)
img2_org = np.asarray(img2_org)
height=img1_org.shape[0]
width=img2_org.shape[1]

windows_size=32


# In[54]:


# stride should be set as 2 or 4 or 8 based on the size of input images
if width>= 500 and height>=500:
    factor=1
    stride=4
else:
    factor=2
    stride=8
    
dim1=(width, height)
dim2 = (int(width*factor), int(height*factor))        
img1 = cv2.resize(img1_org, dim2, interpolation = cv2.INTER_AREA)
img2 = cv2.resize(img2_org, dim2, interpolation = cv2.INTER_AREA)

kernel=np.array([[-1 , -2 , -1],  [0 , 0 , 0],  [1 , 2, 1]])


# In[62]:


#for rgb convert to gray:cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#img1_gray =cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
img1_gray =img1 #for gray image
img1_GY = cv2.filter2D(img1_gray,-1,kernel)
img1_GX = cv2.filter2D(img1_gray,-1,np.transpose(kernel))

#for rgb convert to gray: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
img2_gray =img2 #for gray image

img2_GY = cv2.filter2D(img2_gray,-1,kernel)
img2_GX = cv2.filter2D(img2_gray,-1,np.transpose(kernel))


# In[63]:


#gradient estimation
test_image1_1=img1_gray
test_image1_2=img1_GX
test_image1_3=img1_GY

test_image2_1=img2_gray
test_image2_2=img2_GX
test_image2_3=img2_GY


# In[64]:


#map generation
source1=img1
source2=img2

j=0
MAP=np.zeros([img1.shape[0], img1.shape[1]])
score1=0
score2=0
FUSED=np.zeros(test_image1_1.shape)

windowsize_r = windows_size-1
windowsize_c = windows_size-1

map1=np.zeros([img1.shape[0], img1.shape[1]])
map2=np.zeros([img2.shape[0], img2.shape[1]])

for r in tqdm(range(0,img1.shape[0] - windowsize_r, stride)):
    for c in range(0,img1.shape[1] - windowsize_c, stride):
        
        block_test1_1 = test_image1_1[r:r+windowsize_r+1,c:c+windowsize_c+1]
        block_test1_2 = test_image1_2[r:r+windowsize_r+1,c:c+windowsize_c+1]
        block_test1_3 = test_image1_3[r:r+windowsize_r+1,c:c+windowsize_c+1]
        
        block_test2_1 = test_image2_1[r:r+windowsize_r+1,c:c+windowsize_c+1]
        block_test2_2 = test_image2_2[r:r+windowsize_r+1,c:c+windowsize_c+1]
        block_test2_3 = test_image2_3[r:r+windowsize_r+1,c:c+windowsize_c+1]
        
        block1_1= np.concatenate((block_test1_1, block_test2_1), axis=0)
        block2_1= np.concatenate((block_test2_1, block_test1_1), axis=0)  
        block1_1 = Image.fromarray(block1_1, 'L')
        block2_1 = Image.fromarray(block2_1, 'L')
        block1_2= np.concatenate((block_test1_2, block_test2_2), axis=0)
        block2_2= np.concatenate((block_test2_2, block_test1_2), axis=0)  
        block1_2 = Image.fromarray(block1_2, 'L')
        block2_2 = Image.fromarray(block2_2, 'L')
        block1_3= np.concatenate((block_test1_3, block_test2_3), axis=0)
        block2_3= np.concatenate((block_test2_3, block_test1_3), axis=0)  
        block1_3 = Image.fromarray(block1_3, 'L')
        block2_3 = Image.fromarray(block2_3, 'L')
                 
        imout1_1=tfms1(block1_1)
        imout2_1=tfms1(block2_1)
        imout1_2=tfms2(block1_2)
        imout2_2=tfms2(block2_2)
        imout1_3=tfms3(block1_3)
        imout2_3=tfms3(block2_3)
        
        if use_gpu:
            imout1_1=to_var(imout1_1)
            imout2_1=to_var(imout2_1)
            imout1_2=to_var(imout1_2)
            imout2_2=to_var(imout2_2)
            imout1_3=to_var(imout1_3)
            imout2_3=to_var(imout2_3)
        
        imout1_1=(imout1_1)
        imout2_1=(imout2_1)
        imout1_2=(imout1_2)
        imout2_2=(imout2_2)
        imout1_3=(imout1_3)
        imout2_3=(imout2_3)       
        
        inputs1_1 = imout1_1.unsqueeze(0)
        inputs2_1 = imout2_1.unsqueeze(0)
        inputs1_2 = imout1_2.unsqueeze(0)
        inputs2_2 = imout2_2.unsqueeze(0)
        inputs1_3 = imout1_3.unsqueeze(0)
        inputs2_3 = imout2_3.unsqueeze(0)
        
        #modelevaluation
        model.eval()

        outputs1 = model(inputs1_1,inputs1_2,inputs1_3)
        _, predicted1 = torch.max(outputs1.data, 1)
        
        score1=predicted1.detach().cpu().numpy()

        model.eval()
        
        outputs2 = model(inputs2_1,inputs2_2,inputs2_3)
        _, predicted2 = torch.max(outputs2.data, 1)
        
        score2=predicted2.detach().cpu().numpy()
        
        map2[r:r+windowsize_r+1,c:c+windowsize_c+1] += 1
        
        if score1 <= score2:
            map1[r:r+windowsize_r+1,c:c+windowsize_c+1] += +1 
    else:
            map1[r:r+windowsize_r+1,c:c+windowsize_c+1] += -1            


# In[67]:


map1 = cv2.resize(map1, dim1, interpolation = cv2.INTER_AREA)
test_image1 = img1_org
test_image2 = img2_org

map3=np.zeros([img1_org.shape[0], img2_org.shape[1]])
FUSED=np.zeros(img1_org.shape)

for r in range(0,img1_org.shape[0], 1):
    for c in range(0,img1_org.shape[1], 1):           
        if map1[r,c] < 0:
            map3[r,c] =0
            FUSED[r,c]=img2_org[r,c]            
        else:
            map3[r,c] =1
            FUSED[r,c]=img1_org[r,c]
            
FUSED_8 = FUSED.astype(np.uint8)


# ## **Display**

# In[72]:


#Show output
plt.rcParams['savefig.dpi'] = 300
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12,10))

ax0.imshow(map1, cmap='viridis')
ax0.set_title("Silence map image")
ax0.axis('off')

ax1.imshow(FUSED_8, cmap='gray')
ax1.set_title("Fused image")
ax1.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
fig.tight_layout()

#save output
imageio.imwrite('Data/output.png', FUSED_8)


# In[74]:


#Fused results
plt.style.use('seaborn-white')
plt.rcParams['savefig.dpi'] = 300
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(14, 12))

ax0.imshow(img1, cmap='gray')
ax0.set_title("Left focus image")
ax0.axis('off')

ax1.imshow(img2, cmap='gray')
ax1.set_title("Right focus image")
ax1.axis('off')

ax2.imshow(FUSED_8, cmap='gray')
ax2.set_title("Fused image")
ax2.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
fig.tight_layout()
plt.show();


# ## **------------------END----------------------------**

# In[ ]:




