#!/usr/bin/env python
# coding: utf-8

# ## **Endoscop semantic segmentation convolutional DenseNets-Torch**

# In[125]:


import os
import cv2
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchsummary import summary
from torchvision.models import resnet18
import warnings

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[95]:


seed = 42
def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(seed)


# ## **Loading the Dataset**

# In[96]:


#Data load
class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, image_folder, mask_folder, transform=None):
        self.images = images
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.images[idx])
        mask_path = os.path.join(self.mask_folder, self.images[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask).unsqueeze(0)


# In[97]:


base_path = 'Endoscopy/train/'
image_folder = os.path.join(base_path, 'images')
mask_folder = os.path.join(base_path, 'masks')

images = os.listdir(image_folder)
masks = os.listdir(mask_folder)

train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
test_images, val_images = train_test_split(test_images, test_size=0.5, random_state=42)


# In[5]:


dataset_sizes = [len(train_images), len(test_images), len(val_images)]
labels = ["Train", "Test", "Val"]

plt.pie(dataset_sizes, labels = labels)
plt.show()


# In[6]:


#Data preparation
H,W=[128, 128]
nbatch_size=16

train_transform = A.Compose([A.Resize(H, W), A.HorizontalFlip(p= 0.5), A.VerticalFlip(p= 0.5), A.RandomRotate90(p= 0.5)])
val_transform = A.Compose([A.Resize(H, W), A.HorizontalFlip(p= 0.5), A.VerticalFlip(p= 0.5), A.RandomRotate90(p= 0.5)])
test_transform = A.Compose([A.Resize(H, W)])

trainset = Dataset(train_images, image_folder, mask_folder, transform= train_transform)
testset = Dataset(test_images, image_folder, mask_folder, transform= test_transform)
valset = Dataset(val_images, image_folder, mask_folder, transform= val_transform)

train_batch_size = nbatch_size
val_batch_size = nbatch_size
test_batch_size = nbatch_size

trainloader = torch.utils.data.DataLoader(trainset, batch_size= train_batch_size, num_workers= 2, shuffle= True)
valloader = torch.utils.data.DataLoader(valset, batch_size= val_batch_size, num_workers= 2, shuffle= True)
testloader = torch.utils.data.DataLoader(testset, batch_size= test_batch_size, num_workers= 2, shuffle= True)


# ## **Data visulizatin**

# In[183]:


#Display image
images, masks = next(iter(trainloader))
images = images.permute(0, 2, 3, 1)
masks = masks.permute(0, 2, 3, 1)
plt.figure(figsize=(18, 6))
for i in tqdm(range(6)):
    plt.subplot(2, 6, i+1)
    plt.axis("off")
    plt.imshow(images[i])

    plt.subplot(2, 6, i+7)
    plt.axis("off")
    plt.imshow(masks[i],cmap='tab20')
plt.show()


# ## **Designing the U-NET model**
# 

# In[178]:


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[16, 32, 64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.downs      = nn.ModuleList()
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.ups        = nn.ModuleList()
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.pool       = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Down part of U-net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        # Up part of U-net
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
            
    def forward(self, x):
        skip_connections = []
        
        # Go down the Unet
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Go up the Unet
        for idx in tqdm(range(0, len(self.ups), 2)):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            skip_connection = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](skip_connection)
        
        x = self.final_conv(x)
        return x    


# ## **Build-Up CNN Model**

# In[179]:


in_channels = 3 
out_channels = 1 

model = UNET()

print(model)


# ## **Summury CNN Model**

# In[180]:


#Model summury
summary(model, input_size=(3, H, W))


# ## **Loss function**

# In[181]:


class DICE_BCE_Loss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        intersection = 2*(logits * targets).sum() + self.smooth
        union = (logits + targets).sum() + self.smooth
        dice_loss = 1. - intersection / union
        loss = nn.BCEWithLogitsLoss() #nn.BCEWithLogitsLoss
        bce_loss = loss(logits, targets)

        return dice_loss + bce_loss
    
def dice_coeff(logits, targets):
    intersection = 2*(logits * targets).sum()
    union = (logits + targets).sum()
    if union == 0:
        return 1
    dice_coeff = intersection / union
    
    return dice_coeff.item()


# ## **Train Function**

# In[182]:


#train
def train(model, trainloader, optimizer, loss, epochs=10):
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        train_dice = 0
        for i, (images, masks) in enumerate(trainloader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            l = loss(logits, masks)
            l.backward()
            optimizer.step()
            train_loss += l.item()
            train_dice += dice_coeff(logits, masks)
        train_loss /= len(trainloader)
        train_dice /= len(trainloader)
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        
        #Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for i, (images, masks) in enumerate(valloader):
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                l = loss(logits, masks)
                val_loss += l.item()
                val_dice += dice_coeff(logits, masks)
        val_loss /= len(valloader)
        val_dice /= len(valloader)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        print(f"Epoch:{epoch + 1} Train Loss:{train_loss:.4f}|Train DICE Coeff:{train_dice:.4f}|Val Loss:{val_loss:.4f}|Val DICE Coeff: {val_dice:.4f}")
        
    return train_losses, train_dices, val_losses, val_dices


# ## **Hyperparameters and Training**

# In[171]:


#Train
epochs = 5
loss = DICE_BCE_Loss()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses, train_dices, val_losses, val_dices = train(model, trainloader, optimizer, loss, epochs)


# In[184]:


#Performance plot
plt.figure(figsize= (10, 6))
plt.subplot(1, 2, 1)
plt.plot(np.arange(epochs), train_dices)
plt.plot(np.arange(epochs), val_dices)
plt.xlabel("Epoch")
plt.ylabel("DICE Coeff")
plt.legend(["Train DICE", "Val DICE"])
plt.subplot(1, 2, 2)
plt.plot(np.arange(epochs), train_losses)
plt.plot(np.arange(epochs), val_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train Loss", "Val Loss"])


# ## **Testing the Model**

# In[189]:


#Model Evaluation
images, masks = next(iter(testloader))
with torch.no_grad():
    pred = model(images.to(device)).cpu().detach()
    pred = pred > 0.5

def display_batch(images, masks, pred):
    images = images.permute(0, 2, 3, 1)
    masks = masks.permute(0, 2, 3, 1)
    pred = pred.permute(0, 2, 3, 1)

    images = images.numpy()
    masks = masks.numpy()
    pred = pred.numpy()

    images = np.concatenate(images, axis=1)
    masks = np.concatenate(masks, axis=1)
    pred = np.concatenate(pred, axis=1)

    fig, ax = plt.subplots(3, 1, figsize=(20, 8))
    fig.tight_layout()
    ax[0].imshow(images)
    ax[0].axis('off')
    ax[0].set_title('Images')
    ax[1].axis('off')
    ax[1].imshow(masks, cmap= 'gray')
    ax[1].set_title('Masks')
    ax[2].imshow(pred, cmap='tab20')
    ax[2].axis('off')
    ax[2].set_title('Predictions')
    fig.tight_layout() 
    plt.show()


# In[190]:


# plotting images
display_batch(images, masks, pred)


# ## **Visualise predicted result**

# In[195]:


# Define function for plotting images
def show_random_images(i, img, output, true_mask):
    pred_mask  = torch.argmax(output, dim=1)    
    img = img[i]
    pred_mask = pred_mask[i]
    true_mask = true_mask[i]
    
    for i in tqdm(range(24)):
        pred_mask[0][i] = i
        true_mask[0][i] = i

    # Plot
    fig, axarr = plt.subplots(1,3, figsize=(12,8))
    axarr[0].imshow(img.permute(1,2,0).detach().numpy())
    axarr[0].axis('off')
    axarr[1].imshow(true_mask, cmap='gray')
    axarr[1].axis('off')
    axarr[2].imshow(pred_mask, cmap='gray')
    axarr[2].axis('off')
    fig.tight_layout() 
    plt.show()  


# In[196]:


# Get a batch from validation set
image, mask = next(iter(testloader))
pred = model(image.to(device)).cpu().detach()
mask=mask.permute(0, 2, 3, 1).detach().numpy()

show_random_images(1, image, pred, mask)


# In[ ]:




