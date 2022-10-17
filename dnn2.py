from calendar import firstweekday
from os import listdir
from os.path import isfile, join
import os 
from skimage import io, transform
import cv2
from numba import cuda 
import torch.optim as optim
import numpy as np
from count_answers import spawn
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as tfs
from numpy import dot
from numpy.linalg import norm
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


path = './SimpleCube++/train/PNG'

#class for creation dataset
class HypDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None) -> None:
        self.ground_truth = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.root_dir
        img_name = os.path.join(self.root_dir,
                                self.ground_truth.iloc[idx, 0]) + '.png'
        image = cv2.imread(img_name, cv2.IMREAD_ANYCOLOR | 
                        cv2.IMREAD_ANYDEPTH).astype('float32')
        image = np.clip(image, 2048, np.inf) - 2048
        image = image[:,:432]
        image = image / np.max(image)
        landmarks = self.ground_truth.iloc[idx, 1:].to_numpy(dtype=np.float32)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'landmarks': landmarks} 
        return sample

#initialising train dataset
cam_dataset = HypDataset(csv_file='./SimpleCube++/train/gt.csv',
                        root_dir=path,
                        transform = tfs.Compose([
                                                tfs.ToTensor(),
                                                tfs.RandomHorizontalFlip(0.5),
                                                # tfs.RandomPerspective(),
                                                # tfs.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
                                                # tfs.RandomAdjustSharpness(sharpness_factor=2),
                                                tfs.RandomRotation(30),
                                                # tfs.GaussianBlur((5,9)),
                                                # tfs.Resize(432),
                                                tfs.RandomCrop(64),
                                                # tfs.Resize((512,512)),
                                                # tfs.Normalize((0), (65535))
                                                ]))

train_size = int(0.8 * len(cam_dataset))
val_size = len(cam_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(cam_dataset,
                                                     [train_size, val_size])

#initialising test dataset
test_dataset = HypDataset(csv_file='./SimpleCube++/test/gt.csv',
                        root_dir='./SimpleCube++/test/PNG',
                        transform = tfs.Compose([
                                                tfs.ToTensor(),
                                                tfs.RandomHorizontalFlip(0.5),
                                                # tfs.RandomRotation(30),
                                                tfs.Resize(64),
                                                # tfs.Normalize((0), (65535))
                                                ]))

#batch for training and test
batch = 16

#making dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch,
                        shuffle=True, num_workers=0, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch,
                        shuffle=True, num_workers=0, drop_last=True)
loaders = {"train": train_dataloader, "valid": val_dataloader}

test_dataloader = DataLoader(test_dataset, batch_size=batch,
                        shuffle=False, num_workers=0, drop_last=True)

#net arcgitecture
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 16, 1)

        self.batch1 = nn.BatchNorm2d(16)
        self.batch2 = nn.BatchNorm2d(32)
        self.batch3 = nn.BatchNorm2d(64)
        self.batch4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(256, 400) 
        # self.fc2 = nn.Linear(1000, 500)
        # self.fc3 = nn.Linear(500, 400)
        self.fc4 = nn.Linear(400, 3)
        
        self.ap1 = nn.AvgPool2d((64, 64))

    def forward(self, x):
        x_g = torch.nn.Upsample((64, 64))(self.ap1(x))
        x = torch.cat((x, x_g), axis=1)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = self.batch1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = self.batch2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        x = self.batch3(x)
        x = F.max_pool2d(F.relu(self.conv4(x)), (2,2))
        x = self.batch4(x)
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = torch.nn.Dropout()(x)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
criterion_loss = nn.CosineEmbeddingLoss(reduction='none')#calculate angular loss
criterion = nn.L1Loss()#training loss
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-6)
# optimizer = optim.SGD(net.parameters(), lr=1e-3)
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    mode='min',
                                                    factor=0.1, 
                                                    patience=5, 
                                                    threshold=0.0001, 
                                                    threshold_mode='rel', 
                                                    cooldown=3, 
                                                    min_lr=1e-8, 
                                                    eps=5e-01, 
                                                    verbose=False
                                                    )
# scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
#                                     T_0=10, T_mult=2, eta_min=1e-8, last_epoch=-1)
# scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10,
#                                 eta_min=1e-8, last_epoch=- 1, verbose=False)

def maxim(y_true, y_pred):  
    return tf.math.reduce_max(tf.clip_by_value(
        tf.compat.v1.losses.cosine_distance(y_true, y_pred, axis=1), -1, 1))
print(torch.cuda.is_available())


def disspersion(y_true, y_pred):
    return tf.math.sqrt(
        tf.math.reduce_sum(
            (y_pred-y_true)  ** 2 / (tf.cast(tf.shape(y_true), tf.float32))))


max_epochs = 200
for epoch in range(max_epochs):
    train_err = torch.empty(0).to(device)
    val_err = torch.empty(0).to(device)
    for k, dataloader in loaders.items():
        for i, data in (enumerate(dataloader, 0)):
            x_batch, y_batch = data.values()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if k == "train":
                net.train()
                optimizer.zero_grad()
                outp = net(x_batch)
                y_batch = torch.squeeze(y_batch)
                loss = criterion(outp, y_batch)
                errror = criterion_loss(outp, y_batch, torch.ones(batch).to(device))
                loss.backward()
                optimizer.step()
                # print(train_err.dtype)
                # print(errror.dtype)
                train_err = torch.cat((train_err, errror.float()))
          
                
            else:
                net.eval()
                with torch.no_grad():
                    outp = net(x_batch)
                    errror = criterion_loss(outp, torch.squeeze(y_batch), 
                                            torch.ones(batch).to(device))
                    val_err = torch.cat((val_err, errror))
        if k =='train':
            print('train:', torch.mul(torch.acos(1 - torch.mean(train_err)),
                                         180 / torch.pi).item(), end='\t')
        else:
            print('validation:', torch.mul(torch.acos(1-torch.mean(val_err)),
                                                     180 / torch.pi).item())
    # scheduler.step(torch.mul(torch.acos(1-torch.mean(val_err)),
    #                                                  180 / torch.pi).item())
    scheduler.step(torch.mul(torch.acos(1-torch.mean(val_err)),
                                                     180 / torch.pi).item())
torch.save(net.state_dict(), './net_sc_bstch.pth')
# net.load_state_dict(torch.load('./net_sc.pth', map_location=torch.device('cpu')))

#test model
net.eval()
val_err = torch.empty(0).to(device)
for data in test_dataloader:
            x_batch, y_batch = data.values()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.no_grad(): 
                outp = net(x_batch)
                errror = criterion_loss(outp, torch.squeeze(y_batch), torch.ones(batch).to(device))
                val_err = torch.cat((val_err, errror))

print('test:', torch.mul(torch.acos(1-torch.mean(val_err)), 180 / torch.pi).item())