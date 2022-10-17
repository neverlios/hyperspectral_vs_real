from calendar import firstweekday
from os import listdir
from os.path import isfile, join
import os 
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


path = '../../../images_greyworld/cam'


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
        files = [f for f in listdir(path) if isfile(join(path, f))]
        img_name = os.path.join(self.root_dir,
                                files[idx])
        image = cv2.imread(img_name, cv2.IMREAD_ANYCOLOR |
                     cv2.IMREAD_ANYDEPTH).astype('float64')
        image = image / np.max(image)
        landmarks = self.ground_truth.iloc[idx]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float64')
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'landmarks': landmarks}
        return sample


cam_dataset = HypDataset(csv_file='./answers.csv',
                        root_dir=path,
                        transform = tfs.Compose([
                                                tfs.ToTensor(),
                                                tfs.RandomHorizontalFlip(0.5),
                                                tfs.RandomRotation(30),
                                                # tfs.GaussianBlur((5,9)),
                                                tfs.RandomCrop((64)),
                                                # tfs.Resize((512,512)),
                                                # tfs.Normalize((0), (65535))
    ]))

train_size = int(0.8 * len(cam_dataset))
test_size = len(cam_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(cam_dataset,
                                                    [train_size, test_size])
train_size = int(0.8 * train_size)
val_size = int(len(cam_dataset) - train_size - test_size)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, 
                                                    [train_size, val_size])
batch = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch,
                        shuffle=True, num_workers=0, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch,
                        shuffle=True, num_workers=0, drop_last=True)
loaders = {"train": train_dataloader, "valid": val_dataloader}
test_dataloader = DataLoader(test_dataset, batch_size=batch,
                        shuffle=True, num_workers=0, drop_last=True)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 16, 1)

        self.fc1 = nn.Linear(256, 400) 
        self.fc2 = nn.Linear(400, 3)
        
        self.ap1 = nn.AvgPool2d((64, 64))

    def forward(self, x):
        x_g = torch.nn.Upsample((64, 64))(self.ap1(x))
        x = torch.cat((x, x_g), axis=1)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2,2))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = torch.nn.Dropout()(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().double().to(device)
# criterion = nn.CosineEmbeddingLoss()
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-5)
criterion_loss = nn.CosineEmbeddingLoss(reduction='none')


def maxim(y_true, y_pred):  
    return tf.math.reduce_max(tf.clip_by_value(
        tf.compat.v1.losses.cosine_distance(y_true, y_pred, axis=1), -1, 1))
print(torch.cuda.is_available())


def disspersion(y_true, y_pred):
    return tf.math.sqrt(
        tf.math.reduce_sum(
            (y_pred-y_true)  ** 2 / (tf.cast(tf.shape(y_true), tf.float32))))

max_epochs = 30
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
                train_err = torch.cat((train_err.double(), errror)) 
            else:
                net.eval()
                with torch.no_grad():
                    outp = net(x_batch)
                    errror = criterion_loss(outp, torch.squeeze(y_batch),
                                            torch.ones(batch).to(device))
                    val_err = torch.cat((val_err.double(), errror))
        if k =='train':
            print('train:', torch.mul(torch.acos(1 - torch.mean(train_err)),
                                         180 / torch.pi).item(), end='\t')
        else:
            print('validation:', torch.mul(torch.acos(1 - torch.mean(val_err)),
                                                     180 / torch.pi).item())
torch.save(net.state_dict(), './net.pth')

net.eval()
test_err = torch.empty(0).to(device)
for data in test_dataloader:
            x_batch, y_batch = data.values()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.no_grad(): 
                outp = net(x_batch)
                errror = criterion_loss(outp, torch.squeeze(y_batch),
                                         torch.ones(batch).to(device))
                test_err = torch.cat((test_err.double(), errror))

print('test:', torch.mul(torch.acos(1 - torch.mean(test_err)), 180 / torch.pi).item())
