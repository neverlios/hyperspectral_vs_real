from calendar import firstweekday
from os import listdir
from os.path import isfile, join
import os 
import cv2 
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


path = './SimpleCube++/test/PNG'


class HypDataset(Dataset):  
    def __init__(self, csv_file, root_dir, transform=None) -> None:
        self.ground_truth = pd.read_csv(csv_file)[['mean_r', 'mean_g', 'mean_b']]
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
        image = cv2.resize(image, (648,432))
        image = np.clip(image, 2048, np.inf) - 2048
        image = image[:, :430, :430]
        landmarks = self.ground_truth.iloc[idx]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float64')
        if self.transform is not None:
            image = self.transform(image)
        sample = {'image': image, 'landmarks': landmarks}
        return sample


cpp_dataset = HypDataset(csv_file='./SimpleCube++/test/gt.csv', 
                        root_dir=path, 
                        transform = tfs.Compose([
                                                tfs.ToTensor(),
                                                tfs.Resize((64,64)),
                                                tfs.Normalize((0), (65535))
    ]))

batch = 8

test_dataloader = DataLoader(cpp_dataset, batch_size=batch,
                        shuffle=False, num_workers=0, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().double().to(device)
net.load_state_dict(torch.load('./net.pth', map_location=torch.device('cpu')))
net.eval()

criterion_loss = nn.CosineEmbeddingLoss(reduction='none')
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
