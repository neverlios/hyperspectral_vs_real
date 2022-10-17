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


class PoissonGaussianNoise:
    def __init__(self, alpha=0, beta=0, seed=0):
        self.alpha = alpha
        self.beta  = beta
        self.seed  = seed
        np.random.seed(self.seed)

    def __std(self, source_data):
        return np.sqrt(self.alpha * source_data + self.beta)

    def __call__(self, source_data):
        target_data = np.random.normal(source_data, self.__std(source_data))
        
        mean = np.mean(source_data)
        std  = np.std(target_data - source_data)
        
        snr = np.where( std==0, 0, mean / std)
        return target_data, snr

alpha, beta = 0.00005, 0.00005

noise_model = PoissonGaussianNoise(alpha=alpha, beta=beta)

# clear_src_tristim = src_tristim
# noised_src_tristim, snr = noise_model(clear_src_tristim)

# print(f'SNR = {snr}, max_value = {noised_src_tristim.max()}, min_value = {noised_src_tristim.min()}')

# noised_src_tristim = np.clip(noised_src_tristim, 0, 1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 16, 1)

        self.fc1 = nn.Linear(256, 400) 
        # self.fc2 = nn.Linear(1000, 500)
        # self.fc3 = nn.Linear(500, 400)
        self.fc4 = nn.Linear(400, 3)
        
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
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x.double()


path = '../../../images_greyworld/cam'
snr_glob = []

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
                         cv2.IMREAD_ANYDEPTH).astype(float)
        # max_value = np.max(image)
        image /= 65535
        image, snr = noise_model(image)
        snr_glob.append(snr)
        # print(f'SNR = {snr}, max_value = {image.max()}, min_value = {image.min()}')
        # image = np.clip(image, 2048.0/max_value, 1) - 2048/max_value
        image = np.clip(image, 0, 1)
        
        landmarks = self.ground_truth.iloc[idx]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float')
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'landmarks': landmarks}
        return sample


cpp_dataset = HypDataset(csv_file='./answers.csv', 
                        root_dir=path, 
                        transform = tfs.Compose([
                                                tfs.ToTensor(),
                                                tfs.Resize((64,64)),
    ]))

batch = 8

criterion_loss = nn.CosineEmbeddingLoss(reduction='none')

test_dataloader = DataLoader(cpp_dataset, batch_size=batch,
                        shuffle=False, num_workers=0, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().double().to(device)

net.load_state_dict(torch.load('./net_sc.pth', map_location=torch.device('cpu')))
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

snr_glob = np.array(snr_glob)
print('mean snr:', np.mean(snr_glob))
print('median snr:', np.median(snr_glob))

print('snr < 15:', len(snr_glob[snr_glob < 15]))
print('snr > 25:', len(snr_glob[snr_glob > 25]))
print('test:', torch.mul(torch.acos(1 - torch.mean(test_err)), 180 / torch.pi).item())
values, indices = torch.topk(test_err, 200)
print('200 worst = ', torch.mul(torch.acos(1 - torch.mean(values)), 180 / torch.pi).item())
values, indices = torch.topk(test_err, 1500, largest=False)
print('1500 best = ', torch.mul(torch.acos(1 - torch.mean(values)), 180 / torch.pi).item())
