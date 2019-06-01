import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#csv_data = pd.read_csv('./training.csv')
# print(csv_data.columns)
#print(len(csv_data.iloc[0, -1]))
#print(type(csv_data.iloc[0, 0]))
# x0=csv_data.iloc[0,:-1]
# print(type(x0))


'''
image test
for imagestr in csv_data.iloc[:10, -1]:
    i = 0
    plt.imshow(trainingImage2numpy(imagestr), cmap='gray')
    plt.title('test'+str(i))
    i += 1
    plt.show()
'''


class KPdataset(Data.Dataset):
    def __init__(self, csv_file):
        self.csv_data = pd.read_csv(csv_file)

    def trainingImage2tensor(self, imagestr):
        pixelstrlist = imagestr.split(' ')
        pixel_array = np.array([int(pixel) for pixel in pixelstrlist])
        return torch.from_numpy(pixel_array.reshape((96, 96)))

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        data = (
            torch.unsqueeze(self.trainingImage2tensor(
            self.csv_data.iloc[idx, -1]),0).type(torch.FloatTensor), 
            torch.from_numpy(np.array(list(self.csv_data.iloc[idx, :-1]))).type(torch.FloatTensor)
            )
        return data


train_loader = Data.DataLoader(dataset=KPdataset(
    './training.csv'), batch_size=50, shuffle=True)


class KpDetNN(nn.Module):
    def __init__(self):
        super(KpDetNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conn = nn.Sequential(
            nn.Linear(11*11*128, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.out = nn.Linear(1024, 30)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.conv2(x0)
        x0 = self.conv3(x0)
        x1=x0.view(x0.size(0),-1)
        x2=self.conn(x1)
        out=self.out(x2)
        return out,x1

kpDetNet = KpDetNN().cuda()
optimizer = torch.optim.SGD(kpDetNet.parameters(), lr=0.001)
loss_func = nn.MSELoss()
for step, (x, y) in enumerate(train_loader):
    batch_x = Variable(x).cuda()
    batch_y = Variable(y).cuda()
    originout=kpDetNet(batch_x)
    output = originout[0]
    loss = loss_func(output, batch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print('loss:%.4f' % loss.data.item())
