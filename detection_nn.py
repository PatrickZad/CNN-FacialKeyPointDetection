import torch
import torch.nn as nn
from torch.autograd import Variable
class DetecNN(nn.Module):
    def __init__(self,outsize):
        super(DetecNN, self).__init__()
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
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.out = nn.Linear(512, outsize)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.conv2(x0)
        x0 = self.conv3(x0)
        x1=x0.view(x0.size(0),-1)
        x2=self.conn(x1)
        out=self.out(x2)
        return out,x1