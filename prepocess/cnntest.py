import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
EPOCH = 1
BATCH_SIZE = 50
RATE = 0.001
DOWNLOAD_MNIST = False
train_data = torchvision.datasets.MNIST(
    root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
print(train_data.train_data.size())
print(train_data.train_labels.size())
'''
plt.imshow(train_data . train_data[0] . numpy(), cmap='gray')
plt.title(str(train_data.train_labels[0]))
plt.show()
'''
train_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)
                  ).type(torch.FloatTensor)[:2000]/255  # 原本灰度生成的是intTensor，归一化要使其为FloatTensor
test_y = test_data.test_labels[:2000]
'''
CNN
'''


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 2, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32*6*6,  10)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(x0)
        x2 = x1.view(x.size(0), -1)
        output = self.out(x2)
        return output, x2


if torch.cuda.is_available():
    network = MyCNN().cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            batch_x = Variable(x).cuda()
            batch_y = Variable(y).cuda()
            output = network(batch_x)[0]
            loss = loss_func(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step %100==0 :
                test_out, last_layer = network(test_x.cuda())
                pred_y = torch.max(test_out, 1)[1].data.squeeze()
                accuracy = (pred_y == test_y.cuda()).sum().item()/float(test_y.size(0))
                print('Epoch:', epoch, '|loss:%.4f' %
                      loss.data.item(), '|accuracy: %.2f' % accuracy)
