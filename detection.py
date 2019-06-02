import task0
import task1
import task2
import task3
from detection_nn import DetecNN
import torch.utils.data as Data
import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
import datapocess

def run_task(origindf, pre,loss=100):
    network = DetecNN(pre[0]).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    train_loader = Data.DataLoader(dataset=pre[1], batch_size=50, shuffle=True)
    for step, (x, y) in enumerate(train_loader):
        batch_x = Variable(x).cuda()
        batch_y = Variable(y).cuda()
        output = network(batch_x)[0]
        currentloss = loss_func(output, batch_y)
        optimizer.zero_grad()
        currentloss.backward()
        optimizer.step()
        if currentloss < loss:
            return network


traindata = pd.read_csv('./training.csv')
testdata = pd.read_csv('./test.csv')
outdata=pd.read_csv('./Template.csv') 
'''
build network
'''
nn0 = run_task(traindata, task0.pre_task(traindata))
nn1 = run_task(traindata, task1.pre_task(traindata))
nn2 = run_task(traindata, task2.pre_task(traindata))
nn3 = run_task(traindata, task3.pre_task(traindata),160)
'''
record output
'''
for i in range(len(testdata)):
    baserowid=i*30
    image=torch.unsqueeze(torch.unsqueeze(
        Variable(datapocess.trainingImage2tensor(testdata.iloc[i,1])),0),
        0).type(torch.FloatTensor).cuda()
    out0=torch.squeeze(nn0(image)[0]).detach().cpu().numpy()
    for j in range(4):
        outdata.iloc[baserowid+j,-1]=out0[j]
    for j in range(4,6):
        outdata.iloc[baserowid+16+j,-1]=out0[j]
        outdata.iloc[baserowid+24+j,-1]=out0[j+2]
    out1=torch.squeeze(nn1(image)[0]).detach().cpu().numpy()
    for j in range(8):
        outdata.iloc[baserowid+4+j,-1]=out1[j]
    out2=torch.squeeze(nn2(image)[0]).detach().cpu().numpy()
    for j in range(8):
        outdata.iloc[baserowid+12+j,-1]=out2[j]
    out3=torch.squeeze(nn3(image)[0]).detach().cpu().numpy()
    for j in range(6):
        outdata.iloc[baserowid+22+j,-1]=out3[j]
    
outdata.to_csv('./subm.csv',index=0)
