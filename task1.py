import pandas as pd
import torch.utils.data as Data
import datapocess
import torch
import torch.nn as nn
from detection_nn import DetecNN
from torch.autograd import Variable


class Dataset(Data.Dataset):
    def __init__(self, df):
        self.__points, self.__originimage = self.__data_prepocess(df)

    def __data_prepocess(self, dataOrigin):
        '''
        left_eye_inner , 
        right_eye_inner , 
        left_eye_outter , 
        right_eye_outter , 
        '''
        dataPart = dataOrigin.loc[:, ['left_eye_inner_corner_x', 'left_eye_inner_corner_y', 'left_eye_outer_corner_x',
                                      'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
                                      'right_eye_outer_corner_x', 'right_eye_outer_corner_y', 'Image']]
        nonnan = dataPart.dropna()  # DataFrame
        return nonnan.iloc[:, :-1], nonnan.iloc[:, -1]

    def __len__(self):
        return len(self.__originimage)

    def __getitem__(self, idx):
        data = (
            torch.unsqueeze(datapocess.trainingImage2tensor(
                self.__originimage.iloc[idx]
            ), 0).type(torch.FloatTensor),
            torch.Tensor(list(self.__points.iloc[idx, :])).type(
                torch.FloatTensor)
        )
        return data

def pre_task(origindf):
    dataset=Dataset(origindf)
    outsize=8
    return outsize,dataset
def run_task(origindf):
    network = DetecNN(8).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    train_loader = Data.DataLoader(dataset=Dataset(
        origindf), batch_size=50, shuffle=True)
    for step, (x, y) in enumerate(train_loader):
        batch_x = Variable(x).cuda()
        batch_y = Variable(y).cuda()
        output = network(batch_x)[0]
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss<100:
            return network

'''
df = pd.read_csv('./training.csv')
run_task(df)
'''
