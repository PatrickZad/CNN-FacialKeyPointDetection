from sklearn.preprocessing import Imputer
import pandas as pd
import torch.utils.data as Data
import datapocess
import torch
import torch.nn as nn
from  detection_nn import DetecNN
from torch.autograd import Variable
'''
data split
'''


class Dataset0(Data.Dataset):
    def __init__(self, df):
        self.__points = self.__data_prepocess(df)
        self.__originimage = df.iloc[:, -1]

    def __data_prepocess(self, dataOrigin):
        '''
        left_eye_center , 
        right_eye_center , 
        nose_tip , 
        mouth_center_bottom
        '''
        dataPart = dataOrigin.loc[:, ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x',
                                      'right_eye_center_y', 'nose_tip_x', 'nose_tip_y',
                                      'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']]
        imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imr = imr.fit(dataPart)
        return imr.transform(dataPart.values)  # numpy ndarray
    def __len__(self):
        return len(self.__originimage)

    def __getitem__(self, idx):
        data = (
            torch.unsqueeze(datapocess.trainingImage2tensor(
                self.__originimage.iloc[idx]
            ), 0).type(torch.FloatTensor),
            torch.from_numpy(self.__points[idx]).type(torch.FloatTensor)
        )
        return data


def run_task(origindf):
    network = DetecNN(8).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    train_loader = Data.DataLoader(dataset=Dataset0(
        origindf), batch_size=50, shuffle=True)
    for step, (x, y) in enumerate(train_loader):
        batch_x = Variable(x).cuda()
        batch_y = Variable(y).cuda()
        output = network(batch_x)[0]
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss:%.4f' % loss.data.item())

df = pd.read_csv('./training.csv')
run_task(df)
