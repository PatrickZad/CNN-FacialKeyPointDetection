import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csv_data = pd.read_csv('./training.csv')
#print(csv_data.columns)
#print(len(csv_data.iloc[0, -1]))
#print(type(csv_data.iloc[0, 0]))
#x0=csv_data.iloc[0,:-1]
#print(type(x0))

def trainingImage2numpy(imagestr):
    pixelstrlist = imagestr.split(' ')
    pixel_array = np.array([int(pixel) for pixel in pixelstrlist])
    return pixel_array.reshape((96, 96))


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
        self.csv_data=pd.read_csv(csv_file)
    def __len__(self):
        return len(self.csv_data)
    def __getitem__(self,idx):
        data=(trainingImage2numpy(self.csv_data.iloc[idx,-1]),self.csv_data.iloc[idx,:-1])
        return data

