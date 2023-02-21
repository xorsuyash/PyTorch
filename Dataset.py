import torch 
import torchvision 
from torch.utils.data import Dataset, DataLoader 
import numpy as np 
import math 



class WineDataset(Dataset):
    def __init__(self):
        df=np.loadtxt('./Data/Wine/wine.csv',delimiter=",",dtype=np.float32,skiprows=1)
        self.x= torch.from_numpy(df[:,1:])
        self.y=torch.from_numpy(df[:,[0]])
        self.n_samples = df.shape[0]

    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.n_samples

dataset=WineDataset()
data=DataLoader(dataset,batch_size=4,shuffle=True)
features,labels =next(iter(data))
print(features,labels)