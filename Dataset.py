'''

epoch = 1 forward and backward passs 
batch_size = no of training samples in one forward paas and backward 
number_of_iterations = number of passes ,each pass using batch size 



'''


import torch
import torchvision 

from torch.utils.data import Dataset,DataLoader 
import numpy as np 
import math 


class WineDataset(Dataset):
    def __init__(self):
        df=np.loadtxt('./Data/Wine/wine.csv',delimiter=",",dtype=np.float32,skiprows=1)
        self.x =torch.from_numpy(df[:,1:])
        self.y=torch.from_numpy(df[:,[0]])
        self.n_samples=df.shape[0]
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.n_samples
dataset=WineDataset()

data=DataLoader(dataset=dataset,batch_size=4,shuffle=True)
#dummy trainig loop 

num_epoch=2
total_samples=len(dataset)
n_iterations = math.ceil(total_samples/4)









