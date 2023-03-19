import torch 
import torch.nn as nn 
import numpy as np 

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)
x=np.array([2.0,1.0,0.1])

print(softmax(x))

a=torch.tensor([2.0,1.0,0.1])

output=torch.softmax(a,dim=0)
print("torch:", output)