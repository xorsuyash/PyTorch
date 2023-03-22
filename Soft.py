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


def cross_entropy(actual,predicted):
    loss = -np.sum(actual*np.log(predicted))
    return loss 

y=np.array([1,0,0])

y_pred_good=np.array([0.7,0.2,0.1])
y_pred_bad=np.array([0.1,0.3,0.5])

l1=cross_entropy(y,y_pred_good)
l2=cross_entropy(y,y_pred_bad)

print(f"good prediction : {l1}")
print(f"Bad loss : {l2}")


#loss in pytorch 

loss=nn.CrossEntropyLoss()

#nn loss automatically applies nn.LogSoftmax+nn.NLLLoss(negative log likelihood loss)\
#->no softmax in last layer 
#->y not one encoded 
#->y_pred no softmax here 

Y= torch.tensor([0])
# nsapmles*nclasses = 1x3

Y_pred_good= torch.tensor([[2.0,1.0,0.1]])
Y_pred_baad =torch.tensor([[0.5,2.0,0.3]])
l1=loss(Y_pred_good,Y)
l2=loss(Y_pred_baad,Y)

print(l1.item())
print(l2.item())


_,predictions1=torch.max(Y_pred_good,1)
_, predictions2=torch.max(Y_pred_baad,1)

print(predictions1)
print(predictions2)









