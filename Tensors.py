import torch 
x=torch.rand(2,3)
y=torch.rand(2,3)
#print(x)
#print(y)
z=x+y
z=torch.add(x,y)
#torch.div/.add/.mul/.sub performs basic operations 
#add_/mul_ performs elementwise multiplication
#print(z)
a=y.mul_(x)
#print(a)

#print(x[:,0])



#reshaping 
a=torch.rand(4,4)
b=a.view(16)
#print(a)
#print(b)
#print(b.size())

import numpy as np 
d=torch.ones(5)
print(d)
e=d.numpy()
print(e)
d.add_(1)
print(d)
print(e)
#if they are in cpu instead of cpu they share same memory location


if torch.cuda.is_available():
    device=torch.device("cuda")
    x=torch.ones(5,device=device)
    y=torch.ones(5)
    y=y.to(device)
    z=x+y
    print(z)
    z=z.to("cpu")
    f=z.numpy()
    print(type(f))


#here we have moved the operations to gpu which is must faster than cpu
#then we converted back the gpu to cpu because numpy array dont support gpu 
