import torch 
x=torch.rand(2,3)
y=torch.rand(2,3)
print(x)
print(y)
z=x+y
z=torch.add(x,y)
#torch.div/.add/.mul/.sub performs basic operations 
#add_/mul_ performs elementwise multiplication
print(z)
a=y.mul_(x)
print(a)

print(x[:,0])



#reshaping 
a=torch.rand(4,4)
b=a.view(16)
print(a)
print(b)
print(b.size())