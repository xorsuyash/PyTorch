import torch 

x=torch.randn(3,requires_grad=True)
#print(x)

y=x+2
#print(y)

z=y*y*2
#print(z)

z=z.mean()
#print(z)
z.backward()
#print(x.grad)












a=torch.randn(3,requires_grad=True)

print(a)
a.requires_grad_(False)
print(a)


#x.requires_grad_(False)
#x.detach()
#with torch.no_grad()
