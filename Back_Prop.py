import torch 

x=torch.tensor(1.0,requires_grad=True)
w=torch.tensor(1.0,requires_grad=True)
y=torch.tensor(2.0,requires_grad=True)

y_hat=w*x
loss=(y-y_hat)**2

loss.backward()

print(w.grad)
