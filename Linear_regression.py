 # 1) design model (input ,output, forward_pass)
# 2) construct loss and optimizer 
# 3) training loop 
#   -forward pass 
#   -backward pass 
#   -update weights 


import torch 
import torch.nn as nn 
import numpy as np 
from sklearn import datasets 
import matplotlib.pyplot as plt 

X_numpy,y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)

X=torch.from_numpy(X_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))
y=y.view(y.shape[0],1)
n_samples ,n_features=X.shape
#model 
n_input=n_features
n_output=n_features

class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        self.lin=nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.lin(x)

model=LinearRegression(n_input,n_output)

#model construction completed 

#loss 
loss=nn.MSELoss()

epochs=10000
learning_rate=0.01


optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(epochs):
    y_hat=model(X)
    l=loss(y,y_hat)

    l.backward()

    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1)%10==0:
        print(f'epoch: {epoch+1} , loss = {l.item():.4f}')

#plot

predicted = model(X).detach().numpy()

plt.plot(X_numpy,y_numpy,'ro')
plt.plot(X_numpy,predicted,'b')