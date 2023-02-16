# 1) design model (input ,output, forward_pass)
# 2) construct loss and optimizer 
# 3) training loop 
#   -forward pass 
#   -backward pass 
#   -update weights 


import torch 
import torch.nn as nn

X=torch.tensor([[1.0],[2.0],[3.0],[4.0]],dtype=torch.float32,requires_grad=True)
Y=torch.tensor([[2.0],[4.0],[6.0],[8.0]],dtype=torch.float32)
n_samples, n_features=X.shape
input_size=n_features
output_size=n_features

#model=nn.Linear(input_size,output_size)

class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        self.lin=nn.Linear(input_dim ,output_dim)

    def forward(self,x):
        return self.lin(x)

model=LinearRegression(input_size,output_size)






epochs=100
learning_rate=0.01

loss=nn.MSELoss()

optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(epochs):

    #prediction 
    y_hat=model(X)

    #loss 

    l=loss(Y,y_hat)

    l.backward()

    optimizer.step()

    optimizer.zero_grad() 


    if epoch%1 ==0:
        [w,b]=model.parameters()
        print(f'epoch {epoch+1},loss:-{l:.8f},weight:-{w[0][0].item():.3f}')
X_test=torch.tensor([5],dtype=torch.float32)
print(f'predicted value is :- {model(X_test).item()}')
