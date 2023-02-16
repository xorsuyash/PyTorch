import torch 

X=torch.tensor([1.0,2.0,3.0,4.0],dtype=torch.float32,requires_grad=True)
Y=torch.tensor([2.0,4.0,6.0,8.0],dtype=torch.float32)
w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

def forward(x):
    return w*x

def loss(y,y_hat):
    return ((y-y_hat)**2).mean()


epochs=50
learning_rate=0.01

for epoch in range(epochs):

    #prediction 
    y_hat=forward(X)

    #loss 

    l=loss(Y,y_hat)

    l.backward()

    with torch.no_grad():
        w-= learning_rate*w.grad

    w.grad.zero_()


    if epoch%1 ==0:
        print(f'epoch {epoch+1},loss:-{l:.8f},weight:-{w:.3f}')
print(f'predicted value is :- {forward(5)}')

