import numpy as np 

#f=2*x

X=np.array([1,2,3,4],dtype=np.float32)
Y=np.array([2,4,6,8],dtype=np.float32)

w=0.0

def forward(x):
    return w*x
def loss(y,y_hat):
    return ((y-y_hat)**2).mean()

def gradient(x,y_hat,y):
    return np.dot(2*x,y_hat-y).mean()


learning_rate=0.01
epochs=10

for epoch in range(epochs):
    #prediction
    y_hat=forward(X)

    #loss 
    Loss= loss(Y,y_hat)

    #gradient 

    dw=gradient(X,y_hat,Y)

    #update parameters 

    w=w-learning_rate*dw

    if epoch%1 ==0:
        print(f'epoch {epoch+1},loss:-{Loss:.8f},weight:-{w:.3f}')


print(f'predicted value is {forward(5)}')


