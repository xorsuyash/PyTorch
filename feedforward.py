import torch 
import torch.nn as nn 
import torchvision
import torchvision.transforms as transform 
import matplotlib.pyplot as plt 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#hyperparameters 

input_size= 784
hidden_size =100
num_classes=10
num_epochs=2
batch_size=100
learning_rate=0.001

train_dataset= torchvision.datasets.MNIST(root='./data',train=True,transform=transform.ToTensor(),download=True)
test_dataset=torchvision.datasets.MNIST(root='./data',train=False,transform=transform.ToTensor())


#Data loader


train_dataset = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


examples = iter(test_loader)

example_data,example_target=next(examples)

#for i in range(6):
    #plt.subplot(2,3,i+1)
    #plt.imshow(example_data[i][0],cmap='gray')
#plt.show()



class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.input_size=input_size
        self.l1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(hidden_size,num_classes)

    def forward(self,x):

        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)

        return out 
    
model= NeuralNet(input_size, hidden_size,num_classes)

criterian=nn.CrossEntropyLoss()


parameters=next(model.parameters())







         

