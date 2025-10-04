import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from  tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

#hyper paramètres
#model = torchvision.models.vgg
input_size = 784 #28*28
hidden_layers = 500
num_classes = 10
batch_size = 100
learning_rate = 0.001
epochs = 2

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])

#DATA
train_data = torchvision.datasets.MNIST(train=True,transform=transform,download=True,root='./datasets')

test_data = torchvision.datasets.MNIST(train=False,transform=transform,root='./datasets')

train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)

examples = iter(train_loader)

samples,labels = examples.__next__()
writer = SummaryWriter(f'runs/MNIST/trying_tensorboard')
for i in range(8):
    plt.subplot(3,3,i+1)
    plt.imshow(samples[1][0],cmap='gray')
#plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_layers,num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_layers)
        self.relu = nn.ReLU()
        self.l2 =  nn.Linear(hidden_layers,num_classes)

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out
    
model = NeuralNet(input_size,hidden_layers,num_classes)


#Loss et Optimizer

cirterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#Training 

n_total_steps = len(train_loader)
step = 0
for epoch in tqdm(range(epochs)):
    n_correct = 0
    for i,(images,labels) in enumerate(train_loader):
        images = images.reshape(-1,784)


        outputs = model(images)
        loss =  cirterion(outputs,labels)

        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()

        _,predictions = torch.max(outputs,1)
        n_samples = labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        running_train_acc = float(n_correct)/float(n_samples)

        writer.add_scalar('Trainig Loss',loss,global_step=step)
        writer.add_scalar('Trainig Accuracy',running_train_acc,global_step=step)
        step += 1

with torch.no_grad() :
    n_correct = 0
    n_samples = 0
    for images,labels in test_loader :
        images = images.reshape(-1,28*28) 
        outputs = model(images)
        _,predictions = torch.max(outputs,1)
        n_samples = labels.shape[0]
        n_correct += (predictions == labels).sum().item()



#file = 'models.pth'
#torch.save(model,file)
#model = torch.load(file)

torch.save(model.state_dict(),'mnist_ffd.pth')