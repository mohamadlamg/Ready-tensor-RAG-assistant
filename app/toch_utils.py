import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

#Load model



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
    
input_size = 784 #28*28
hidden_layers = 100
num_classes = 10
    
model = NeuralNet(input_size,hidden_layers,num_classes)

model.load_state_dict(torch.load('mnist_ffd.pth'))
model.eval()

def transform_images(images_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28,28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))])
    
    image = Image.open((io.BytesIO(images_bytes)))

    return transform(image).unsqueeze(0)

def get_predict(image_tensors):
    images = image_tensors.reshape(-1,28*28) 
    outputs = model(images)
    _,predictions = torch.max(outputs,1)

    return predictions

        