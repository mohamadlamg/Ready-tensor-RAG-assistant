import torch
import torch.nn as nn
import torchvision
from  torch.utils.data import DataLoader,Dataset
import numpy as np
import math

class WineData(Dataset) :
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        return super().__getitem__(index)
    
model = torchvision.models.resnet18(pretrained=True)
print(model)
#model.fc = nn.Linear(100,10,bias=True)
