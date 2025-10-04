import torch
import torch.nn as nn
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, input_size,hidden_layers,hidden_size1,hidden_size2,outputs_size,dropout_rate):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_layers)
        self.l2 = nn.Linear(hidden_layers,hidden_layers)
        self.l3 = nn.Linear(hidden_layers,hidden_layers)
        self.l4 = nn.Linear(hidden_layers,outputs_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
       
    def forward(self,x):
        out = self.l1(x)
        
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.l2(out)
        
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.l3(out)
        
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.l4(out)
        # Pas de softmax ici car CrossEntropyLoss l'inclut
        return out
    