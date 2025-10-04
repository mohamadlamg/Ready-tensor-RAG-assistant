import json
from nltk_utils import tokenize,stem,bag_of_word
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from model_chat import NeuralNet


with open('intents.json','r') as f :
    intents = json.load(f)

#print(intents)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

ignored_words =['?','!','.','"','/']
all_words = {stem(w) for w in all_words if w not in ignored_words}
all_words = sorted(set(all_words))
tags = sorted(set(tags))
#print(all_words)

X_train = []
y_train = []

for (pattern_sentence,tag) in xy :
    bag = bag_of_word(pattern_sentence,all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    

batch_size = 16
input_size = len(X_train[0])
#print(input_size)
hidden_layers = 32
outputs_size = len(tags)
learning_rate = 0.0005
num_epochs = 2000
dropout_rate = 0.3
hidden_size1 = 64    
hidden_size2 = 32 
dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset,batch_size=8,shuffle=True)

model = NeuralNet(input_size,hidden_layers,hidden_size1,hidden_size2,outputs_size,dropout_rate)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for(words,labels) in train_loader:
        outputs = model(words)
        loss = criterion(outputs,labels)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1)%100 == 0 :
        print(f'epoch{epoch+1}/{num_epochs} , Loss : {loss.item():.4f}')

    
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": outputs_size,
    "hidden_layers": hidden_layers,
    "all_words":all_words,
    "tags":tags,
    "hidden_size1":hidden_size1,
    "hidden_size2":hidden_size2,
    "dropout_rate":dropout_rate

}

file = "chatBot.pth"
torch.save(data,file)