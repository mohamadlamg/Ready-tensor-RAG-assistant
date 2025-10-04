import torch
import torch.nn as nn
import random 
import json
from nltk_utils import tokenize,stem,bag_of_word
from model_chat import NeuralNet
import numpy as np

with open('intents.json','r') as f :
    intents = json.load(f)

file = "chatBot.pth"
data = torch.load(file)
input_size = data['input_size']
outputs_size = data['output_size']
hidden_layers = data['hidden_layers']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']
hidden_size1 = data['hidden_size1']
hidden_size2 = data['hidden_size2']
dropout_rate = data['dropout_rate']

model = NeuralNet(input_size,hidden_layers,hidden_size1,hidden_size2,outputs_size,dropout_rate)

model.load_state_dict(model_state)
model.eval()

bot_name = "Momo Bot"
print("Bienvenue Excellence !")
print("Je suis un Chatbot developpé par Mohamadou et nous pouvons avoir de causeries sympas.Pour m'arrêter dites stop !")

while True:
    sentence = input('Vous: ')
    if sentence == 'stop' or sentence == 'Stop':
        break

    sentence = tokenize(sentence)
    X = bag_of_word(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _,prediction = torch.max(output,dim=1)
    tag  = tags[prediction.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][prediction.item()]

    if prob.item() > 0.78 :

        for intent in intents['intents']:
            if tag == intent['tag'] :
                print(f"{bot_name}: {random.choice(intent['reponses'])}")

    else:
        print(f'{bot_name}:Votre question va au délà de ma reflexion ....')
