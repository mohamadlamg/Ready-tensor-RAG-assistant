import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import networkx 

import matplotlib.pyplot as plt

import os
os.environ['KERAS_BACKEND'] = 'torch'

# Maintenant vos autres imports
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

model = load_model("best_model.keras")

with open("test.p", "rb") as f:
    dataset = pickle.load(f)

with open("valid.p", "rb") as d:
    validation = pickle.load(d)


# Vérifier le type
print(type(dataset))
print(type(validation))



# chemin_complet = "/home/mohamadoucoulibaly04/machine_learning/best_models.keras"
# model = keras.models.load_model(chemin_complet)

X_val = validation['features']
y_val = validation['labels']

X_val = X_val/ X_val.max()


# #print(W.shape)

X = dataset['features']
y = dataset['labels']

X = X / X.max()

# num_classes = len(np.unique(y))
# print(num_classes)

# print('W',W.shape)
# print('Z',Z.shape)



# Si c'est un dict
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


score = model.evaluate(X_test,y_test,verbose=0)

print("Test loss :",score[0])
print("Test accuracy :",score[1])
