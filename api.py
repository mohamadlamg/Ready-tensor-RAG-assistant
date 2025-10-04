import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import os
os.environ['KERAS_BACKEND'] = 'torch'

# Maintenant vos autres imports
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Charger le fichier
with open("test.p", "rb") as f:
    dataset = pickle.load(f)

with open("valid.p", "rb") as d:
    validation = pickle.load(d)

# Vérifier le type
print(type(dataset))
print(type(validation))

X_val = validation['features']
y_val = validation['labels']

X_val = X_val/ X_val.max()


#print(W.shape)

X = dataset['features']
y = dataset['labels']

X = X / X.max()

num_classes = len(np.unique(y))
print(num_classes)

# print('W',W.shape)
# print('Z',Z.shape)



# Si c'est un dict
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# print(X_train.shape)
# print(y_train.shape)

# print(X_test.shape)
# print(y_test.shape)

def model1(height, width, channels):

   
    model = keras.models.Sequential([
        keras.layers.Input((height, width, channels)),
            
            # Normalisation des données
        #keras.layers.Rescaling(1./255),
            
            # Block 1
        keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Dropout(0.25),
            
            # Block 2
        keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Dropout(0.25),
            
            # Block 3
        keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Dropout(0.25),
            
            # Flatten
        keras.layers.Flatten(),
            
            # Classifier
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')  
    ])
    
    return model



model = model1(32,32,3)


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

bacth_size = 256
epochs = 50

early_stop = EarlyStopping(
    monitor='val_accuracy',   # on surveille la perte sur la validation
    patience=7,           # combien d’epochs d’attente avant d’arrêter
    restore_best_weights=True  # reprendre les poids du meilleur epoch
)

checkpoint = ModelCheckpoint(
    'best_model.keras',   # fichier de sauvegarde
    monitor='val_accuracy',  # on surveille l’accuracy validation
    save_best_only=True,  # on garde seulement le meilleur
    mode='max'
)

callbacks = [early_stop,checkpoint]

history = model.fit(X_train,y_train,batch_size=bacth_size,epochs=epochs,verbose=1,validation_data=(X_val,y_val),callbacks=callbacks)


# Courbe Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Courbe Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



score = model.evaluate(X_test,y_test,verbose=0)

print("Test loss :",score[0])
print("Test accuracy :",score[1])

# y_pred = model.predict(W)

# score_final = accuracy_score(y_pred,Z)

# print("Accuracy final .",score_final)