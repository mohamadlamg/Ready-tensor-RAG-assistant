import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import tqdm
from sklearn.metrics import accuracy_score,classification_report

X_num,y_num = make_regression(n_features=1,n_samples=300,noise=20,random_state=1)

X = torch.from_numpy(X_num.astype(np.float32))
y= torch.from_numpy(y_num.astype(np.float32))

y = y.view(y.shape[0],1)

n_samples,n_features = X.shape

input_size = n_features
output_size = 1

model = nn.Linear(input_size,output_size)

learning_rate = 0.01
lossing = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

epochs = 100
for epoch in range (100) :
    y_predicted = model(X)
    loss = lossing(y_predicted,y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

prediction = model(X).detach().numpy()



plt.plot(X_num,y_num,'ro')
plt.plot(X_num,prediction)
plt.show()