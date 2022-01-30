#https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

import pickle
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Net(nn.Module):
   def __init__(self,input_dim, output_dim):
       super(Net, self).__init__()
       self.hidden1 = torch.nn.Linear(input_dim, 128, bias=True)
       self.act1 = torch.nn.Tanh()
       self.hidden2 = torch.nn.Linear(128, output_dim, bias=True)
       self.act2 = torch.nn.Tanh()

   def forward(self, x):
       x = self.hidden1(x)
       x = self.act1(x)
       x = self.hidden2(x)
       x = self.act2(x)
       return x

with open('12k.pkl','rb') as f:
    data = pickle.load(f)

Xstar = []
Xcurr = []
RE = []


for j in range(int(len(data))):
    Xstar.append(data[j]["xstar"])
    Xcurr.append(data[j]["x"])
    RE.append(data[j]["RE"])

Xstar = np.asarray(Xstar)
Xcurr = np.asarray(Xcurr)
RE = np.asarray(RE)

num_dim_x = 4
input_dim = 2 * num_dim_x
output_dim = 1

Xstar = Xstar.reshape(len(Xstar),num_dim_x)
Xcurr = Xcurr.reshape(len(Xcurr),num_dim_x)
RE = RE.reshape(len(RE))

X = np.concatenate((Xstar,Xcurr),axis=1)

x = torch.from_numpy(X).float()
y = torch.from_numpy(RE.reshape(-1,1)).float()




net = Net(input_dim, output_dim)
print(net)


# Define Optimizer and Loss Function
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()


loss_values = []
epochs =[]

# First Shuffle (X, Y)
permutation = list(np.random.permutation(y.shape[0]))
shuffled_X = x[permutation, :]
shuffled_Y = y[permutation, :]

mini_batch_size = 100
num_complete_minibatches = np.int(np.floor(y.shape[0]/mini_batch_size))


#https://discuss.pytorch.org/t/validation-and-training-loss-per-batch-and-epoch/108308/2
#https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

#Notes : print(f"t: \n {t} \n")

for i in range(1000):
    loss_value = 0
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[mini_batch_size * (k):mini_batch_size * (k + 1), :]
        mini_batch_Y = shuffled_Y[mini_batch_size * (k):mini_batch_size * (k + 1), :]
        #print(mini_batch_X.shape, mini_batch_Y.shape)

        optimizer.zero_grad()

        prediction = net(mini_batch_X)
        loss = loss_func(prediction, mini_batch_Y)
        loss_value += loss.item()
        loss.backward()
        optimizer.step()

    print(loss_value)
    loss_values.append(loss_value)
    epochs.append(i + 1)

plt.plot(epochs,loss_values)
plt.xlabel("# of epochs")
plt.ylabel("Loss")
plt.title("Loss for different number of epochs")

