import pickle
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random

torch.manual_seed(0)  #for reproducibility


class Net(nn.Module):
   def __init__(self,input_dim, output_dim):
       super(Net, self).__init__()
       self.hidden1 = torch.nn.Linear(input_dim, 128, bias=True)
       self.act1 = torch.nn.Tanh()
       # self.hidden2 = torch.nn.Linear(128, 128, bias=True)
       # self.act2 = torch.nn.Tanh()
       # self.hidden3 = torch.nn.Linear(128, 128, bias=True)
       # self.act3 = torch.nn.Tanh()
       # self.hidden4 = torch.nn.Linear(128, 128, bias=True)
       # self.act4 = torch.nn.Tanh()
       self.hidden5 = torch.nn.Linear(128, output_dim, bias=True)
       #self.act5 = torch.nn.Tanh()


   def forward(self, x):
       x = self.hidden1(x)
       x = self.act1(x)
       # x = self.hidden2(x)
       # x = self.act2(x)
       # x = self.hidden3(x)
       # x = self.act3(x)
       # x = self.hidden4(x)
       # x = self.act4(x)
       x = self.hidden5(x)
       #x = self.act5(x)
       return x

def Data(length,num_dim_x,datafile):

    Xstar = []
    Xcurr = []
    RE = []

    data = datafile[:,0:length]

    for j in range(length):
        Xstar.data[j]["xstar"]
        Xcurr.data[j]["x"]
        RE.data[j]["RE"]

    Xstar = np.asarray(Xstar)
    Xcurr = np.asarray(Xcurr)
    RE = np.asarray(RE)


    Xstar = Xstar.reshape(len(Xstar), num_dim_x)
    Xcurr = Xcurr.reshape(len(Xcurr), num_dim_x)
    RE = RE.reshape(len(RE), 1)

    X = np.concatenate((Xstar, Xcurr), axis=1)
    inputs = torch.from_numpy(X).float()
    outputs = torch.from_numpy(RE).float()

    dataset = TensorDataset(inputs, outputs)

    train_set, val_set = torch.utils.data.random_split(dataset, [int(3/4 * (len(dataset))), int(1/4 * (len(dataset)))])

    batch_size_train = len(train_set)
    batch_size_test = len(val_set)
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(val_set, batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss #.item()

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

# def data_sets(num_test,num_dim_x,num_dim_control):
#     np.random.seed(1024)
#     v_l = 1.
#     v_h = 2.
#     np.random.seed(1)
#
#     X_MIN = np.array([-5., -5., -np.pi, v_l]).reshape(-1, 1)
#     X_MAX = np.array([5., 5., np.pi, v_h]).reshape(-1, 1)
#
#     lim = 1.
#     XE_MIN = np.array([-lim, -lim, -lim, -lim]).reshape(-1, 1)
#     XE_MAX = np.array([lim, lim, lim, lim]).reshape(-1, 1)
#
#     U_MIN = np.array([-3., -3.]).reshape(-1, 1)
#     U_MAX = np.array([3., 3.]).reshape(-1, 1)
#
#     def sample_xef():
#         return (X_MAX - X_MIN) * np.random.rand(num_dim_x, 1) + X_MIN
#
#     def sample_x(xref):
#         xe = (XE_MAX - XE_MIN) * np.random.rand(num_dim_x, 1) + XE_MIN
#         x = xref + xe
#         x[x > X_MAX] = X_MAX[x > X_MAX]
#         x[x < X_MIN] = X_MIN[x < X_MIN]
#         return x
#
#     def sample_uref():
#         return (U_MAX - U_MIN) * np.random.rand(num_dim_control, 1) + U_MIN
#
#     def sample_full():
#         xref = sample_xef()
#         uref = sample_uref()
#         x = sample_x(xref)
#         return (x, xref, uref)
#
#     #X_tr = [sample_full() for _ in range(num_train)]
#     X_te = [sample_full() for _ in range(num_test)]
#
#     x = []
#     xref = []
#
#     for id in range(len(X_te)):
#         x.append(torch.from_numpy(X_te[id][0]).float())
#         xref.append(torch.from_numpy(X_te[id][1]).float())
#
#     x, xref = (torch.stack(d).detach() for d in (x, xref))
#
#     return x, xref



# ##### test points for which RE is negative
#
# num_test = 12000
# num_dim_x = 4
# num_dim_control = 2
#
# x_test,xref_test = data_sets(num_test,num_dim_x, num_dim_control)
#
# X_test = torch.concat((x_test.reshape(num_test,num_dim_x),xref_test.reshape(num_test,num_dim_x)), dim = 1)
#
# #X_test = np.concatenate((Xclosed,Xstar),axis=1).reshape(Xstar.shape[0],2* Xstar.shape[1])
# #X_test = torch.tensor(X_test)
#
# with torch.no_grad():
#         pred = model(X_test)
#
# RE_n_pts = X_test[torch.where(pred < 0)[0],:]
#
# plt.scatter(X_test[:,0].numpy(),X_test[:,4].numpy(),marker='^',c="k")
# plt.scatter(RE_n_pts[:,0].numpy(),RE_n_pts[:,4].numpy(),marker='o',c="g")
# plt.xlabel("x")
# plt.ylabel("xstar")
# plt.title("Landscape of negative Riemmanian Energy")
# plt.legend(["All data points","Data points of negative RE"])
#
# plt.savefig('input_land_closed.png')
