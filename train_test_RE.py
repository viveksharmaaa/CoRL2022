import pickle
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt

torch.manual_seed(0)  #for reproducibility


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
RE = RE.reshape(len(RE),1)

X = np.concatenate((Xstar,Xcurr),axis=1)
inputs = torch.from_numpy(X).float()
outputs = torch.from_numpy(RE).float()

dataset = TensorDataset(inputs, outputs)

train_set, val_set = torch.utils.data.random_split(dataset, [9000, 3000])

batch_size_train = 9000
batch_size_test = 500
train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(val_set, batch_size=batch_size_test, shuffle=True)

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
    return loss.item()

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

model = Net(input_dim, output_dim)
print(model)


# Define Optimizer and Loss Function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

iter = 500
train_loss = []
test_loss = []
epochs =[]


for t in range(iter):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss.append(train_loop(train_loader, model, loss_fn, optimizer))
    test_loss.append(test_loop(test_loader, model, loss_fn))
    epochs.append(t+1)
print("Done!")

#epochs = np.linspace(1,100,len(train_loss)).reshape(100,1)

plt.plot(epochs,train_loss)
plt.plot(epochs,test_loss)
plt.xlabel("# of epochs")
plt.ylabel("Training and Testing Loss")
plt.title("Training and testing Loss for different number of epochs")


##### test points for which RE is negative

def data_sets(num_test,num_dim_x,num_dim_control):
    np.random.seed(1024)
    v_l = 1.
    v_h = 2.
    np.random.seed(1)

    X_MIN = np.array([-5., -5., -np.pi, v_l]).reshape(-1, 1)
    X_MAX = np.array([5., 5., np.pi, v_h]).reshape(-1, 1)

    lim = 1.
    XE_MIN = np.array([-lim, -lim, -lim, -lim]).reshape(-1, 1)
    XE_MAX = np.array([lim, lim, lim, lim]).reshape(-1, 1)

    U_MIN = np.array([-3., -3.]).reshape(-1, 1)
    U_MAX = np.array([3., 3.]).reshape(-1, 1)

    def sample_xef():
        return (X_MAX - X_MIN) * np.random.rand(num_dim_x, 1) + X_MIN

    def sample_x(xref):
        xe = (XE_MAX - XE_MIN) * np.random.rand(num_dim_x, 1) + XE_MIN
        x = xref + xe
        x[x > X_MAX] = X_MAX[x > X_MAX]
        x[x < X_MIN] = X_MIN[x < X_MIN]
        return x

    def sample_uref():
        return (U_MAX - U_MIN) * np.random.rand(num_dim_control, 1) + U_MIN

    def sample_full():
        xref = sample_xef()
        uref = sample_uref()
        x = sample_x(xref)
        return (x, xref, uref)

    #X_tr = [sample_full() for _ in range(num_train)]
    X_te = [sample_full() for _ in range(num_test)]

    x = []
    xref = []

    for id in range(len(X_te)):
        x.append(torch.from_numpy(X_te[id][0]).float())
        xref.append(torch.from_numpy(X_te[id][1]).float())

    x, xref = (torch.stack(d).detach() for d in (x, xref))

    return x, xref


num_test = 1000
num_dim_x = 4
num_dim_control = 2

x_test,xref_test = data_sets(num_test,num_dim_x, num_dim_control)

X_test = torch.concat((x_test.reshape(num_test,num_dim_x),xref_test.reshape(num_test,num_dim_x)), dim = 1)

with torch.no_grad():
        pred = model(X_test)
