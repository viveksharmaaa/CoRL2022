import pickle
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt

torch.manual_seed(0)  #for reproducibility

num_dim_x = 4
input_dim = 2 * num_dim_x
output_dim = 1

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

def Data(length):
    with open('25k_samples.pkl', 'rb') as f:  #25k_samples data_closed_x0_X
        full_data = pickle.load(f)

    Xstar = []
    Xcurr = []
    RE = []

    data = full_data[0:length]

    for j in range(int(len(data))):
        Xstar.append(data[j]["xstar"])
        Xcurr.append(data[j]["x"])
        RE.append(data[j]["RE"])

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

def _RE_negative_points():
    with open('25k_samples.pkl', 'rb') as f:  #25k_samples data_closed_x0_X
        full_data = pickle.load(f)

    Xstar = []
    Xcurr = []
    RE = []

    data = full_data[len(full_data)-1000:len(full_data)]

    for j in range(int(len(data))):
        Xstar.append(data[j]["xstar"])
        Xcurr.append(data[j]["x"])
        RE.append(data[j]["RE"])

    Xstar = np.asarray(Xstar)
    Xcurr = np.asarray(Xcurr)
    RE = np.asarray(RE)


    Xstar = Xstar.reshape(len(Xstar), num_dim_x)
    Xcurr = Xcurr.reshape(len(Xcurr), num_dim_x)
    RE = RE.reshape(len(RE), 1)

    X = np.concatenate((Xstar, Xcurr), axis=1)
    inp = torch.from_numpy(X).float()
    re = torch.from_numpy(RE).float()

    with torch.no_grad():
        pred = model(inp)

    X_Pts = inp[torch.where(pred < 0)[0], :]
    RE_true = re[torch.where(pred < 0)[0], :]

    neg_points_no = len(X_Pts)
    return neg_points_no, loss_fn(pred[torch.where(pred < 0)[0], :], RE_true).item()


model = Net(input_dim, output_dim)
print(model)


# Define Optimizer and Loss Function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

iter = 100
epochs =[]
tr_loss = []
val_loss = []

length = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000]


neg_pts = []
mse_neg_pts = []

for i in range(len(length)):
    train_loader, test_loader = Data(length[i])
    train_loss = []
    test_loss = []
    for t in range(iter):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss.append(train_loop(train_loader, model, loss_fn, optimizer))
        test_loss.append(test_loop(test_loader, model, loss_fn))
    epochs.append(t+1)
    tr_loss.append(train_loss[-1])
    val_loss.append(test_loss[-1])
    npts, l_mse = _RE_negative_points()
    neg_pts.append(npts)
    mse_neg_pts.append(l_mse)
    print("Done!")

#epochs = np.linspace(1,100,len(train_loss)).reshape(100,1)


#of violations

plt.bar(['2000', '4000', '6000', '8000', '10000', '12000','14000', '16000', '18000', '20000', '22000', '24000'],neg_pts, color='royalblue', alpha=0.7)
plt.xlabel("# of data points")
plt.ylabel("# of negative violations")
plt.title("# of negative RE violations with # of data points")


#MSE of negative violations

plt.bar(['2000', '4000', '6000', '8000', '10000', '12000','14000', '16000', '18000', '20000', '22000', '24000'],mse_neg_pts, color='black', alpha=0.7)
plt.xlabel("# of data points")
plt.ylabel("MSE of points negative violations")
plt.title("MSE of negative with # of data points")

plt.plot(length,tr_loss)
plt.xlabel("# of data points")
plt.ylabel("Training Loss")
plt.title("Training loss variation with # of data points")
#plt.savefig('train_loss_vs_data_closed.png')


plt.plot(length,val_loss)
plt.xlabel("# of data points")
plt.ylabel("Validation Loss")
plt.title("Validation loss variation with # of data points")
#plt.savefig('test_loss_vs_data_closed.png')



##### test points for which RE is negative

num_test = 12000
num_dim_x = 4
num_dim_control = 2

x_test,xref_test = data_sets(num_test,num_dim_x, num_dim_control)

X_test = torch.concat((x_test.reshape(num_test,num_dim_x),xref_test.reshape(num_test,num_dim_x)), dim = 1)

#X_test = np.concatenate((Xclosed,Xstar),axis=1).reshape(Xstar.shape[0],2* Xstar.shape[1])
#X_test = torch.tensor(X_test)

with torch.no_grad():
        pred = model(X_test)

RE_n_pts = X_test[torch.where(pred < 0)[0],:]

plt.scatter(X_test[:,0].numpy(),X_test[:,4].numpy(),marker='^',c="k")
plt.scatter(RE_n_pts[:,0].numpy(),RE_n_pts[:,4].numpy(),marker='o',c="g")
plt.xlabel("x")
plt.ylabel("xstar")
plt.title("Landscape of negative Riemmanian Energy")
plt.legend(["All data points","Data points of negative RE"])

plt.savefig('input_land_closed.png')





