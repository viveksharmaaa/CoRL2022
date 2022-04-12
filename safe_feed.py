import numpy as np
import torch
import pickle

num_dim_x = 2
num_points = 100000
W = np.array([[4.26, -0.93],[-0.93, 3.77]])
myList = []

X_MIN = np.array([-5., -5.]).reshape(-1, 1)
X_MAX = np.array([5., 5.]).reshape(-1, 1)

def data_sets(num_train,num_dim_x):
    def sample_x():
        return (X_MAX - X_MIN) * np.random.rand(num_dim_x, 1) + X_MIN

    def sample_xref():
        return (X_MAX - X_MIN) * np.random.rand(num_dim_x, 1) + X_MIN

    def sample_full():
        x = sample_x()
        xref = sample_xref()
        return (x, xref)

    X_tr = [sample_full() for _ in range(num_train)]

    x = []
    xref = []

    for id in range(len(X_tr)):
        x.append(torch.from_numpy(X_tr[id][0]).float())
        xref.append(torch.from_numpy(X_tr[id][1]).float())

    x, xref = (torch.stack(d).detach() for d in (x, xref))

    return x, xref

x,xref= data_sets(num_points,num_dim_x)
xstar = xref.numpy().reshape(num_points,num_dim_x,1)
xcurr = x.numpy().reshape(num_points,num_dim_x,1)

for i in range(num_points):
    gamma = xcurr[i] - xstar[i]
    E0 = np.matmul(np.matmul(np.transpose(gamma),W),gamma)
    myList.append({"xstar": xstar[i], "x": xcurr[i], "RE": E0})
print("DONE")

with open('Ex2_SFMP.pkl', 'wb') as f:
     pickle.dump(myList, f)


# Training and testing

import importlib
import matplotlib.pyplot as plt

func = importlib.import_module('train_test_RE')

ndx = 2  # State Space dimension
input_dim = 2 * ndx
output_dim = 1

#Instantiate the model
model = func.Net(input_dim, output_dim)
print(model)

# Define Optimizer and Loss Function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

iter = 400
epochs =[]
tr_loss = []
val_loss = []

#length = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000]
length = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

file1 = "Ex2_SFMP.pkl"
train_l= []
test_l = []

for i in range(len(length)):
    train_loader, test_loader = func.Data(length[i], ndx, file1)
    train_loss = train_l
    test_loss = test_l
    for t in range(iter):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss.append(func.train_loop(train_loader, model, loss_fn, optimizer))
        test_loss.append(func.test_loop(test_loader, model, loss_fn))
    tr_loss.append(train_loss[-1])
    val_loss.append(test_loss[-1])

print("Done!")

#Training  and testing Loss
plt.plot(['10k', '20k', '30k', '40k', '50k', '60k','70k', '80k', '90k','100k'],tr_loss)
plt.plot(['10k', '20k', '30k', '40k', '50k', '60k','70k', '80k', '90k','100k'],val_loss)
plt.xlabel("# of data points")
plt.ylabel("MSE Loss")
plt.title("Loss variation with # of data points")
plt.legend(["Train Loss", "Test Loss"])



