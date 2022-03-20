import importlib
import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np

func = importlib.import_module('train_test_RE')

ndx = 4
input_dim = 2 * ndx
output_dim = 1

#Instantiate the model
model = func.Net(input_dim, output_dim)
print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        # pred = torch.sigmoid(torch.square((torch.norm(X[:, 0:ndx] - X[:, ndx:], dim=1)).reshape(X.shape[0],1)))*(model(X) * model(X)) + \
        #        0.1 * torch.square(torch.norm(X[:, 0:ndx] - X[:, ndx:], dim=1)).reshape(X.shape[0],1)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # pred = torch.sigmoid(torch.square((torch.norm(X[:, 0:ndx] - X[:, ndx:], dim=1)).reshape(X.shape[0],1)))*(model(X) * model(X)) + \
            #    0.1 * torch.square(torch.norm(X[:, 0:ndx] - X[:, ndx:], dim=1)).reshape(X.shape[0],1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

def _RE_negative_points(evalfile):
    with open(evalfile,'rb') as f:  #'data_CAR_closed_x0_from_X'
        full_data = pickle.load(f)

    Xstar = []
    Xcurr = []
    RE = []

    data = full_data[len(full_data)-15000:len(full_data)]

    for j in range(int(len(data))):
        Xstar.append(data[j]["xstar"])
        Xcurr.append(data[j]["x"])
        RE.append(data[j]["RE"])

    Xstar = np.asarray(Xstar)
    Xcurr = np.asarray(Xcurr)
    RE = np.asarray(RE)


    Xstar = Xstar.reshape(len(Xstar), ndx)
    Xcurr = Xcurr.reshape(len(Xcurr), ndx)
    RE = RE.reshape(len(RE), 1)

    X = np.concatenate((Xstar, Xcurr), axis=1)
    inp = torch.from_numpy(X).float()
    re = torch.from_numpy(RE).float()

    with torch.no_grad():
        pred = model(inp)
    #     pred =  torch.sigmoid(torch.square(torch.norm(inp[:, 0:ndx] - inp[:, ndx:], dim=1)).reshape(X.shape[0], 1)) * (model(inp) * model(inp)) + \
    # 0.1 * torch.square(torch.norm(inp[:, 0:ndx] - inp[:, ndx:], dim=1)).reshape(X.shape[0], 1)


    X_Pts = inp[torch.where(pred < 0)[0], :]
    RE_true = re[torch.where(pred < 0)[0], :]

    neg_points_no = len(X_Pts)
    return neg_points_no, loss_fn(pred,re).item()


iter = 100
epochs =[]
tr_loss = [[],[],[],[]]
val_loss = [[],[],[],[]]

length = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000]

file1 = "data_CAR_closed_x0_from_X_40k_pts.pkl"
file2 = "40k_samples.pkl"
neg_pts = [[],[],[],[]]  # first two is evaluation at traj points for NN trained on traj points and samples respectively
mse_neg_pts = [[],[],[],[]]  # first two is evaluation at traj points for NN trained on traj points and samples respectively

data_f = [file1, file2]
train_l= [[],[],[],[]]
test_l = [[],[],[],[]]


# for l in range(10):
#     for k in range(2):
#         for j in range(2):
#             for i in range(len(length)):
#                 train_loader, test_loader = func.Data(length[i], ndx, data_f[j])
#                 train_loss = train_l[j + 2 * k]
#                 test_loss = test_l[j + 2 * k]
#                 for t in range(iter):
#                     print(f"Epoch {t + 1}\n-------------------------------")
#                     train_loss.append(train_loop(train_loader, model, loss_fn, optimizer))
#                     test_loss.append(test_loop(test_loader, model, loss_fn))
#                 tr_loss[j + 2 * k].append(train_loss[-1])
#                 val_loss[j + 2 * k].append(test_loss[-1])
#                 npts, l_mse = _RE_negative_points(data_f[k])
#                 neg_pts[j + 2 * k].append(npts)
#                 mse_neg_pts[j + 2 * k].append(l_mse)
# print("Done!")

for j in range(2):
    for i in range(len(length)):
        train_loader, test_loader = func.Data(length[i], ndx, data_f[j])
        train_loss = train_l[j]
        test_loss = test_l[j]
        for t in range(iter):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loss.append(train_loop(train_loader, model, loss_fn, optimizer))
            test_loss.append(test_loop(test_loader, model, loss_fn))
        tr_loss[j].append(train_loss[-1])
        val_loss[j].append(test_loss[-1])
print("Done!")


#Training  and testing Loss

phi1_mse_train=[]
phi1_mse_val=[]
phi1_train = np.asarray(tr_loss[0])
phi1_test = np.asarray(val_loss[0])

for i in range(12):
    phi1_mse_train.append(phi1_train[9*(i+1)])
    phi1_mse_val.append(phi1_test[9*(i+1)])


#for Phi1
plt.plot(['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k'],tr_loss[0]) #phi1_mse_train
plt.plot(['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k'],val_loss[0])
plt.xlabel("# of data points")
plt.ylabel("MSE Loss")
plt.title("Loss variation for $\phi_2^{*}$ with # of data points")
plt.legend(["Train Loss", "Test Loss"])
plt.savefig('Phi2_train_test__Parametric_orig.png')

#All 3 ReLU is the best
#
# plt.plot(['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k'],val_loss[1]) #phi1_mse_val
# plt.xlabel("# of data points")
# plt.ylabel("MSE Loss")
# plt.title("Validation loss variation with # of data points")
# #plt.savefig('Phi1_test_Parametric.png')

#for Phi2

# phi2_mse_train=[]
# phi2_mse_val=[]
# phi2_train = np.asarray(tr_loss[1])
# phi2_test = np.asarray(val_loss[1])
#
# for i in range(12):
#     phi2_mse_train.append(phi2_train[9*(i+1)])
#     phi2_mse_val.append(phi2_test[9*(i+1)])
#
# plt.plot(['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k'],phi2_mse_train)
# plt.xlabel("# of data points")
# plt.ylabel("MSE Loss")
# plt.title("Training loss variation with # of data points")
# plt.savefig('Phi2_train_Parametric.png')
#
# plt.plot(['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k'],phi2_mse_val)
# plt.xlabel("# of data points")
# plt.ylabel("MSE Loss")
# plt.title("Validation loss variation with # of data points")
# plt.savefig('Phi2_test_Parametric.png')

# Performance Comparison
X = ['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k']
X_axis = np.arange(len(X))

phi1_traj_mse=[]
phi2_traj_mse=[]
traj1 = np.asarray(mse_neg_pts[0])
traj2 = np.asarray(mse_neg_pts[1])

for i in range(12):
    phi1_traj_mse.append(np.sum(traj1[10*i:(i+1)*10])/10)
    phi2_traj_mse.append(np.sum(traj2[10*i:(i+1)*10])/10)


plt.bar(X_axis - 0.2, phi1_traj_mse, 0.4, label=r'$\Phi_1^{*}$') #phi1 is traj trained
plt.bar(X_axis + 0.2, phi2_traj_mse, 0.4, label=r'$\Phi_2^{*}$') #phi2 is non-traj trained

plt.xticks(X_axis, X)
plt.xlabel("# of data points")
plt.ylabel("Mean Square Error")
plt.title("MSE for testing with trajectory sampled data-points")
plt.legend()
plt.savefig('Phis_traj_mse_Parametric.png')

# #sampled points evaluations
#
# phi1_samp_pts = neg_pts[2]
# phi2_samp_pts = neg_pts[3]
#
#
# plt.bar(X_axis - 0.2, phi1_samp_pts, 0.4, label=r'$\phi_1$') #phi1 is traj trained
# plt.bar(X_axis + 0.2, phi2_samp_pts, 0.4, label=r'$\phi_2$') #phi2 is non-traj trained
#
# plt.xticks(X_axis, X)
# plt.xlabel("# of data points")
# plt.ylabel("# of Negative Violations")
# plt.title("# of negative violations for uniformly sampled data-points")
# plt.legend()
#
#
# #mse

phi1_samp_mse=[]
phi2_samp_mse=[]
samp1 = np.asarray(mse_neg_pts[2])
samp2 = np.asarray(mse_neg_pts[3])

for i in range(12):
    phi1_samp_mse.append(np.sum(samp1[10*i:(i+1)*10])/10)
    phi2_samp_mse.append(np.sum(samp2[10*i:(i+1)*10])/10)


plt.bar(X_axis - 0.2, phi1_samp_mse, 0.4, label=r'$\Phi_1^{*}$') #phi1 is traj trained
plt.bar(X_axis + 0.2, phi2_samp_mse, 0.4, label=r'$\Phi_2^{*}$') #phi2 is non-traj trained

plt.xticks(X_axis, X)
plt.xlabel("# of data points")
plt.ylabel("Mean Square Error")
plt.title("MSE for testing with uniformly sampled data-points")
plt.legend()
plt.savefig('Phis_samples_mse_Parametric.png')

##############

with open('nonparam.pickle', 'rb') as f:  # data_closed_x0_X 25k_samples '25k_samples.pkl'
    non_param = pickle.load(f)

with open('NN_nonparam.pkl', 'wb') as f:
    pickle.dump([tr_loss, val_loss], f)

#[tr_loss, val_loss, mse_neg_pts, neg_pts]

phi1_traj_mse_param=[]
phi1_traj_mse_nonparam=[]

traj1 = np.asarray(mse_neg_pts[3])
traj2 = np.asarray(non_param[2][3])

for i in range(12):
    phi1_traj_mse_param.append(np.sum(traj1[10*i:(i+1)*10])/10)
    phi1_traj_mse_nonparam.append(np.sum(traj2[10*i:(i+1)*10])/10)


plt.bar(X_axis - 0.2, phi1_traj_mse_param, 0.4, label=r'$\Phi_2^{*}$') #phi1 is traj trained
plt.bar(X_axis + 0.2, phi1_traj_mse_nonparam, 0.4, label=r'$\Phi_2$') #phi2 is non-traj trained

plt.xticks(X_axis, X)
plt.xlabel("# of data points")
plt.ylabel("Mean Square Error")
plt.title("MSE for testing with uniformly sampled data-points")
plt.legend()
plt.savefig('Phi2s_samples_mse.png')

# Train and test loss

with open('NN_nonparam.pkl', 'rb') as f:  # data_closed_x0_X 25k_samples '25k_samples.pkl'
    param = pickle.load(f)

plt.plot(['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k'],tr_loss[0])
plt.plot(['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k'],val_loss[0])
plt.plot(['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k'],param[0][0],'--')
plt.plot(['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k'],param[1][0],'--')
plt.xlabel("# of data points")
plt.ylabel("MSE Loss")
plt.title("Loss variation with # of data points")
plt.legend(["$\Phi_1$ Train Loss", "$\Phi_1$ Test Loss","$\Phi_1^{*}$ Train Loss","$\Phi_1^{*}$ Test Loss"])
plt.savefig('Phi1s_comparative_mse.png')

