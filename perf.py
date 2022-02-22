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

# Define Optimizer and Loss Function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()


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

    X_Pts = inp[torch.where(pred < 0)[0], :]
    RE_true = re[torch.where(pred < 0)[0], :]

    neg_points_no = len(X_Pts)
    return neg_points_no, loss_fn(pred[torch.where(pred < 0)[0], :], RE_true).item()


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
train_l= [[],[],[],[]]  # first is uCCM and second is without uCCM
test_l = [[],[],[],[]]  # first is uCCM and second is without uCCM

for k in range(2):
    for j in range(2):
        for i in range(len(length)):
            train_loader, test_loader = func.Data(length[i], ndx, data_f[j])
            train_loss = train_l[j+2*k]
            test_loss = test_l[j+2*k]
            for t in range(iter):
                print(f"Epoch {t + 1}\n-------------------------------")
                train_loss.append(func.train_loop(train_loader, model, loss_fn, optimizer))
                test_loss.append(func.test_loop(test_loader, model, loss_fn))
            tr_loss[j+2*k].append(train_loss[-1])
            val_loss[j+2*k].append(test_loss[-1])
            npts, l_mse = _RE_negative_points(data_f[k])
            neg_pts[j+2*k].append(npts)
            mse_neg_pts[j+2*k].append(l_mse)
    print("Done!")




#Training  and testing Loss
plt.plot(['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k'],tr_loss[0])
plt.xlabel("# of data points")
plt.ylabel("MSE Loss")
plt.title("Training loss variation with # of data points")
plt.savefig('Phi1_train.png')


plt.plot(['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k'],val_loss[0])
plt.xlabel("# of data points")
plt.ylabel("MSE Loss")
plt.title("Validation loss variation with # of data points")
plt.savefig('Phi1_test.png')

#for Phi2

plt.plot(['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k'],tr_loss[1])
plt.xlabel("# of data points")
plt.ylabel("MSE Loss")
plt.title("Training loss variation with # of data points")
plt.savefig('Phi2_train.png')


plt.plot(['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k'],val_loss[1])
plt.xlabel("# of data points")
plt.ylabel("MSE Loss")
plt.title("Validation loss variation with # of data points")
plt.savefig('Phi2_test.png')

# Performance Comparison

X = ['2k', '4k', '6k', '8k', '10k', '12k','14k', '16k', '18k', '20k', '22k', '24k']
phi1_traj_pts = neg_pts[0]
phi2_traj_pts = neg_pts[1]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, phi1_traj_pts, 0.4, label=r'$\phi_1$') #phi1 is traj trained
plt.bar(X_axis + 0.2, phi2_traj_pts, 0.4, label=r'$\phi_2$') #phi2 is non-traj trained  Use label=r'$\sin (x)$'

plt.xticks(X_axis, X)
plt.xlabel("# of data points")
plt.ylabel("# of Negative Violations")
plt.title("# of negative violations for trajectory sampled data-points")
plt.legend()
plt.savefig('NN_negpts_traj.png')


#MSE comparisons

phi1_traj_mse = mse_neg_pts[0]
phi2_traj_mse = mse_neg_pts[1]

plt.bar(X_axis - 0.2, phi1_traj_mse, 0.4, label=r'$\phi_1$') #phi1 is traj trained
plt.bar(X_axis + 0.2, phi2_traj_mse, 0.4, label=r'$\phi_2$') #phi2 is non-traj trained

plt.xticks(X_axis, X)
plt.xlabel("# of data points")
plt.ylabel("Mean Square Error")
plt.title("MSE of negative violation points for trajectory sampled data-points")
plt.legend()
plt.savefig('MSE_traj.png')

#sampled points evaluations

phi1_samp_pts = neg_pts[2]
phi2_samp_pts = neg_pts[3]


plt.bar(X_axis - 0.2, phi1_samp_pts, 0.4, label=r'$\phi_1$') #phi1 is traj trained
plt.bar(X_axis + 0.2, phi2_samp_pts, 0.4, label=r'$\phi_2$') #phi2 is non-traj trained

plt.xticks(X_axis, X)
plt.xlabel("# of data points")
plt.ylabel("# of Negative Violations")
plt.title("# of negative violations for uniformly sampled data-points")
plt.legend()
plt.savefig('NN_negpts_samples.png')

#mse

phi1_samp_mse = mse_neg_pts[2]
phi2_samp_mse = mse_neg_pts[3]


plt.bar(X_axis - 0.2, phi1_samp_mse, 0.4, label=r'$\phi_1$') #phi1 is traj trained
plt.bar(X_axis + 0.2, phi2_samp_mse, 0.4, label=r'$\phi_2$') #phi2 is non-traj trained

plt.xticks(X_axis, X)
plt.xlabel("# of data points")
plt.ylabel("Mean Square Error")
plt.title("MSE of negative violation points for uniformly sampled data-points")
plt.legend()
plt.savefig('MSE_samples.png')


