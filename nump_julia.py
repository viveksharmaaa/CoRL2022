import numpy as np
import torch
import importlib

import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')

np.random.seed(0)
task = 'QUADROTOR_8D'

config = importlib.import_module('config_'+ task)
model = importlib.import_module('model_'+ task)
system = importlib.import_module('system_'+ task)

X_MIN = config.X_MIN
X_MAX = config.X_MAX

dim_control = system.num_dim_control
dim_x = system.num_dim_x
effective_dim_start = model.effective_dim_start
effective_dim_end = model.effective_dim_end

CCM = torch.load('log_' + task + '/model_best.pth.tar', map_location=torch.device('cpu'))
W1 = CCM['model_W']['0.weight'].numpy()
b1 = CCM['model_W']['0.bias'].numpy()
W2 = CCM['model_W']['2.weight'].numpy()
Wbot1 = CCM['model_Wbot']['0.weight'].numpy()
bbot1 = CCM['model_Wbot']['0.bias'].numpy()
Wbot2 = CCM['model_Wbot']['2.weight'].numpy()


model_W, model_Wbot, _, _, W_func, _ = model.get_model(dim_x, dim_control, w_lb=0.1, use_cuda=False)
with torch.no_grad():
    model_W[0].weight.copy_(CCM['model_W']['0.weight'])
    model_W[0].bias.copy_(CCM['model_W']['0.bias'])
    model_W[2].weight.copy_(CCM['model_W']['2.weight'])
    model_Wbot[0].weight.copy_(CCM['model_Wbot']['0.weight'])
    model_Wbot[0].bias.copy_(CCM['model_Wbot']['0.bias'])
    model_Wbot[2].weight.copy_(CCM['model_Wbot']['2.weight'])


# x is (1,effective_dim_end - effective_dim_start)

x = torch.tensor([[1.,2.,3.,4.,5.,6.,7.,8.,9.]])
#x1[:, effective_dim_start:effective_dim_end].detach().numpy()

## NN Model
# W = model_W(x[:, effective_dim_start:effective_dim_end]).view(dim_x, dim_x)
# Wbot = model_Wbot(x[:, effective_dim_start:effective_dim_end - dim_control]).view(dim_x - dim_control,
#                                                                                       dim_x - dim_control)
# W[0:dim_x - dim_control, 0:dim_x - dim_control] = Wbot
# W[dim_x - dim_control::, 0:dim_x - dim_control] = 0
#
# W = W.transpose(1,2).matmul(W)
# W = W + 0.1 * torch.eye(dim_x).view(dim_x, dim_x).type(x.type())
###
#numpy execution

model_W_new = np.matmul(np.tanh(np.matmul(x[:, effective_dim_start:effective_dim_end].detach().numpy(),W1.transpose()) + b1),W2.transpose())
model_W_new= model_W_new.reshape(dim_x,dim_x)
model_Wbot_new = np.matmul(np.tanh(np.matmul(x[:,0:dim_x-dim_control].detach().numpy(),Wbot1.transpose()) + bbot1),Wbot2.transpose()) #x[:, effective_dim_start:effective_dim_end-dim_control]
model_Wbot_new = model_Wbot_new.reshape(dim_x-dim_control,dim_x-dim_control)
model_W_new[0:dim_x - dim_control, 0:dim_x - dim_control] = model_Wbot_new
model_W_new[dim_x - dim_control::, 0:dim_x - dim_control] = 0
model_W_new = np.matmul(model_W_new.transpose(),model_W_new) + 0.1 * np.eye(dim_x).reshape(dim_x,dim_x)

np.savez(task+'.npz',W1,b1,W2,Wbot1,bbot1,Wbot2,dim_x,dim_control,effective_dim_start,effective_dim_end,X_MIN,X_MAX)

# npzfile = np.load('CAR_sampled.npz')
# npzfile.files
# npzfile['arr_0']

#npzfile = np.load("/home/vivek/PycharmProjects/CoRL2022/Npz_file/CAR_files/CAR_sampled.npz")