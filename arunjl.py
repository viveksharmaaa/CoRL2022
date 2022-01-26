import torch
from torch.autograd import grad
import torch.nn.functional as F

import importlib
import numpy as np
import time
from tqdm import tqdm

from torch import nn
effective_dim_start = 2
effective_dim_end = 4
num_dim_x = 4
num_dim_control = 2

num_train =1
num_test = 5

def data_sets(num_train,num_test):
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

    X_tr = [sample_full() for _ in range(num_train)]
    X_te = [sample_full() for _ in range(num_test)]

    return X_tr, X_te


# CAR



x= []
xref = []
uref = []

for id in range(len(X_tr)) :
    x.append(torch.from_numpy(X_tr[id][0]).float())
    xref.append(torch.from_numpy(X_tr[id][1]).float())
    uref.append(torch.from_numpy(X_tr[id][2]).float())

x, xref, uref = (torch.stack(d).detach() for d in (x,xref,uref))
x = x.requires_grad_()

bs = x.shape[0]
x = x.squeeze(-1)
w_lb = 0.1


def W_func(x,w_lb, bs = 1, effective_dim_start = 2, effective_dim_end = 4, num_dim_x = 4, num_dim_control = 2):

    model_Wbot = torch.nn.Sequential(
        torch.nn.Linear(1, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, (num_dim_x - num_dim_control) ** 2, bias=False))

    dim = effective_dim_end - effective_dim_start
    model_W = torch.nn.Sequential(
        torch.nn.Linear(dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, num_dim_x * num_dim_x, bias=False))

    CCM = torch.load('log_CAR/model_best.pth.tar', map_location=torch.device('cpu'))

    with torch.no_grad():
        model_W[0].weight.copy_(CCM['model_W']['0.weight'])
        model_W[0].bias.copy_(CCM['model_W']['0.bias'])
        model_W[2].weight.copy_(CCM['model_W']['2.weight'])
        model_Wbot[0].weight.copy_(CCM['model_Wbot']['0.weight'])
        model_Wbot[0].bias.copy_(CCM['model_Wbot']['0.bias'])
        model_Wbot[2].weight.copy_(CCM['model_Wbot']['2.weight'])

    W = model_W(x[:, effective_dim_start:effective_dim_end]).view(bs, num_dim_x, num_dim_x)
    Wbot = model_Wbot(torch.ones(bs, 1).type(x.type())).view(bs, num_dim_x - num_dim_control,
                                                             num_dim_x - num_dim_control)
    W[:, 0:num_dim_x - num_dim_control, 0:num_dim_x - num_dim_control] = Wbot
    W[:, num_dim_x - num_dim_control::, 0:num_dim_x - num_dim_control] = 0
    W = W.transpose(1, 2).matmul(W)
    W = W + w_lb * torch.eye(num_dim_x).view(1, num_dim_x, num_dim_x).type(x.type())

    return W


def Jacobian_Matrix(M, x):
    # NOTE that this function assume that data are independent of each other
    # along the batch dimension.
    # M: B x m x m
    # x: B x n x 1
    # ret: B x m x m x n
    bs = x.shape[0]
    m = M.size(-1)
    n = x.size(1)
    J = torch.zeros(bs, m, m, n).type(x.type())
    for i in range(m):
        for j in range(m):
            J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True)[0].squeeze(-1)
    return J


  #Batch size x n x n x n

matrix = W.detach().numpy().reshape(W.shape[1],W.shape[2])

def ps_params(config):
    #nodes = CGLnodes(config)
    #weights = ClenshawCurtisWeight(config)
    nodes, weights = _chebpts(config)
    return {"nodes" : nodes,"weights" : weights}







