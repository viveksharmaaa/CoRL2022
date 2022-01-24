from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from np2pth import get_system_wrapper, get_controller_wrapper
import torch
import importlib
from utils import EulerIntegrate
import time

import os
import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')
import argparse

np.random.seed(0)

system = importlib.import_module('system_CAR')
f, B, _, num_dim_x, num_dim_control = get_system_wrapper(system)
controller = get_controller_wrapper('log_CAR/controller_best.pth.tar')

config = importlib.import_module('config_CAR')
t = config.t
time_bound = config.time_bound
time_step = config.time_step
XE_INIT_MIN = config.XE_INIT_MIN
XE_INIT_MAX = config.XE_INIT_MAX

x_0, xstar_0, ustar = config.system_reset(np.random.rand())
ustar = [u.reshape(-1,1) for u in ustar]
xstar_0 = xstar_0.reshape(-1,1)
xstar, _ = EulerIntegrate(None, f, B, None, ustar, xstar_0, time_bound, time_step, with_tracking=False)

x_closed = []
controls = []
errors = []
xinits = []

xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
xinit = xstar_0 + xe_0.reshape(-1,1)
xinits.append(xinit)

x, u = EulerIntegrate(controller, f, B, xstar,ustar,xinit,time_bound,time_step,with_tracking=True,sigma=0)

t = np.arange(0,time_bound,time_step)

trace = []
u = []

xcurr = xinit
trace.append(xcurr)
i = 0
xe = xcurr - xstar[i]

#test yo = np.array([1.62678826, -0.21326992,  1.69686806,  1.78666505]).reshape(-1,1)
ui = controller(xcurr, xe, ustar[i])  #xcurr,xe and ustar[i] is array of size 4,4 and 2 resp.

#xinit is x(0) aka
# [array([[ 0.29126162],
 #      [-0.05174176],
# [-0.0304127 ],
#       [ 0.52701741]])



cont= torch.load('log_CAR/controller_best.pth.tar', map_location=torch.device('cpu'))

#Contraction Metric

num_dim_x = 4
num_dim_control = 2
effective_dim_start = 2
effective_dim_end = 4
#x : bs x n x 1
x= torch.rand(1,num_dim_x,1)#.type(torch.float64)
bs = x.shape[0]
x = x.squeeze(-1)
x0= x[:,effective_dim_start:effective_dim_end]

CCM = torch.load('log_CAR/model_best.pth.tar', map_location=torch.device('cpu'))

#CLASS torch.nn.Linear(in_features, out_features, bias=True)
#Applies a linear transformation to the incoming data: y = x*W^T + b

W1 = CCM['model_W']['0.weight']#.type(torch.float64)
b1 = CCM['model_W']['0.bias']#.type(torch.float64)
W2 = CCM['model_W']['2.weight']#.type(torch.float64)
#W= W2.matmul(torch.tanh(W1.matmul(x) + b1)).view(bs,num_dim_x,num_dim_x)
W = torch.tanh(x0.matmul(torch.transpose(W1,0,1))+b1).matmul(torch.transpose(W2,0,1)).view(bs,num_dim_x,num_dim_x)
Wbot1 = CCM['model_Wbot']['0.weight']#.type(torch.float64)
bbot1 = CCM['model_Wbot']['0.bias']#.type(torch.float64)
Wbot2 = CCM['model_Wbot']['2.weight']#.type(torch.float64)


Wbot = torch.tanh(torch.ones(bs, 1).matmul(torch.transpose(Wbot1,0,1))+bbot1).matmul(torch.transpose(Wbot2,0,1)).view(bs, num_dim_x-num_dim_control, num_dim_x-num_dim_control)
W[:, 0:num_dim_x-num_dim_control, 0:num_dim_x-num_dim_control] = Wbot
W[:, num_dim_x-num_dim_control::, 0:num_dim_x-num_dim_control] = 0

W = W.transpose(1,2).matmul(W)

def is_psd(mat):
    return bool((mat.transpose(0, 1) == mat).all() and (torch.view_as_real(torch.linalg.eigvals(mat))[:,0] >= 0).all())



# Test for W to be pd
for i in range(bs):
    print(is_psd(W[i,:,:]))

M=torch.inverse(W)

#https://www.philipzucker.com/deriving-the-chebyshev-polynomials-using-sum-of-squares-optimization-with-sympy-and-cvxpy/
#https://github.com/suttond/scikit-geodesic
#https://stackoverflow.com/questions/53438453/package-to-compute-geodesic-given-a-riemannian-metric
#https://ee227c.github.io/

#chebpts



#gamma0 = np.zeros((num_dim_x,N-2))

#x0 = np.zeros((num_dim_x,1))

import numpy as np
#num_dim_x=2
def _chebpts(N):
    K = N - 1
    n = 0.5 * (np.cos(np.pi * np.arange(K, -1, -1) / K) + 1)
    w = np.zeros(np.shape(n))

    Kh = K ** 2 - 1 if K % 2 == 0 else K ** 2
    w[0] = w[-1] = 0.5 / Kh
    Kt = np.floor(K / 2).astype(int)

    for k in range(Kt):
        wk = 0.0
        for j in range(Kt+1):
            beta = 1 if (j == 0) | (j == K / 2) else 2
            wk += beta / K / (1 - 4 * j ** 2) * np.cos(2 * np.pi * j * k / K)
        w[K - k] = w[k] = wk

    return n, w


def _chebpoly(config, s, ps):
    # D : Max degree
    N = np.size(ps["nodes"])
    T = np.zeros((config["deg"]+1,config["N"]+1))
    for i in range(config["deg"]+1):
        for j in range(N):
            if i == 0:
                T[i, j] = 1
            elif i == 1:
                T[i, j] = ps["nodes"][j]
            else:
                T[i, j] = 2 * ps["nodes"][j] * T[i - 1, j] - T[i - 2, j]
    return T[:,np.where(ps["nodes"].squeeze(-1) == s)[0][0]].reshape(T.shape[0],1)


def _chebpolyder(T, nodes, D):
    # D : Max degree

    N = np.size(nodes)
    dT = np.zeros((D+1,N))

    for i in range(D+1):
        for j in range(N):
            if i == 0:
                dT[i, j] = 0
            elif i == 1:
                dT[i, j] = 1
            else:
                dT[i, j] = 2*T[i-1, j] + 2*nodes[j]*dT[i-1, j] - dT[i-2, j]
    return dT

#xf = np.ones((num_dim_x,1))
#gamma = np.ones((num_dim_x,D))
#gamma0 = np.zeros((num_dim_x,N-2))

def coeffs(gamma, x0, xf, A, T):
    Ti = T[:,1:-1]
    z = np.matmul(A,np.concatenate((np.matmul(2*Ti,np.transpose(gamma)),np.transpose(x0),np.transpose(xf)), axis=0))
    return np.transpose(z[0:-2,:])

def energy(c, T, Ts, weights, W):
    E = 0.0
    x = np.matmul(c, T)
    gamma_s = np.matmul(c, Ts)
    for k in range(np.size(weights)):
        xv = x[:,k]
        gamma_sv = gamma_s[:,k]
        E = E + np.matmul(np.transpose(gamma_sv),np.matmul(np.linalg.inv(W), gamma_sv) * weights[k])  #W has to be W(xsv)
    return E

def _energy(gamma, x0, xf, A, T, Ts, weights, W):
    c = coeffs(gamma, x0, xf, A, T)
    return energy(c, T, Ts, weights, W)

#https://stackoverflow.com/questions/31292374/how-do-i-put-2-matrix-into-scipy-optimize-minimize
#https://stackoverflow.com/questions/61646792/gekko-optimization-in-matrix-form

#Objective Gamma

from gekko import GEKKO
import numpy as np
m = GEKKO(remote=False)

num_dim_x = 2
N= 7
gamma = m.Array(m.Var,(num_dim_x,N-2))

def _RiemmanianEnergy(N=7,D=5):
    nodes, weights = _chebpts(N)
    T = _chebpoly(nodes, D)
    Ts = _chebpolyder(T, nodes, D)
    Ti = T[:,1:-1]
    Te = T[:, [0, N-1]]
    A = np.concatenate((np.concatenate((np.matmul(2 * Ti, np.transpose(Ti)), Te), axis=1),np.concatenate((np.transpose(Te), np.zeros((2, 2))), axis=1)), axis=0)
    return T, Ts, nodes, weights,A


def RE(gamma):
    T, Ts, nodes, weights,A = _RiemmanianEnergy(N=7, D=5)
    x0 = np.zeros((num_dim_x, 1))
    xf = np.ones((num_dim_x, 1))
    W = np.array([[4.25828, -0.93423], [-0.93423, 3.76692]])
    c = coeffs(gamma, x0, xf, A, T)
    E = 0.0
    x = np.matmul(c, T)
    #expr = np.zeros((np.size(weights)))
    gamma_s = np.matmul(c, Ts)
    expr = ["0", "0", "0", "0", "0", "0", "0"]
    for k in range(np.size(weights)):
        xv = x[:, k]
        gamma_sv = gamma_s[:, k]
        expr[k] = np.matmul(np.transpose(gamma_sv),np.matmul(np.linalg.inv(W), gamma_sv) * weights[k])  # W has to be W(xsv)
        m.Obj(expr[k])
    return m.Obj(m.sum(expr))


    #Google APM model error: string >       15000  characters
    #https: // github.com / BYU - PRISM / GEKKO / issues / 72
    return RE

m.Minimize(RE(gamma))

m.solve()


import numpy as np
import scipy.optimize as opt
from gekko import GEKKO

p= np.array([4, 5, 6.65, 12]) #p = prices
pmx = np.triu(p - p[:, np.newaxis]) #pmx = price matrix, upper triangular
m = GEKKO(remote=False)
q = m.Array(m.Var,(4,4),lb=0,ub=10)

for i in range(4):
    for j in range(4):
        if j<=i:
            q[i,j].upper=0 # set upper bound = 0

def profit(q):
    profit = np.sum(q.flatten() * pmx.flatten())
    return profit

for i in range(4):
    m.Equation(np.sum(q[i,:])<=10)
    m.Equation(np.sum(q[:,i])<=8)

m.Maximize(profit(q))

m.solve()

print(q)

x1 = m.Var(value=20,lb=20, ub=6555)  #integer=True
x2 = m.Var(value=1,lb=1,ub=100)  #integer=True
x3 = m.sos1([30, 42, 45, 55])
x3.value = 1.0

x = [x1, x2, x3]


def fun(x):
    return 22223 + (x[0] * x[1] * x[2])**0.83

m.Obj(fun(x))

m.options.SOLVER=1
m.solve(disp=True)


import cvxpy as cp
import numpy

# Problem data.
m = 30
n = 20
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
result = prob.solve()
# The optimal value for x is stored in x.value.
print(x.value)
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
print(constraints[0].dual_value)


# Import packages.
import cvxpy as cp
import numpy as np

# Generate a random non-trivial quadratic program.
m = 15
n = 10
p = 5
np.random.seed(1)
P = np.random.randn(n, n)
P = P.T @ P
q = np.random.randn(n)
G = np.random.randn(m, n)
h = G @ np.random.randn(n)
A = np.random.randn(p, n)
b = np.random.randn(p)

# Define and solve the CVXPY problem.
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                 [G @ x <= h,
                  A @ x == b])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution corresponding to the inequality constraints is")
print(prob.constraints[0].dual_value)
