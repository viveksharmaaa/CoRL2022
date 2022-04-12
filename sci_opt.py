import numpy as np
from scipy.optimize import minimize
import time
import scipy
#import cvxpy as cp

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


def _chebpoly(nodes, D):
    # D : Max degree
    N = np.size(nodes)
    T = np.zeros((D,N))
    for i in range(D):
        for j in range(N):
            if i == 0:
                T[i, j] = 1
            elif i == 1:
                T[i, j] = nodes[j]
            else:
                T[i, j] = 2 * nodes[j] * T[i - 1, j] - T[i - 2, j]
    return T


def _chebpolyder(T, nodes, D):
    # D : Max degree
    N = np.size(nodes)
    dT = np.zeros((D,N))

    for i in range(D):
        for j in range(N):
            if i == 0:
                dT[i, j] = 0
            elif i == 1:
                dT[i, j] = 1
            else:
                dT[i, j] = 2*T[i-1, j] + 2*nodes[j]*dT[i-1, j] - dT[i-2, j]
    return dT


# def _coeffs(gamma, x0, xf, A, T):
#     Ti = T[:,1:-1]
#     z = np.matmul(A,np.concatenate((np.matmul(2*Ti,np.transpose(gamma)),np.transpose(x0),np.transpose(xf)), axis=0))
#     return np.transpose(z[0:-2,:])

def coeffs(gamma, x0, xf, A, T,num_dim_x,D):
    Ti = T[:,1:-1]
    yo = np.zeros((D, num_dim_x)).reshape(D, num_dim_x)
    for i in range(num_dim_x):
        for j in range(D):
            yo[j, i] = 2 * np.sum(Ti[j, :] * gamma[5*i:5*(i+1)])
    z = np.transpose(np.matmul(A[0:-2,:],np.concatenate((yo,np.transpose(x0),np.transpose(xf)))))
    return z

# #-1.07914e-13   1.07914e-13  -1.07914e-13  z in Julia
#   0.1          -0.1           0.1
#  -2.81108e-13   2.81108e-13  -2.81108e-13
#   5.63993e-14  -5.63993e-14   5.63993e-14
#  -1.92624e-14   1.92624e-14  -1.92624e-14
#  -3.53016e-15   3.53016e-15  -3.53016e-15
#  -2.38004e-15   2.38004e-15  -2.38004e-15


def W(x):
    WC = np.array([[0.20365, 0.00015, -0.00566],
                   [0.00015, 0.21671, -0.00566],
                   [-0.00566, -0.00566, 0.21607]])
    WL = np.array([[0.0, -0.4073, -0.00133],
                   [-0.4073, -0.00061, 0.00998],
                   [-0.00133, 0.00998, 0.06785]])
    WQ = np.array([[0.0, 0.0, 0.00053],
                   [0.0, 0.81459, 0.00308],
                   [0.00053, 0.00308, 0.00343]])
    return WC + WL*x[0] + WQ*x[0]*x[0]

# def energy(c, T, Ts, weights):
#     E = 0.0
#     x = np.matmul(c, T)
#     gamma_s = np.matmul(c, Ts)
#     for k in range(np.size(weights)):
#         xv = x[:,k]
#         gamma_sv = gamma_s[:,k]
#         E = E + np.matmul(np.transpose(gamma_sv),np.matmul(np.linalg.inv(W(xv)), gamma_sv) * weights[k])  #W(xv)
#     return E

def energy(c, T, Ts, weights, WC, WL, WQ):
    E = 0.0
    for k in range(np.size(weights)):
        E = E + np.matmul(np.transpose(np.matmul(c, Ts)[:, k]),
                          np.matmul(np.linalg.inv(WC + WL * np.matmul(c, T)[:, k][0] + WQ * np.matmul(c, T)[:, k][0] * np.matmul(c, T)[:, k][0]),
                                    np.matmul(c, Ts)[:, k]) * weights[k]) #WC + WL * np.matmul(c, T)[:, k][0] + WQ * np.matmul(c, T)[:, k][0] *
                                     #np.matmul(c, T)[:, k][0]

        #W(np.matmul(c, T)[:, k])


    # np.matmul(np.transpose(np.matmul(c, Ts)[:, k]),
    #           (np.linalg.inv((WC + WL * np.matmul(c, T)[:, k][0] + WQ * np.matmul(c, T)[:, k][0] *
    #                           np.matmul(c, T)[:, k][0]), np.matmul(c, Ts)[:, k]) * weights[k]))

    return E

def _energy(gamma, x0, xf, A, T, Ts, weights,num_dim_x, D, WL, WC, WQ):
    #E = 0.0
    c = coeffs(gamma, x0, xf, A, T, num_dim_x, D)
    # Ti = T[:, 1:-1]
    # yo = np.zeros((D, num_dim_x)).reshape(D, num_dim_x)
    # for i in range(num_dim_x):
    #     for j in range(D):
    #         yo[j, i] = 2 * np.sum(Ti[j, :] * gamma[5 * i:5 * (i + 1)])
    # c = np.transpose(np.matmul(A[0:-2, :], np.concatenate((yo, np.transpose(x0), np.transpose(xf)))))
    # for k in range(np.size(weights)):
    #     E = E + np.matmul(np.transpose(np.matmul(c, Ts)[:, k]),
    #                       np.matmul(np.linalg.inv(WC + WL * np.matmul(c, T)[:, k][0] + WQ * np.matmul(c, T)[:, k][0] * np.matmul(c, T)[:, k][0]),
    #                                 np.matmul(c, Ts)[:, k]) * weights[k])
    return energy(c, T, Ts, weights,WL, WC, WQ)

# def objective(gamma):
#     x0 = np.array([[0.1],[-0.1],[0.1]])
#     xf = np.array([[0],[0],[0]])
#     N = 7
#     D = 5
#     nodes, weights = _chebpts(N)
#     T = _chebpoly(nodes, D)
#     Ts = _chebpolyder(T, nodes, D)
#     Ti = T[:, 1:-1]
#     Te = T[:, [0, N-1]]
#     A = np.concatenate((np.concatenate((np.matmul(2 * Ti, np.transpose(Ti)), Te), axis=1), np.concatenate((np.transpose(Te), np.zeros((2, 2))), axis=1)), axis=0)
#     return _energy(gamma, x0, xf, A, T, Ts, weights)


#x0 = np.array([[0], [0], [0]]) #np.array([[0.1], [-0.1], [0.1]])
#xf = np.array([[0.1], [-0.1], [0.1]])
#xf = np.array([[0.1], [-0.09854243787363658], [0.09906431167166042]])

N = 7
D = 5
nodes, weights = _chebpts(N)
T = _chebpoly(nodes, D)
Ts = _chebpolyder(T, nodes, D)
Ti = T[:, 1:-1]
Te = T[:, [0, N - 1]]
A = np.linalg.inv(np.concatenate((np.concatenate((np.matmul(2 * Ti, np.transpose(Ti)), Te), axis=1),
                    np.concatenate((np.transpose(Te), np.zeros((2, 2))), axis=1)), axis=0))

WC = np.array([[0.20365, 0.00015, -0.00566],
                   [0.00015, 0.21671, -0.00566],
                   [-0.00566, -0.00566, 0.21607]])
WL = np.array([[0.0, -0.4073, -0.00133],
                   [-0.4073, -0.00061, 0.00998],
                   [-0.00133, 0.00998, 0.06785]])
WQ = np.array([[0.0, 0.0, 0.00053],
                   [0.0, 0.81459, 0.00308],
                   [0.00053, 0.00308, 0.00343]])

#from another paper

# WC = np.array([[2.285714286, 0.1428571429, -0.4285714286],
# [0.1428571429,1.571428571,0.2857142857],
# [-0.4285714286,0.2857142857,1.142857143]])
#
# # % linear
# WL = np.array([[0,-4.57142857142857,0],
#                 [-4.57142857142857, -0.571549759244019,0.857142857142854],
#                 [0, 0.857142857142854,0]])
#
# # % quadratic
# WQ = np.array([[0, 0, 0],
#                 [0,9.14297833067258,0],
#                     [0,0,0]])

#objective_function = lambda gamma,x0, xf, A, T, Ts, weights,num_dim_x, D : _energy(gamma, x0, xf, A, T, Ts, weights,num_dim_x, D)


 # x0 is x and xf is xstar

# gamma0 = np.array([[-0.05, -0.05,  0.,  0.,  0.],
#        [ 0.05,  0.05,  0.,  0.,  0.],
#        [ 0.05,  0.05,  0.,  0.,  0.]])


#https://stackoverflow.com/questions/62390625/scipy-optimize-minimize-stops-at-x0
#gamma0 = cp.Variable((num_dim_x,5))

# yo = np.zeros((D,num_dim_x))
# for i in range(num_dim_x):
#     for j in range(D):
#         yo[j,i] = 2 * np.sum(Ti[j, :] * gamma0[5 * i:5 * (i + 1)])

import pandas as pd
t = pd.read_excel('~/Desktop/ex1_jl_energy.ods', engine='odf',sheet_name="Sheet1")
x = t['x'].str.split(',',expand=True).astype(float).to_numpy()

num_dim_x = 3
out = np.zeros((x.shape[0],2)).reshape(x.shape[0],2)

def eq_constraint(gamma):
    return x[0] - x[1]

con = {'type': 'eq', 'fun': eq_constraint}

start_time = time.time()
for i in range(x.shape[0]):
    xf = np.array([[x[i][0]],[x[i][1]],[x[i][2]]])
    x0 = np.array([[0], [0], [0]])
    gamma0 = np.zeros((num_dim_x, 5))
    for o in range(len(nodes) - 2):
        gamma0[:, o] = (x0 * (1 - nodes[o + 1]) + xf * (nodes[o + 1])).reshape(num_dim_x)
    res = minimize(
        _energy,
        gamma0.flatten(),
        args=(x0, xf, A, T, Ts, weights, num_dim_x, D, WL, WC, WQ),
        method='SLSQP',
    constraints= cons) #'BFGS'
    out[i,:] = [res.fun,res.success]

print("--- %s seconds ---" % (time.time() - start_time))





#ret = scipy.optimize.minimize(objective, gamma0, method='BFGS')

#prob = cp.Problem(cp.Minimize(objective))


import numpy as np
from scipy.optimize import minimize
import matplotlib as plt

n = 10000

c = np.tile(np.random.normal(size=(n,1)),(1,2))
k = np.random.normal(size=(n,2))

def generate_moments(params,c,k):
    mu = params[0]
    sigma = params[1]
    util = mu + (sigma*c) + k
    choice = util > 0

    #a = np.zeros((2,1))
    #a[0,0] = np.mean(choice[choice[:,0]==0,1])
    #a[1,0] = np.mean(choice[choice[:,0]==1,1])
    #return a
    return (np.mean(choice[choice[:,0]==0,1]),np.mean(choice[choice[:,0]==1,1]))

#moments: p(choice|prior not-choice) = .07; p(choice|prior choice) = .35
def SMM_obj(params,W,c,k):
    sim_moments = generate_moments(params,c,k)
    diff = sim_moments - np.array([0.07,0.35])
    return np.matmul(np.matmul(diff,W),diff)

W = np.identity(2)
estimates = minimize(SMM_obj,[0,1],args=(W,c,k),options={'disp':True})

for eps in [1e-4, 1e-3, 1e-2, 1e-1]:
    print(minimize(SMM_obj,[0,1],args=(W,c,k), method="BFGS",
          options={"eps":eps, "disp":True}))


# for xstar = [0, 0 , 0]  and x = [0.1, -0.1, 0.1]

# #Iniatize gamma based on nodes 's' (leave node 0 and 1 )
# γ(s) = xs*(1-s) + x0*s
#
# gamma0 = [column 1 = γ(s), column 2 = γ(s+1) ]





