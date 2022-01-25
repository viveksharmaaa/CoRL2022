import numpy as np
import scipy
import cvxpy as cp

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


def coeffs(gamma, x0, xf, A, T):
    Ti = T[:,1:-1]
    # num_dim_x = 3
    # to = np.zeros((Ti.shape[0],num_dim_x))
    # for i in range(to.shape[0]):
    #     for j in range(to.shape[1]):
    #         to[i][j] =
    z = np.matmul(A,np.concatenate((np.matmul(2*Ti,np.transpose(gamma)),np.transpose(x0),np.transpose(xf)), axis=0))
    return np.transpose(z[0:-2,:])

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

def energy(c, T, Ts, weights):
    E = 0.0
    x = np.matmul(c, T)
    gamma_s = np.matmul(c, Ts)
    for k in range(np.size(weights)):
        xv = x[:,k]
        gamma_sv = gamma_s[:,k]
        E = E + np.matmul(np.transpose(gamma_sv),np.matmul(np.linalg.inv(W(xv)), gamma_sv) * weights[k])
    return E

def _energy(gamma, x0, xf, A, T, Ts, weights):
    c = coeffs(gamma, x0, xf, A, T)
    return energy(c, T, Ts, weights)

def objective(gamma):
    x0 = np.array([[0.1],[-0.1],[0.1]])
    xf = np.array([[0],[0],[0]])
    N = 7
    D = 5
    nodes, weights = _chebpts(N)
    T = _chebpoly(nodes, D)
    Ts = _chebpolyder(T, nodes, D)
    Ti = T[:, 1:-1]
    Te = T[:, [0, N-1]]
    A = np.concatenate((np.concatenate((np.matmul(2 * Ti, np.transpose(Ti)), Te), axis=1), np.concatenate((np.transpose(Te), np.zeros((2, 2))), axis=1)), axis=0)
    return _energy(gamma, x0, xf, A, T, Ts, weights)

num_dim_x = 3
gamma0 = np.zeros((num_dim_x,5))
gamma0 = cp.Variable((num_dim_x,5))


#ret = scipy.optimize.minimize(objective, gamma0, method='BFGS')

prob = cp.Problem(cp.Minimize(objective))