import numpy as np
import time
import torch
from torch.autograd import grad

config = {
  "N": 7,
  "deg": 4,
  "alpha0" : 1,  # initial step size
    "c": 0.1,  # termination condition for backtracking line search
"tau" : 0.1, # rescale factor for backtracking line search
"sparse_eps" : 1E-20,
"rel_tol" : 1E-7,
"repetition" : 1,
  "dim": 4,
    "type":"chebyshev",
    "w_lb" : 0.1,
    "num_dim_control" : 2,
    "effective_dim_start" : 2,
    "effective_dim_end" : 4
}

def data_sets(num_train,num_test,num_dim_x,num_dim_control):
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

    x = []
    xref = []
    uref = []

    for id in range(len(X_tr)):
        x.append(torch.from_numpy(X_tr[id][0]).float())
        xref.append(torch.from_numpy(X_tr[id][1]).float())
        uref.append(torch.from_numpy(X_tr[id][2]).float())

    x, xref, uref = (torch.stack(d).detach() for d in (x, xref, uref))

    return x, xref, uref

def ClenshawCurtisWeight(config):
    N = config["N"]
    w = np.zeros(N+1).reshape(N+1,1)
    if N%2 == 0:
        w[0] = 1/(N**2 - 1)
        w[-1] = w[0]
        for s in range(np.floor(N/2).astype(int)):
            wi = 0
            for j in range(np.floor(N/2).astype(int)+1):
                if j == 0:
                    wi = wi + 0.5/(1-4*(j**2)) * np.cos(2*np.pi*j*(s+1)/N)
                elif j == N/2:
                    wi = wi + 0.5/(1-4*(j**2)) * np.cos(2*np.pi*j*(s+1)/N)
                else:
                    wi = wi + 1/(1-4*(j**2)) * np.cos(2*np.pi*j*(s+1)/N)
            w[s+1] = 4/N*wi
            w[N-s-1] = w[s+1]
    return w/2

def _chebpts(config):
    K = config["N"]
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

    return n.reshape(n.shape[0],1), w.reshape(w.shape[0],1)

def CGLnodes(config): #N is number of nodes
    return np.array([0.5 - 0.5*np.cos(k*np.pi/config["N"]) for k in range(config["N"]+1)]).reshape(config["N"]+1,1)   #columnvector

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

def _chebpolyder(config, s, ps, T):
    # D : Max degree

    N = np.size(ps["nodes"])
    dT = np.zeros((config["deg"]+1,config["N"]+1))

    for i in range(config["deg"]+1):
        for j in range(N):
            if i == 0:
                dT[i, j] = 0
            elif i == 1:
                dT[i, j] = 1
            else:
                dT[i, j] = 2*T[i-1, j] + 2*ps["nodes"][j]*dT[i-1, j] - dT[i-2, j]
    return dT[1:,np.where(ps["nodes"].squeeze(-1) == s)[0][0]].reshape(dT.shape[0]-1,1)

def ps_params(config):
    #nodes = CGLnodes(config)
    #weights = ClenshawCurtisWeight(config)
    nodes, weights = _chebpts(config)
    return {"nodes" : nodes,"weights" : weights}


def ChebyshevPolynomial(config, s):  #https://people.sc.fsu.edu/~jburkardt/py_src/chebyshev_polynomial/chebyshev_polynomial.html
    s = 2*s -1
    T0 = np.ones(np.size(s))
    T1 = s
    T= np.hstack((T0,T1))
    if config["deg"] >=2:
        for i in range(1,config["deg"]):
            T2 = 2*s*T1 - T0
            T = np.hstack((T, T2))
            T0 = T1
            T1 = T2
    return np.transpose(T.reshape(1,config["deg"]+1))   #nodes go across, basis goes down


# def evalMetric (config, x):
#     return config["W"][0] + config["W"][1]*x + config["W"][2]*x*x

def W_func(x,w_lb, bs, effective_dim_start, effective_dim_end, num_dim_x, num_dim_control):

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


def W_JW(config, ps, X):
    W, DWdx = eval_metric_jacobian(config, ps, X)
    return {"W": W, "DWdx": DWdx}


def eval_metric_jacobian(config, ps, X):
    xp = []
    for i in range(ps["nodes"].shape[0]):
        xp.append(torch.from_numpy(X[:, i]).float())
    x = torch.stack(xp).reshape(config["N"] + 1, config["dim"], 1)
    x = x.requires_grad_()
    bs = x.shape[0]
    x = x.squeeze(-1)

    w_lb = config["w_lb"]
    num_dim_x = config["dim"]
    num_dim_control = config["num_dim_control"]
    effective_dim_start = config["effective_dim_start"]
    effective_dim_end = config["effective_dim_end"]

    W = W_func(x, w_lb, bs, effective_dim_start, effective_dim_end, num_dim_x,num_dim_control)
    DWDx = Jacobian_Matrix(W, x).detach().numpy().reshape(config["N"] + 1, num_dim_x, num_dim_x, num_dim_x)
    W = W.detach().numpy().reshape(config["N"] + 1, num_dim_x, num_dim_x)
    return W, DWDx


def sparsify(x,eps):
    return np.zeros(np.shape(x)) if (abs(x)< eps).all() else x

def ChebyshevSecondKind(config, s, k = 0):
    deg = config["deg"] - k
    s = 2*s -1
    U0 = np.ones(np.size(s))
    U1 = 2*s
    if deg == 0:
        U = U0
    elif deg == 1:
        U = np.hstack((U0, U1))
    else:
        U = np.hstack((U0, U1))
    if deg >=2:
        for i in range(1,deg):
            U2 = 2*s*U1 - U0
            U = np.hstack((U,U2))
            U0 = U1
            U1 = U2
    return np.transpose(U.reshape(1,deg+1))

def DiffChebyshevPolynomial(config, s, ps, L):
        bot = np.vstack((0,_chebpolyder(config, s, ps,  L)))
        return sparsify(2*np.linspace(0,config["deg"],config["deg"]+1).reshape(config["deg"]+1,1) * bot,config["sparse_eps"])

def Constraints(config,ps,xstar, xcurr):
    #Z = np.zeros(config["deg"]+1).reshape(1,config["deg"]+1)
    start = 0
    finish = 0
    start = np.transpose(_chebpoly(config, 0, ps))
    finish = np.transpose(_chebpoly(config, 1, ps))

    Am = np.zeros((2 * config["dim"], config["dim"] * (config["deg"] + 1)))
    Am[0, 0:config["deg"]+1] = start
    Am[config["dim"],0:config["deg"]+1] = finish

    for i in range(1,config["dim"]): #modified
        Am[i,i*(config["deg"]+1) : (i+1)*(config["deg"]+1)] = start
        Am[config["dim"]+i,i*(config["deg"]+1) : (i+1)*(config["deg"]+1)] = finish

    b= np.concatenate((xstar,xcurr),axis=0)
    return Am, b

def computeEnergy(config,ps,C, L, dL):
    C_shape = np.reshape(C, (config["dim"], config["deg"] + 1))
    X = np.matmul(C_shape, L)
    dX = np.matmul(C_shape, dL)
    W_Jw = W_JW (config, ps, X)
    E = 0.0
    for i in range(config["N"]+1):
        E = E + np.matmul(dX[:,i].reshape(1,dX.shape[0]),np.matmul(np.linalg.inv(W_Jw["W"][i]),dX[:,i].reshape(dX.shape[0],1)) * ps["weights"][i]) #INVERSE
    return E


def computeJacobian(config,ps,C,L,dL):
    N = config["N"]
    deg = config["deg"]
    C_reshape = np.reshape(C, (config["dim"],config["deg"] + 1))
    X = np.matmul(C_reshape,L)
    dX = np.matmul(C_reshape,dL)
    W_Jw = W_JW(config, ps, X)
    # gx = np.zeros((deg + 1, N + 1)).reshape(deg+1, N+1)
    # gy = np.zeros((deg + 1, N + 1)).reshape(deg+1, N+1)
    # gz = np.zeros((deg + 1, N + 1)).reshape(deg+1, N+1)
    # g4 = np.zeros((deg + 1, N + 1)).reshape(deg+1, N+1) #added for CAR
    gradient = np.zeros((config["dim"],deg + 1, N + 1)).reshape(config["dim"],deg+1, N+1)
    for j in range(N+1):
        Wx = W_Jw["W"][j]  # change the function size : (dim,dim)
        M = np.linalg.inv(Wx) #INVERSE
        d = dX[:,j].reshape(dX.shape[0],1)
        dsj = ps["weights"][j]
        dLds = dL[:,j]*dsj
        Lds = L[:,j]* dsj
        Md = 2 * np.matmul(M,d)
        # gx[:,j] = Md[0]*dLds + np.matmul(np.matmul(np.transpose(d), -1 * np.matmul(np.matmul(M, W_Jw["DWdx"][j][:][:][0] ),M)),d)[0] * Lds #stacked in 'i' column of gx
        # gy[:,j] = Md[1]*dLds + np.matmul(np.matmul(np.transpose(d), -1 * np.matmul(np.matmul(M, W_Jw["DWdx"][j][:][:][1] ),M)),d)[0] * Lds
        # gz[:,j] = Md[2]*dLds + np.matmul(np.matmul(np.transpose(d), -1 * np.matmul(np.matmul(M, W_Jw["DWdx"][j][:][:][2] ),M)),d)[0] * Lds
        # g4[:,j] = Md[3]*dLds + np.matmul(np.matmul(np.transpose(d), -1 * np.matmul(np.matmul(M, W_Jw["DWdx"][j][:][:][3] ),M)),d)[0] * Lds
        for i in range(config["dim"]):
            gradient[i, :, j] = Md[i] * dLds + np.matmul(np.matmul(np.transpose(d), -1 * np.matmul(np.matmul(M, W_Jw["DWdx"][j][:][:][i]), M)), d)[0] * Lds
    #g =np.concatenate((gx,gy,gz,g4),axis=0)
    g = np.concatenate(gradient,axis = 0)
    return sparsify(g.sum(axis=1).reshape((g.shape[0],1)),config["sparse_eps"])

def pseudospectral_geodesic(xstar,xcurr): #had config before
    ps = ps_params(config)
    C = np.zeros(config["dim"] * (config["deg"] + 1)).reshape(config["dim"] * (config["deg"] + 1),1)
    E0 = 0
    t = time.time()
    L = 0
    dL = 0
    for rep in range(config["repetition"]):
        C = np.zeros(config["dim"] * (config["deg"] + 1)).reshape((config["dim"] * (config["deg"] + 1)),1)
        L = np.hstack([_chebpoly(config, ni, ps) for ni in ps["nodes"]])
        dL = np.hstack([DiffChebyshevPolynomial(config, ni, ps, L) for ni in ps["nodes"]])
        for q in range(config["dim"]):
            C[(config["deg"] + 1) * (q - 1) + 1 - 1] = 0.5 * (xstar[q] + xcurr[q])
            C[(config["deg"] + 1) * (q - 1) + 2 - 1] = 0.5 * (xcurr[q] - xstar[q])
        A, b = Constraints(config, ps, xstar, xcurr)
        E0 = computeEnergy(config, ps, C, L, dL)
        H = np.eye((config["dim"] * (config["deg"] + 1)))
        H_default = H
        g = computeJacobian(config, ps, C, L, dL)  #shape (dim *(deg+1),1)
        step_dir = 1
        alpha = 1
        while np.linalg.norm(step_dir * alpha) > config["rel_tol"]:
            if np.linalg.det(H) < 0:
                H = H_default
            KKT_mat = np.vstack((np.hstack((H,np.transpose(A))),np.hstack((A,np.zeros((2*config["dim"],2*config["dim"]))))))
            KKT_sol = np.matmul(np.linalg.inv(KKT_mat),np.vstack((g,np.matmul(A,C)-b))) #sth weird
            step_dir = -1 * KKT_sol[0:(config["dim"] *(config["deg"]+1))]
            m = np.matmul(np.transpose(g),step_dir)
            t = - config["c"] * m
            alpha = config["alpha0"]
            EE = 0
            while True:
                EE = computeEnergy(config, ps, C+alpha*step_dir, L, dL)
                #print(EE)
                if E0 - EE >= (alpha*t)[0]:
                    break
                alpha = config["tau"] * alpha
            E0 = EE
            s = alpha * step_dir  # column
            C = C + s
            g0 = g  # column
            g = computeJacobian(config, ps, C, L, dL)
            gamma = (g - g0)  # column
            H = H - (np.matmul(np.matmul(H,np.matmul(s,np.transpose(s))),H)/(np.matmul(np.transpose(s),np.matmul(H,s)))) + (np.multiply(gamma,np.transpose(gamma)) / np.matmul(np.transpose(gamma),s))
    comp_time = (time.time() - t) / config["repetition"]
    #to = {"E0" : E0,"ps" : ps, "comp_time" : comp_time, "C": C}
    return {"E0" : E0,"ps" : ps, "comp_time" : comp_time, "C": C}


#
# #Run the following lines
# num_train = 12 * 1000
# num_test = 1
# x,xref,uref = data_sets(num_train,num_test,config["dim"], config["num_dim_control"])
#
# xstar = xref.numpy().reshape(num_train,config["dim"],1)
# xcurr = x.numpy().reshape(num_train,config["dim"],1)
#
#
# myList = []
#
# start_time = time.time()
# for i in range(num_train):
#     ps_result = pseudospectral_geodesic(config, xstar[i], xcurr[i])
#     print(ps_result["E0"])
#     myList.append({"xstar": xstar[i], "x": xcurr[i], "RE": ps_result["E0"]})
# print("--- %s seconds ---" % (time.time() - start_time))
# print("Done")
#
#
# #Loading data from saved file pickle.pkl
import pickle
with open('12k_closed.pkl', 'wb') as f:
    pickle.dump(myList, f)

# with open('1k.pkl', 'rb') as f:
#     data = pickle.load(f)




