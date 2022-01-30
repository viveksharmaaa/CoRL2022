import numpy as np
import time

WC = np.array([[2.285714286, 0.1428571429, -0.4285714286],
              [0.1428571429, 1.571428571, 0.2857142857],
              [-0.4285714286, 0.2857142857, 1.142857143]])


WL = np.array([[0, -4.57142857142857, 0 ],
              [-4.57142857142857, -0.571549759244019, 0.857142857142854],
             [0, 0.857142857142854, 0]])

WQ = np.array([[0, 0, 0 ],
             [0, 9.14297833067258, 0],
             [0, 0, 0]])


# W = np.array([[4.25828, -0.93423], [-0.93423, 3.76692]]) #for 2x2 example from Arun's paper



# WC = np.array([[0.20365, 0.00015,  -0.00566],
#            [0.00015,   0.21671,  -0.00566],
#            [-0.00566,  -0.00566,   0.21607]])
# WL = np.array([[0.0,      -0.4073,   -0.00133],
#            [-0.4073,   -0.00061,   0.00998],
#            [-0.00133,   0.00998,   0.06785]])
# WQ = np.array([[0.0,      0.0,      0.00053],
#             [0.0,      0.81459,  0.00308],
#             [0.00053,  0.00308,  0.00343]])

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

def CGLnodes(config): #N is number of nodes
    return np.array([0.5 - 0.5*np.cos(k*np.pi/config["N"]) for k in range(config["N"]+1)]).reshape(config["N"]+1,1)   #columnvector

def evalMetric (config, x):
    return config["W"][0] + config["W"][1]*x + config["W"][2]*x*x

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
    Z = np.zeros(config["deg"]+1).reshape(1,config["deg"]+1)
    start = 0
    finish = 0
    start = np.transpose(_chebpoly(config, 0, ps))
    finish = np.transpose(_chebpoly(config, 1, ps))

    A= np.zeros((2*config["dim"],config["dim"]*(config["deg"]+1)))

    # A[0, :] = np.concatenate((start, Z), axis=1)
    # A[1, :] = np.concatenate((Z, start), axis=1)
    # A[2, :] = np.concatenate((finish, Z), axis=1)
    # A[3, :] = np.concatenate((Z, finish), axis=1)

    A[0,:] = np.concatenate((start,Z,Z),axis=1)
    A[1,:] = np.concatenate((Z,start,Z),axis=1)
    A[2,:] = np.concatenate((Z,Z,start),axis=1)
    A[3,:] = np.concatenate((finish,Z,Z),axis=1)
    A[4,:] = np.concatenate((Z,finish,Z),axis=1)
    A[5,:] = np.concatenate((Z,Z,finish),axis=1)

    b= np.concatenate((xstar,xcurr),axis=0)
    return A, b

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

def computeEnergy(config,ps,C,L,dL):
    C_reshape = np.reshape(C, (config["dim"],config["deg"] + 1))
    X = np.matmul(C_reshape,L)
    dX = np.matmul(C_reshape,dL)
    E = 0.0
    for i in range(config["N"]+1):
        E = E + np.matmul(dX[:,i].reshape(1,dX.shape[0]),np.matmul(np.linalg.inv(evalMetric(config, X[0,i])),dX[:,i].reshape(dX.shape[0],1)) * ps["weights"][i]) #INVERSE
    return E

def computeJacobian(config,ps,C,L,dL):
    N = config["N"]
    deg = config["deg"]
    C_reshape = np.reshape(C, (config["dim"],config["deg"] + 1))
    X = np.matmul(C_reshape,L)
    dX = np.matmul(C_reshape,dL)
    gx= np.zeros((deg+1,N+1)).reshape(deg+1,N+1)
    gy = np.zeros((deg + 1, N + 1)).reshape(deg+1,N+1)
    gz = np.zeros((deg + 1, N + 1)).reshape(deg+1,N+1)
    for j in range(N+1):
        me = X[0,j]  # [;,j] for W from CCM ) (size: a scalar)
        Wx = evalMetric (config, me)  # change the function size : (dim,dim)
        M = np.linalg.inv(Wx) #INVERSE
        dW = config["W"][1]+ 2*config["W"][2]*me  #dW is DW/dxi np.zeros((config["dim"],config["dim"])).reshape(config["dim"],config["dim"])
        Mdot = -1 * np.matmul(np.matmul(M,dW),M)
        d = dX[:,j].reshape(dX.shape[0],1)
        dsj = ps["weights"][j]
        dLds = dL[:,j]*dsj
        Lds = L[:,j]* dsj
        Md = 2 * np.matmul(M,d)
        gx[:,j] = Md[0]*dLds + np.matmul(np.matmul(np.transpose(d),Mdot),d)[0] * Lds; #stacked in 'i' column of gx
        gy[:,j] = Md[1]*dLds
        gz[:,j] = Md[2]*dLds
    g =np.concatenate((gx,gy,gz),axis=0)
    return sparsify(g.sum(axis=1).reshape((g.shape[0],1)),config["sparse_eps"])

def pseudospectral_geodesic(config,xstar,xcurr):
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
            KKT_sol = np.matmul(np.linalg.inv(KKT_mat),np.vstack((g,np.matmul(A,C)-b)))
            step_dir = -1 * KKT_sol[0:(config["dim"] *(config["deg"]+1))]
            m = np.matmul(np.transpose(g),step_dir)
            t = - config["c"] * m
            alpha = config["alpha0"]
            EE = 0
            while True:
                EE = computeEnergy(config, ps, C+alpha*step_dir, L, dL)
                print(EE)
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
    return {"E0" : E0,"ps" : ps, "comp_time" : comp_time}

config = {
  "N": 7,
  "deg": 4,
  "alpha0" : 1,
    "c": 0.1,
"tau" : 0.1,
"sparse_eps" : 1E-20,
"rel_tol" : 1E-7,
    "W": [WC,WL,WQ],
"repetition" : 100,
  "dim": 3,
    "type":"chebyshev"}

xstar = 0. * np.array([[1.],[1.],[1.]])
xcurr = 0.1 * np.array([[1.],[-1.],[1.]])

ps_result = pseudospectral_geodesic(config, xstar, xcurr)
print(ps_result["E0"])




