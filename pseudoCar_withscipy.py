xstar = np.array([[0], [0], [0]])
xcurr = np.array([[0.1], [-0.1], [0.1]])
config = {
  "N": 7,
  "deg": 4,
  "alpha0" : 1,  # initial step size
    "c": 0.1,  # termination condition for backtracking line search
"tau" : 0.1, # rescale factor for backtracking line search
"sparse_eps" : 1E-20,
"rel_tol" : 1E-7,
"repetition" : 1,
  "dim": 3,
    "type":"chebyshev",
    "w_lb" : 0.1,
    "num_dim_control" : system.num_dim_control,
    "effective_dim_start" : model.effective_dim_start,
    "effective_dim_end" : model.effective_dim_end
}

def computeEnergy(config,ps,C, L, dL):
    C_shape = np.reshape(C, (config["dim"], config["deg"] + 1))
    X = np.matmul(C_shape, L)
    dX = np.matmul(C_shape, dL)
    W_Jw = W_JW (config, ps, X)
    E = 0.0
    for i in range(config["N"]+1):
        E = E + np.matmul(dX[:,i].reshape(1,dX.shape[0]),np.matmul(np.linalg.inv(W_Jw["W"][i]),dX[:,i].reshape(dX.shape[0],1)) * ps["weights"][i]) #INVERSE
    return E

from scipy.optimize import minimize, LinearConstraint
constraint = np.matmul(A,C) + b

res = minimize(
        computeEnergy,
        C,
        args = (config, ps, L, dL),
        constraints=np.asarray(constraint)
        #options={'gtol':1e-1, 'disp':True}
    )


#
# ps = ps_params(config)
# C = np.zeros(config["dim"] * (config["deg"] + 1)).reshape(config["dim"] * (config["deg"] + 1),1)
# E0 = 0
# L = 0
# dL = 0
# C = np.zeros(config["dim"] * (config["deg"] + 1)).reshape((config["dim"] * (config["deg"] + 1)), 1)
# L = np.hstack([_chebpoly(config, ni, ps) for ni in ps["nodes"]])
# dL = np.hstack([DiffChebyshevPolynomial(config, ni, ps, L) for ni in ps["nodes"]])
#
# for q in range(config["dim"]):
#     C[(config["deg"] + 1) * (q - 1) + 1 - 1] = 0.5 * (xstar[q] + xcurr[q])
#     C[(config["deg"] + 1) * (q - 1) + 2 - 1] = 0.5 * (xcurr[q] - xstar[q])
#
# A, b = Constraints(config, ps, xstar, xcurr)
#
# # np.matmul(A,C) + b constraint for scipy


