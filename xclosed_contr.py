import numpy as np
from np2pth import get_system_wrapper, get_controller_wrapper
import pickle

import importlib

import importlib
from utils import EulerIntegrate
import time

import os
import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')

np.random.seed(0)

task = 'CAR'
nTraj = 150

system = importlib.import_module('system_'+ task)
geod = importlib.import_module('pseudospec_CAR')
f, B, _, num_dim_x, num_dim_control = get_system_wrapper(system)
controller = get_controller_wrapper('log_CAR/controller_best.pth.tar')


if __name__ == '__main__':
    config = importlib.import_module('config_'+ task)
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
    xinits = []
    myList = []
    for _ in range(nTraj):
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        xinit = xstar_0 + xe_0.reshape(-1, 1)
        xinits.append(xinit)
        x, u = EulerIntegrate(controller, f, B, xstar, ustar, xinit, time_bound, time_step, with_tracking=True,
                              sigma=0.2)
        x_closed.append(x)
        controls.append(u)

    Xstar = np.asarray(xstar)
    Xcurr = np.asarray(x_closed)

    Xstar = np.tile(Xstar,(len(Xcurr),1,1))
    Xclosed = np.concatenate(Xcurr, axis=0)

    #Xstar = Xstar.reshape(Xstar.shape[0],Xstar.shape[1],1)
    #Xclosed = Xcurr.reshape(Xcurr.shape[1], Xcurr.shape[2],1)

    start_time = time.time()
    for i in range(len(Xstar)):
        ps_result = geod.pseudospectral_geodesic(Xstar[i], Xclosed[i])
        print(ps_result["E0"])
        myList.append({"xstar": Xstar[i], "x": Xclosed[i], "RE": ps_result["E0"]})
    print("--- %s seconds ---" % (time.time() - start_time))
    print("DONE")

    with open('data_closed_x0_X.pkl', 'wb') as f:
       pickle.dump(myList, f)