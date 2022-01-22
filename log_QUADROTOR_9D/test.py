from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from np2pth import get_system_wrapper, get_controller_wrapper

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