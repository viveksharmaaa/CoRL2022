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

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 13
HUGE_SIZE = 25

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=HUGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=HUGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)

left = 0.14  # the left side of the subplots of the figure
right = 0.98   # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.925     # the top of the subplots of the figure

parser = argparse.ArgumentParser(description="")
parser.add_argument('--task', type=str,
                        default='CAR')
parser.add_argument('--pretrained', type=str)
parser.add_argument('--plot_type', type=str, default='2D')
parser.add_argument('--plot_dims', nargs='+', type=int, default=[0,1])
parser.add_argument('--nTraj', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--sigma', type=float, default=0.)
parser.add_argument('--xd0scaler',type=float, default=0.5)
parser.add_argument('--x0scaler',type=float, default=0.5)
args = parser.parse_args()

np.random.seed(args.seed)

system = importlib.import_module('system_'+args.task)
f, B, _, num_dim_x, num_dim_control = get_system_wrapper(system)
controller = get_controller_wrapper(args.pretrained)

if __name__ == '__main__':
    config = importlib.import_module('config_'+args.task)
    t = config.t
    time_bound = config.time_bound
    time_step = config.time_step
    XE_INIT_MIN = config.XE_INIT_MIN
    XE_INIT_MAX = config.XE_INIT_MAX
    # x_0,xstar_0, ustar = config.system_reset(np.random.rand()) #x_0, xstar_0
    # #added
    # #x_0 = args.x0scaler * (config.X_MIN.reshape(-1) + np.random.rand(len(config.X_MIN.reshape(-1))) * (config.X_MAX.reshape(-1) - config.X_MIN.reshape(-1)))
    # #xstar_0 = args.xd0scaler * (config.X_MIN.reshape(-1) + np.random.rand(len(config.X_MIN.reshape(-1))) * (config.X_MAX.reshape(-1) - config.X_MIN.reshape(-1)))
    # print(x_0)
    # print(xstar_0)
    # #added
    # ustar = [u.reshape(-1,1) for u in ustar]
    # xstar_0 = xstar_0.reshape(-1,1)
    # xstar, _ = EulerIntegrate(None, f, B, None, ustar, xstar_0, time_bound, time_step, with_tracking=False)


    fig = plt.figure(figsize=(8.0, 5.0))
    if args.plot_type=='3D':
        ax = fig.gca(projection='3d')
    else:
        ax = fig.gca()

    if args.plot_type == 'time':
        cmap = plt.get_cmap('plasma')
        colors = [cmap(i) for i in np.linspace(0, 1, len(args.plot_dims))]

    x_closed = []
    controls = []
    errors = []
    xinits = []
    x_star = [] #added
    xstar0 = [] #added
    for _ in range(args.nTraj):
        #added different xstar in the loop
        x_0, xstar_0, ustar = config.system_reset(np.random.rand())
        ustar = [u.reshape(-1, 1) for u in ustar]
        xstar_0 = xstar_0.reshape(-1, 1)
        xstar, _ = EulerIntegrate(None, f, B, None, ustar, xstar_0, time_bound, time_step, with_tracking=False)
        #added different xstar in the loop


        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        #xinit = args.x0scaler * (config.X_MIN + np.random.rand(len(config.X_MIN)).reshape(-1,1) * (config.X_MAX - config.X_MIN)) #xstar_0 + xe_0.reshape(-1,1)
        xinit = xstar_0 + xe_0.reshape(-1, 1)
        #print(xinit)
        xinits.append(xinit)
        x, u = EulerIntegrate(controller, f, B, xstar,ustar,xinit,time_bound,time_step,with_tracking=True,sigma=args.sigma)
        x_closed.append(x)
        controls.append(u)
        x_star.append(xstar)
        xstar0.append(xstar_0)

    for n_traj in range(args.nTraj):
        initial_dist = np.sqrt(((x_closed[n_traj][0] - x_star[n_traj][0])**2).sum())
        errors.append([np.sqrt(((x-xs)**2).sum()) / initial_dist for x, xs in zip(x_closed[n_traj][:-1], x_star[n_traj][:-1])]) #xstar

        if args.plot_type=='2D':
            plt.plot([x[args.plot_dims[0],0] for x in x_closed[n_traj]], [x[args.plot_dims[1],0] for x in x_closed[n_traj]], 'g', label='closed-loop traj' if n_traj==0 else None)
        elif args.plot_type=='3D':
            plt.plot([x[args.plot_dims[0],0] for x in x_closed[n_traj]], [x[args.plot_dims[1],0] for x in x_closed[n_traj]], [x[args.plot_dims[2],0] for x in x_closed[n_traj]], 'g', label='closed-loop traj' if n_traj==0 else None)
        elif args.plot_type=='time':
            for i, plot_dim in enumerate(args.plot_dims):
                plt.plot(t, [x[plot_dim,0] for x in x_closed[n_traj]][:-1], color=colors[i])
        elif args.plot_type=='error':
            plt.plot(t, [np.sqrt(((x-xs)**2).sum()) for x, xs in zip(x_closed[n_traj][:-1],  x_star[n_traj][:-1])], 'g')

        #added

        if args.plot_type == '2D':
            plt.plot([x[args.plot_dims[0], 0] for x in x_star[n_traj]],
                     [x[args.plot_dims[1], 0] for x in x_star[n_traj]], 'k--', label='Reference' if n_traj==0 else None)
            plt.plot(xstar0[n_traj][args.plot_dims[0]], xstar0[n_traj][args.plot_dims[1]], 'ro',
                     markersize=3.)  # xstar_0
            plt.plot(xinits[n_traj][args.plot_dims[0]], xinits[n_traj][args.plot_dims[1]], 'bs', markersize=3.)
            plt.xlabel("x")
            plt.ylabel("y")
        elif args.plot_type == '3D':
            plt.plot([x[args.plot_dims[0], 0] for x in x_star[n_traj]],
                     [x[args.plot_dims[1], 0] for x in x_star[n_traj]],
                     [x[args.plot_dims[2], 0] for x in x_star[n_traj]], 'k--', label='Reference' if n_traj==0 else None)
            plt.plot(xstar0[n_traj][args.plot_dims[0]], xstar0[n_traj][args.plot_dims[1]],
                     xstar0[n_traj][args.plot_dims[2]], 'ro', markersize=3.)
            plt.plot(xinits[n_traj][args.plot_dims[0]], xinits[n_traj][args.plot_dims[1]],
                     xinits[n_traj][args.plot_dims[2]], 'bs', markersize=3.)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        elif args.plot_type == 'time':
            for plot_dim in args.plot_dims:
                plt.plot(t, [x[plot_dim, 0] for x in x_star[n_traj]][:-1], 'k')
            plt.xlabel("t")
            plt.ylabel("x")
        elif args.plot_type == 'error':
            plt.xlabel("t")
            plt.ylabel("error")

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    plt.legend(frameon=True)
    plt.show()
