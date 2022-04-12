import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from np2pth import get_system_wrapper, get_controller_wrapper
from matplotlib import pyplot as plt

import importlib
from utils import EulerIntegrate
import time

import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')

#np.random.seed(0)

task = 'QUADROTOR_8D'
nTraj = 250
plot_type = '2D'
plot_dims = [0,1] # refer to state x1 and x2

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 13
HUGE_SIZE = 25

left = 0.14  # the left side of the subplots of the figure
right = 0.98   # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.925     # the top of the subplots of the figure

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=HUGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=HUGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)

system = importlib.import_module('system_'+ task)
#geod = importlib.import_module('pseudospec_CAR')
f, B, _, num_dim_x, num_dim_control = get_system_wrapper(system)
controller = get_controller_wrapper('log_'+ task + '/controller_best.pth.tar')

if __name__ == '__main__':
    config = importlib.import_module('config_'+ task)
    t = config.t
    time_bound = config.time_bound
    time_step = config.time_step
    XE_INIT_MIN = config.XE_INIT_MIN
    XE_INIT_MAX = config.XE_INIT_MAX

    #x_0, xstar_0, ustar = config.system_reset(np.random.rand())
    _, _, ustar = config.system_reset(np.random.rand())
    # added x_0 and xstar_0 sampled for mathcal{X} but scaled by a number to keep the traj in mathcal{X}
    x_0 = 0.5 * (config.X_MIN.reshape(-1) + np.random.rand(len(config.X_MIN.reshape(-1))) * (config.X_MAX.reshape(-1) - config.X_MIN.reshape(-1)))
    xstar_0 = 0.5 * (config.X_MIN.reshape(-1) + np.random.rand(len(config.X_MIN.reshape(-1))) * (config.X_MAX.reshape(-1) - config.X_MIN.reshape(-1)))
    print(x_0)
    print(xstar_0)
    # added
    ustar = [u.reshape(-1,1) for u in ustar]
    xstar_0 = xstar_0.reshape(-1,1)
    xstar, _ = EulerIntegrate(None, f, B, None, ustar, xstar_0, time_bound, time_step, with_tracking=False)

    fig = plt.figure(figsize=(8.0, 5.0))
    if plot_type=='3D':
        ax = fig.gca(projection='3d')
    else:
        ax = fig.gca()

    if plot_type == 'time':
        cmap = plt.get_cmap('plasma')
        colors = [cmap(i) for i in np.linspace(0, 1, len(plot_dims))]

    x_closed = []
    controls = []
    errors = []
    xinits = []
    myList = []

    plot_type = '2D'
    for _ in range(nTraj):
        #xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        #xinit =  xstar_0 + xe_0.reshape(-1, 1)
        #x0 sampled from mathcal{X} but scaled by a factor to keep traj in mathcal{X}
        #added
        #xinit = 0.001 * (config.X_MIN + np.random.rand(len(config.X_MIN)).reshape(-1, 1) * (config.X_MAX - config.X_MIN))
        xinit_min = 10 * (config.X_INIT_MIN + config.XE_INIT_MIN)
        xinit_max = 10 * (config.X_INIT_MAX + config.XE_INIT_MAX)
        xinit = xinit_min.reshape(-1, 1) + np.random.rand(len(xinit_min)).reshape(-1, 1) * (
                    xinit_max - xinit_min).reshape(-1, 1)
        xinit[xinit > config.X_MAX] = config.X_MAX[xinit > config.X_MAX]
        xinit[xinit < config.X_MIN] = config.X_MIN[xinit < config.X_MIN]
        #added
        xinits.append(xinit)
        x, u = EulerIntegrate(controller, f, B, xstar, ustar, xinit, time_bound, time_step, with_tracking=True,sigma=0)
        x_closed.append(x)
        controls.append(u)

    for n_traj in range(nTraj):
        initial_dist = np.sqrt(((x_closed[n_traj][0] - xstar[0]) ** 2).sum())
        errors.append([np.sqrt(((x - xs) ** 2).sum()) / initial_dist for x, xs in zip(x_closed[n_traj][:-1], xstar)])
        if plot_type == '2D':
            plt.plot([x[plot_dims[0], 0] for x in x_closed[n_traj]], [x[plot_dims[1], 0] for x in x_closed[n_traj]],
                     'g',
                     label='closed-loop traj' if n_traj == 0 else None)
        elif plot_type == '3D':
            plt.plot([x[plot_dims[0], 0] for x in x_closed[n_traj]], [x[plot_dims[1], 0] for x in x_closed[n_traj]],
                     [x[plot_dims[2], 0] for x in x_closed[n_traj]], 'g',
                     label='closed-loop traj' if n_traj == 0 else None)
        elif plot_type == 'time':
            for i, plot_dim in enumerate(plot_dims):
                plt.plot(t, [x[plot_dim, 0] for x in x_closed[n_traj]][:-1], color=colors[i])
        elif plot_type == 'error':
            plt.plot(t, [np.sqrt(((x - xs) ** 2).sum()) for x, xs in zip(x_closed[n_traj][:-1], xstar)], 'g')

    if plot_type == '2D':
        plt.plot([x[plot_dims[0], 0] for x in xstar], [x[plot_dims[1], 0] for x in xstar], 'k--',
                 label='Reference')
        plt.plot(xstar_0[plot_dims[0]], xstar_0[plot_dims[1]], 'ro', markersize=3.)
        plt.xlabel("x")
        plt.ylabel("y")
    elif plot_type == '3D':
        plt.plot([x[plot_dims[0], 0] for x in xstar], [x[plot_dims[1], 0] for x in xstar],
                 [x[plot_dims[2], 0] for x in xstar], 'k--', label='Reference')
        plt.plot(xstar_0[plot_dims[0]], xstar_0[plot_dims[1]], xstar_0[plot_dims[2]], 'ro',
                 markersize=3.)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    elif plot_type == 'time':
        for plot_dim in plot_dims:
            plt.plot(t, [x[plot_dim, 0] for x in xstar][:-1], 'k')
        plt.xlabel("t")
        plt.ylabel("x")
    elif plot_type == 'error':
        plt.xlabel("t")
        plt.ylabel("error")

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    plt.legend(frameon=True)
    plt.show()


# Code to Save x and xstar data points from the trajectory
    Xstar = np.asarray(xstar)
    Xcurr = np.asarray(x_closed)

    Xstar = np.tile(Xstar,(len(Xcurr),1,1))
    Xclosed = np.concatenate(Xcurr, axis=0)
    #Xstar = Xstar.reshape(Xstar.shape[0],Xstar.shape[1],1)
    #Xclosed = Xcurr.reshape(Xcurr.shape[1], Xcurr.shape[2],1)
    np.savez(task + '_closed_pts.npz',Xstar.reshape(Xstar.shape[0],Xstar.shape[1]), Xclosed.reshape(Xclosed.shape[0],Xclosed.shape[1]))
    npzfile = np.load(task + '_closed_pts.npz')

    # npzfile.files
    # npzfile['arr_0']



    #Use [0,:] for data points
    # start_time = time.time()
    # for i in range(len(Xstar)):
    #     ps_result = geod.pseudospectral_geodesic(Xstar[i], Xclosed[i])
    #     print(ps_result["E0"])
    #     myList.append({"xstar": Xstar[i], "x": Xclosed[i], "RE": ps_result["E0"]})
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print("DONE")
    #
    # with open('data_QUADROTOR8D_closed_40k_pts.pkl', 'wb') as f:
    #    pickle.dump(myList, f)


xinit_min = 1.1* (config.X_INIT_MIN + config.XE_INIT_MIN)
xinit_max = 1.1 * (config.X_INIT_MAX + config.XE_INIT_MAX)
xinit =  xinit_min.reshape(-1, 1) + np.random.rand(len(xinit_min)).reshape(-1, 1) * (xinit_max - xinit_min).reshape(-1, 1)
xinit[xinit > config.X_MAX] = config.X_MAX[xinit > config.X_MAX]
xinit[xinit < config.X_MIN] = config.X_MIN[xinit < config.X_MIN]



sum(np.asarray(xinits)[i,:,:] > (config.X_INIT_MAX + config.XE_INIT_MAX).reshape(-1,1) for i in range(len(xinits)))

sum(np.linalg.norm(np.asarray(xinits)[i,:,:]) > np.linalg.norm((config.X_INIT_MAX + config.XE_INIT_MAX).reshape(-1,1)) for i in range(len(xinits)))