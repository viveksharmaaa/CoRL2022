import importlib
import numpy as np
import time
import pickle

np.random.seed(0)
task = 'SEGWAY'

geod = importlib.import_module('pseudospec_CAR')
system = importlib.import_module('system_'+task)

# #Run the following lines
num_train = 40000
num_test = 1
x,xref,uref = geod.data_sets(num_train,num_test,system.num_dim_x, system.num_dim_control)

xstar = xref.numpy().reshape(num_train,system.num_dim_x,1)
xcurr = x.numpy().reshape(num_train,system.num_dim_x,1)

myList = []

start_time = time.time()
for i in range(num_train):
    ps_result = geod.pseudospectral_geodesic(xstar[i], xcurr[i])
    print(ps_result["E0"])
    myList.append({"xstar": xstar[i], "x": xcurr[i], "RE": ps_result["E0"]})
print("--- %s seconds ---" % (time.time() - start_time))
print("Done")

with open('data_SEGWAY_sampled_40k.pkl', 'wb') as f:
    pickle.dump(myList, f)