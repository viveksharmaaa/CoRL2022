import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize

objective_function = lambda x: (4*x[0]**2) - (4*x[0]**4) + (x[0]**(6/3) + (x[0]*x[1]) - (4*x[1]**2) + (4*x[1]**4))

X_1 = np.linspace(0,1, 100)
X_2 = np.linspace(0,1, 100)
X_1, X_2 = np.meshgrid(X_1, X_2)
Z = objective_function((X_1, X_2))

#Contour plot
fig, ax = plt.subplots()
fig.set_size_inches(14.7, 8.27)

cs = ax.contour(X_1, X_2, Z, 50, cmap='jet')
plt.clabel(cs, inline=1, fontsize=10) # plot objective function

plt.axvline(0.1, color='g', label=r'$x_1 \geq 0.1$') # constraint 1
plt.axhline(0.25, color='r', label=r'$x_2 \geq 0.25$') # constraint 2
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1)

# tidy up
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.show()


def geneate_starting_points(number_of_points):
    '''
    number_of_points: how many points we want to generate

    returns a list of starting points
    '''
    starting_points = []

    for point in range(number_of_points):
        starting_points.append((random.random(), random.random()))

    return starting_points

# define our objective function
objective_function = lambda x: np.matmul(np.random.rand(1,2),x) #(4*x[0]**2) - (4*x[0]**4) + (x[0]**(6/3) + (x[0]*x[1]) - (4*x[1]**2) + (4*x[1]**4))

# define our constraints
# note that since we have >= constraints we use the 'ineq' constaint type
cons = [
    {'type': 'ineq', 'fun': lambda x: x[0] - 0.1}, # indoor seating >= 0.1
    {'type': 'ineq', 'fun': lambda x: x[1] - 0.25} # outdoor seating >= 0.25
]

# define boundaries
boundaries = [(0,1), (0,1)]


# generate a list of N potential starting points
starting_points = geneate_starting_points(50)

first_iteration = True
for point in starting_points:
    # for each point run the algorithim
    res = minimize(
        objective_function,
        [point[0], point[1]],
        method='SLSQP',
        bounds=boundaries,
        constraints=cons
    )
    # first iteration always gonna be the best so far
    if first_iteration:
        better_solution_found = False
        best = res
    else:
        # if we find a better solution, lets use it
        if res.success and res.fun < best.fun:
            better_solution_found = True
            best = res

# print results if algorithim was successful
if best.success:
    print(f"""Optimal solution found:
      -  Proportion of indoor seating to make available: {round(best.x[0], 3)}
      -  Proportion of outdoor seating to make available: {round(best.x[1], 3)}
      -  Risk index score: {round(best.fun, 3)}""")
else:
    print("No solution found to problem")