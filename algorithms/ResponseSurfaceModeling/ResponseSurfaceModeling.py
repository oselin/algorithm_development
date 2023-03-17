import numpy as np

import numpy as np
from scipy.optimize import minimize



# Define the response surface model
def response_surface(x, a, b, c, d, e, f):
    return a + b*x[0] + c*x[1] + d*x[2] + e*x[0]**2 + f*x[1]**2

# Define the constraints
def constraint1(x):
    return 1.0 - x[0]**2 - x[1]**2 - x[2]**2

# Define the bounds
b = (0.0, 1.0)
bounds = (b, b, b)

# Generate the experimental design
X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
Y = np.array([objective(x) for x in X])

# Calculate the coefficients of the response surface model
A = np.column_stack((np.ones(6), X[:,0], X[:,1], X[:,2], X[:,0]**2, X[:,1]**2))
B = np.linalg.solve(A, Y)

# Define the initial guess for the optimization
x0 = np.array([0.5, 0.5, 0.5])

# Define the objective function for the optimization
def objective_for_optimization(x):
    return response_surface(x, *B)

# Define the constraints for the optimization
constraint1_for_optimization = {'type': 'ineq', 'fun': constraint1}

# Perform the optimization
result = minimize(objective_for_optimization, x0, method='SLSQP', bounds=bounds, constraints=[constraint1_for_optimization])

# Print the results
print('Optimal solution: x =', result.x)
print('Objective function value at optimal solution: f(x) =', result.fun)
