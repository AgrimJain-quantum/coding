import numpy as np
from scipy.optimize import minimize
# Objective function
def objective(x):
 return (x[0] - 1)**2 + (x[1] - 2)**2
# Inequality constraint: x + y - 2 >= 0
def ineq_constraint(x):
 return x[0] + x[1] - 2
# Equality constraint: x - y = 0
def eq_constraint(x):
 return x[0] - x[1]
bounds = [(0, None), (0, None)]
constraints = [
 {'type': 'ineq', 'fun': ineq_constraint},
 {'type': 'eq', 'fun': eq_constraint}
]
x0 = [0.5, 0.5]
solution = minimize(objective, x0, method='SLSQP',
 bounds=bounds,
 constraints=constraints)
print("x =", solution.x[0])
print("y =", solution.x[1])
print("Minimum value =", solution.fun)