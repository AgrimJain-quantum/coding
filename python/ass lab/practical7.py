from scipy.optimize import linprog
c = [3, 5, 4]
A = [
 [-1, -1, -1],
 [2, 1, 3],
 [1, 4, 2]
]
b = [-10, 20, 25]
bounds = [(0, None), (0, None), (0, None)]
result = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
print(result.success)
print("x =", result.x[0])
print("y =", result.x[1])
print("z =", result.x[2])
print("Minimum cost =", result.fun)