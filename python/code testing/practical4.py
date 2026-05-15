# ==============================
# Experiment 4: NumPy & Matrix Operations
# ==============================

import numpy as np

# -------- Array Creation --------

voltage = np.array([220, 225, 230, 228])
print("Voltage readings:", voltage)

# -------- Basic Operations --------

print("Maximum Voltage:", np.max(voltage))
print("Minimum Voltage:", np.min(voltage))
print("Average Voltage:", np.mean(voltage))
print("Sum of voltages:", np.sum(voltage))

# -------- Power Calculation --------

voltage = np.array([220, 225, 230])
current = np.array([5, 5.2, 4.8])

power = voltage * current
print("Power values:", power)

# -------- Special Arrays --------

zeros_array = np.zeros(5)
print(zeros_array)

ones_array = np.ones(5)
print(ones_array)

# -------- Reshaping --------

data = np.array([1,2,3,4,5,6])
matrix = data.reshape(2,3)
print(matrix)

# -------- Matrix Creation --------

A = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

print("Matrix A:\n", A)

# -------- Arange --------

B = np.arange(1,13)
print(B)
C = B.reshape(3,4)
print(C)

# -------- Accessing Elements --------

print("Element:", A[1,2])
print("Row:", A[1])
print("Column:", A[:,1])
print("Submatrix:\n", A[0:2,0:2])

# -------- Delete Row & Column --------

B = np.delete(A, 1, axis=0)
print("After deleting row:\n", B)

C = np.delete(A, 1, axis=1)
print("After deleting column:\n", C)

# -------- Add Row & Column --------

new_row = [10,11,12]
D = np.vstack((A, new_row))
print("After adding row:\n", D)

new_col = [[10],[11],[12]]
E = np.hstack((A, new_col))
print("After adding column:\n", E)

# -------- Flatten --------

flat = A.flatten()
print("Flattened:", flat)

# -------- Transpose --------

G = A.T
print("Transpose:\n", G)

# -------- Matrix Multiplication --------

A1 = np.array([[1,2],
               [3,4]])

B1 = np.array([[5,6],
               [7,8]])

C1 = np.matmul(A1, B1)
print("Matrix Multiplication:\n", C1)