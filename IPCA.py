import numpy as np
import math
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.linalg import subspace_angles

def get_true_values(n, N, base_values, fluctuations, actual_m):
    X = np.zeros((n, N))
    for i in range(n):
        if base_values[i] == None:
            row = dependent_variable_start_idx + i - actual_m - 1
            constraint_row = actual_constraint_matrix[row].tolist()
            variable_num = dependent_variable_start_idx + i - 1
            dependent_var_coeff = -constraint_row[variable_num-1]
            
            for j in range(0, variable_num-1):
                X[i] = np.add(X[i], constraint_row[j] * X[j])
            
            X[i] = X[i]/dependent_var_coeff
            continue
        low  = base_values[i] - fluctuations[i]
        high = base_values[i] + fluctuations[i]
        X[i] = np.random.uniform(low, high, size=N)
    return X

def get_normal_errors(error_means, error_std_devs):
    errors = np.ones((n, N))
    for i in range(n):
        errors[i] = np.random.normal(loc = error_means[i], scale = error_std_devs[i])
    return errors

def get_diagonal_cholesky_element(A, xpos, ypos, L):
    element = None
    if xpos == 0:
        element = A[xpos][ypos]**0.5
    else:
        horizontal_sq_sum = 0.0
        for j in range(ypos):
            horizontal_sq_sum += (L[xpos][j]**2)
        element = (A[xpos][ypos] - horizontal_sq_sum) ** 0.5
    return element

def get_non_diagonal_cholesky_element(A, xpos, ypos, L):
    element = None
    if xpos == 0:
        element = A[xpos][ypos]/L[ypos][ypos]
        return element
    product_sum = 0.0
    for j in range(ypos):
        product_sum += (L[ypos][j] * L[xpos][j])
    element = (A[xpos][ypos] - product_sum)/L[ypos][ypos]
    return element

def get_cholesky_element(A, xpos, ypos, L):
    element = None
    if xpos == ypos:
        element = get_diagonal_cholesky_element(A, xpos, ypos, L)
    else:
        element = get_non_diagonal_cholesky_element(A, xpos, ypos, L)
    return element

def get_cholesky_matrix(A):
    matrix_size = A.shape[0]
    L = np.zeros((matrix_size, matrix_size))
    for i in range(matrix_size):
        for j in range(i+1):
            L[i][j] = get_cholesky_element(A, i, j, L)
    return L

actual_constraint_matrix = np.array(([1, 1, -1, 0, 0], 
                                     [0, 0, 1, -1, 0], 
                                     [0, -1, 0, 1, -1]))
actual_A = actual_constraint_matrix.copy()
n = actual_constraint_matrix.shape[1]
actual_model_order = actual_constraint_matrix.shape[0]
actual_m = actual_constraint_matrix.shape[0]
N = 1000
dependent_variable_start_idx = n - actual_m
base_values  = [10, 10, None, None, None]
fluctuations = [1.0, 2.0, None, None, None]
X = get_true_values(n, N, base_values, fluctuations, actual_m)
error_means    = [0] * n
error_std_devs = [0.1, 0.08, 0.15, 0.2, 0.18]
errors = get_normal_errors(error_means, error_std_devs)
Y = np.add(X, errors)