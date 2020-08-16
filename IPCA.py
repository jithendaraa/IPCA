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


def main():
    def objective(x):
        m = actual_m
        S = np.abs(np.diag(np.array(x)))
        A_k = constraint_matrix_estimates[k].copy()
        S_r = np.dot(np.dot(A_k, S), A_k.T)
        S_r_inverse = np.linalg.inv(S_r)
        second_term = 0.0
        for i in range(N):
            second_term += np.sum(np.dot(np.dot(R[i].T, S_r_inverse), R[i]))
        res = np.log(np.linalg.det(S_r)) + second_term/N
        return res
    # Step 1
    k = 1 
    lambdas = [0]
    # Step 2
    error_cov_matrix_estimates = {}
    constraint_matrix_estimates = {}
    error_cov_matrix_estimate = np.zeros((n, n))
    init_multiplier = 1e-4
    init_estimate = np.diag([init_multiplier] * 5)
    S_y = np.dot(Y, Y.T)/N
    
    for i in range(len(init_estimate)):
        init_estimate[i][i] = S_y[i][i] * init_estimate[i][i]

    error_cov_matrix_estimates[k] = init_estimate.copy()
    objectives = []
    predicted_A = []
    for i in range(10):
        # Step 3 
        S_k = error_cov_matrix_estimates[k].copy()
        L = get_cholesky_matrix(S_k)
        L_inverse = np.linalg.inv(L)
        Y_scaled = np.dot(L_inverse, Y)
        # Step 4
        U, S, V = np.linalg.svd(Y_scaled)
        least_m_eigen_vecs_U = U.T[-actual_m:].T
        A_k = np.dot(least_m_eigen_vecs_U.T, L_inverse).copy()
        constraint_matrix_estimates[k] = A_k.copy()
        # Step 5
        curr_lambda = np.sum(S[-actual_m:])
        prev_lambda = lambdas[k-1]
        rel_lambda_change = (curr_lambda - prev_lambda)/curr_lambda
        # print("EIGEN SUM: ", curr_lambda)
        # print("REL LAMBDA CHANGE: ", rel_lambda_change)
        lambdas.append(curr_lambda)
        # Step 6
        R = {}
        flattened_R = None
        for j in range(N):
            R[j] = np.dot(A_k, Y.T[j]).reshape(actual_m, 1)
            if j == 0:
                flattened_R = R[j].reshape(R[j].shape[0])
            else:
                flattened_R = np.concatenate((flattened_R, R[j].reshape(R[j].shape[0])))
        x_k = []
        for i in range(n):
            x_k.append(S_k[i][i])
        objectives.append(objective(x_k))
        print("Objective:", objective(x_k))
        if np.abs(rel_lambda_change) < 1e-12:
            break
        #   Todo: Impose lower bouns to ensure +ve definite
        b = (1e-3, 1e10)
        bnds = (b,b,b,b,b)
        sol = minimize(objective, x_k, bounds=bnds, method='SLSQP')
        res = sol.x
        next_S = np.abs(np.diag(np.array(res)))
        k += 1
        error_cov_matrix_estimates[k] = next_S
    
if __name__ == "__main__":
    main()


    