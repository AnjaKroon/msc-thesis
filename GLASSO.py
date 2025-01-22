# Time varying Graphical LASSO for dynamic connectivity estimation
import numpy as np
from sklearn.covariance import graphical_lasso
from syntheticDataGen import generate_data_matrices 

def time_varying_graphical_lasso(Y, alpha=0.1, beta=0.05, max_iter=10, tol=1e-4):
    """
    Perform Time-Varying Graphical Lasso to estimate time-varying weighted adjacency matrices.
    
    Parameters:
        Y (numpy.ndarray): Data matrix of shape (N, T), where N is the number of nodes, and T is the number of time steps.
        alpha (float): L1 regularization parameter for graphical lasso.
        beta (float): Regularization parameter for temporal smoothness between consecutive time steps.
        max_iter (int): Maximum number of iterations for optimization.
        tol (float): Convergence tolerance for optimization.
    
    Returns:
        A_matrices (list): List of weighted adjacency matrices for each time step.
    """
    N, T = Y.shape
    A_matrices = [None] * T
    S = [np.cov(Y[:, t:t + 2]) for t in range(T - 1)]  

    
    for t in range(T):
        emp_cov = np.cov(Y[:, max(0, t - 1):min(T, t + 2)])  
        _, A = graphical_lasso(emp_cov, alpha=alpha)
        A_matrices[t] = A

    # Optimization loop
    for it in range(max_iter):
        max_diff = 0
        for t in range(1, T):
            # Smoothness regularization between A_t and A_{t-1}
            S_t = np.cov(Y[:, max(0, t - 1):min(T, t + 2)])
            _, A_new = graphical_lasso(S_t + beta * A_matrices[t - 1], alpha=alpha)
            diff = np.linalg.norm(A_new - A_matrices[t], ord='fro')
            max_diff = max(max_diff, diff)
            A_matrices[t] = A_new

        if max_diff < tol:
            break

    return A_matrices


if __name__ == "__main__":
    
    NODES = 10
    SAMPLES = 50
    K_PARAM = 1
    EPSILON = 1e-10
    VALUE_THRESHOLD = 1e10
    ALPHA = 0.95
    NOISE_MEAN, NOISE_COVAR = 1, 0.5
    INITIAL_WEIGHT_MEAN, INITIAL_WEIGHT_VAR = 1, 0.01
    INITIAL_SIGNAL_MEAN, INITIAL_SIGNAL_VAR = 1, 0.01
    DECREASE = 0.8
    INCREASE = 1.2

    # Y_static, Y_dynamic_piecewiseSlow, Y_dynamic_smooth = generate_data_matrices(N=10, M=10, K=1)
    Generated_Y_static_SEM, \
        Generated_Y_static_SVARM, \
        Generated_Y_dynamic_fixedF, \
        Generated_Y_dynamicSVARM_fixedF, \
        Generated_Y_dynamic_piecewiseSlow_SEM, \
        Generated_Y_dynamic_piecewiseMedium_SEM, \
        Generated_Y_dynamic_piecewiseFast_SEM, \
        Generated_Y_dynamic_piecewiseSlow_SVARM, \
        Generated_Y_dynamic_piecewiseMedium_SVARM, \
        Generated_Y_dynamic_piecewiseFast_SVARM = generate_data_matrices(N=3, M=10, K=1, EP=1e-10, \
                                                                     VT=1e10, AL=0.9, NM=5, NC=0.1,\
                                                                         IWM=5, IWV=3, \
                                                                            ISM=20, ISV=5, DF=0.97, IF=1.03)
    
    alpha = 0.1
    beta = 0.05
    
    A_matrices = time_varying_graphical_lasso(Generated_Y_static_SEM, alpha=alpha, beta=beta)

    for t, A in enumerate(A_matrices):
        print(f"Adjacency matrix at time {t}:\n{np.round(A, 2)}\n")

