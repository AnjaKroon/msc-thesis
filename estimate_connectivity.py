import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import time
import seaborn as sns
import sys
import math
import numpy as np
import cvxpy as cp

from MyDataGen_GGLASSOSolver import time_varying_graphical_lasso
from gglasso.problem import glasso_problem
from scipy.optimize import minimize
from MyDataGen import generate_data_matrices 
from scipy.stats import rankdata 
from tqdm import tqdm

def calculate_optimal_F_H(y_t_minus_1, y_t_minus_2, y_t_minus_3, 
                          h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4,
                          lambda_param, F_t_minus_1_optimal, H_t_optimal, ground_truth_H=None, THRESH_SCALE=0.1, H_t_minus_1=None):
    """
    Calculate optimal matrices F_{t-1}^* and H_t^* for a given set of inputs.

    Parameters:
        y_t_minus_1, y_t_minus_2, y_t_minus_3: Graph signals at previous timesteps.
        h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4: past state vectors.
        lambda_param (float): reg parameter.

    Returns:
        F_t_minus_1_optimal (numpy.ndarray): Optimized matrix F_{t-1}^*.
        H_t_optimal (numpy.ndarray): Optimized matrix H_t^*.
    """
    # INITIALIZATIONS 
    n = len(y_t_minus_1)
    Nodes = int(math.sqrt(h_t_minus_1.shape[0]))
    epsilon = 0.1
    y_t_minus_1 = y_t_minus_1.reshape(n, 1)
    y_t_minus_2 = y_t_minus_2.reshape(n, 1)
    y_t_minus_3 = y_t_minus_3.reshape(n, 1)


    # DEFINE OBJ FUNCT FOR H AND F
    def objective_H(H_flat):
        H = H_flat.reshape(n, n*n)
        part1 = np.linalg.norm(y_t_minus_3 - H @ h_t_minus_3, 2)
        part2 = np.linalg.norm(y_t_minus_2 - H @ h_t_minus_2, 2)
        part3 = np.linalg.norm(y_t_minus_1 - H @ h_t_minus_1, 2)

        reg = 0
        # group sparsity constraint by column on H
        for col in range(H.shape[1]):
            norm_col = np.linalg.norm(H[:, col], 2)
            if norm_col > 0:
                reg += norm_col
        reg = lambda_param * reg

        exp = (part1 + part2 + part3)

        # Temporal consistency regularization
        if H_t_minus_1 is not None:
            reg_temporal = np.linalg.norm(H - H_t_minus_1, 2)
        else:
            reg_temporal = 0 
        
        # return exp + 10*reg + 10*reg_temporal
        # return exp + reg + reg_temporal
        return exp
        # return part1

    def objective_F(F_flat):
        F = F_flat.reshape(n*n, n*n)
        part1 = np.linalg.norm(h_t_minus_3 - F @ h_t_minus_4, 2)
        part2 = np.linalg.norm(h_t_minus_2 - F @ h_t_minus_3, 2)
        part3 = np.linalg.norm(h_t_minus_1 - F @ h_t_minus_2, 2)
        # h1_1 - h2_1 <= ep, h1_2 = h2_2 <= ep, h1_3 = h2_3 <= ep, h1_4 = h2_4 <= ep ... h1_N^2 = h2_N^2 <= ep
    
        # F_vectorized = F.flatten()
        reg = lambda_param * np.linalg.norm(F, 'fro')
        reg_sparsity = lambda_param * np.linalg.norm(F, 1)
        return (1/3)*(part1 + part2 + part3) + reg + reg_sparsity
    
    def constraint_funct(F_flat):
        F = F_flat.reshape(n*n, n*n)
        h1_approx = (F @ h_t_minus_2).flatten()
        h2_approx = (F @ h_t_minus_3).flatten()
        h3_approx = (F @ h_t_minus_4).flatten()
        return np.concatenate([
            epsilon - (h1_approx - h2_approx),  # h1_i - h2_i <= epsilon
            epsilon - (h2_approx - h1_approx),  # h2_i - h1_i <= epsilon
            epsilon - (h2_approx - h3_approx),  # h2_i - h3_i <= epsilon
            epsilon - (h3_approx - h2_approx)   # h3_i - h2_i <= epsilon
        ])

    def get_symmetric_index_pairs(N):
        """Finds index pairs in the vectorized form of an NxN symmetric matrix that should be equal."""
        index_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                index1 = i * N + j
                index2 = j * N + i
                index_pairs.append((index1, index2))
        return index_pairs

    
    # THRESHOLDING THE H_INITIAL PREPROCESSING
    max_H_matrix = np.amax(np.abs(ground_truth_H))
    threshold = THRESH_SCALE * max_H_matrix
    ground_truth_H[np.abs(ground_truth_H) < threshold] = 0

    H_initial = H_t_minus_1.flatten() if H_t_minus_1 is not None else np.zeros(n, n*n).flatten() # first instance via moore-penrose pseudoinverse, later, last value

    F_initial = np.eye(n*n, n*n).flatten() if F_t_minus_1_optimal is None else F_t_minus_1_optimal.flatten()
    constraints_F = {'type': 'ineq', 'fun': constraint_funct}
    
    H_optimization = minimize(objective_H, H_initial, method='L-BFGS-B', options={'ftol': 1e-20, 'disp': False}) 
    H_optimal = H_optimization.x.reshape(n, n*n)


    # THRESHOLDING the H MATRIX POSTPROCESSING
    max_H_matrix = np.amax(np.abs(H_optimal))
    threshold = THRESH_SCALE * max_H_matrix
    H_optimal[np.abs(H_optimal) < threshold] = 0

    # DEBUGGING H OPTIMIZATION

    equal = np.allclose(H_optimal, ground_truth_H, atol=1e-1)
    if not equal:
        print("\033[91mH_optimal is NOT equal to ground truth H!\033[0m")  # Prints in red
        print("The difference between H_optimal and ground truth H is:", np.linalg.norm(H_optimal - ground_truth_H, 2))

    # print("H ground truth based on the pseudoinverse is")
    # printPretty(ground_truth_H)

    print("H optimal is solved to be")
    printPretty(H_optimal)
    
    # F_optimization = minimize(objective_F, F_initial, method='SLSQP', constraints=constraints_F)
    # F_optimal = F_optimization.x.reshape(n*n, n*n)
    F_optimal = np.eye(n*n, n*n)

    # DEBUGGING FOR H
    # if H_optimal @ h_t_minus_1 not close to y_t_minus_1, then there is a problem etc.
    maximum_value_in_y = np.amax(np.abs(y_t_minus_1))
    tolerance = maximum_value_in_y*n

    y1_approx = H_optimal @ h_t_minus_1
    y2_approx = H_optimal @ h_t_minus_2
    y3_approx = H_optimal @ h_t_minus_3

    # print("y1_approx", y1_approx, "\n y_t_minus_1", y_t_minus_1)
    # print("y2_approx", y2_approx, "\n y_t_minus_2", y_t_minus_2)
    # print("y3_approx", y3_approx, "\n y_t_minus_3", y_t_minus_3)

    assert np.allclose(y1_approx, y_t_minus_1, atol=tolerance), \
        f"H_optimal @ h_t_minus_1 is not close to y_t_minus_1! Difference: {np.linalg.norm(y1_approx - y_t_minus_1)}"

    assert np.allclose(y2_approx, y_t_minus_2, atol=tolerance), \
        f"H_optimal @ h_t_minus_2 is not close to y_t_minus_2! Difference: {np.linalg.norm(y2_approx - y_t_minus_2)}"

    assert np.allclose(y3_approx, y_t_minus_3, atol=tolerance), \
        f"H_optimal @ h_t_minus_3 is not close to y_t_minus_3! Difference: {np.linalg.norm(y3_approx - y_t_minus_3)}"

    # If F_optimal @ h_t_minus_1_approx is not close to h_t_approx, then there is a problem
    h3_approx = F_optimal @ h_t_minus_4
    h2_approx = F_optimal @ h_t_minus_3
    h1_approx = F_optimal @ h_t_minus_2

    assert np.allclose(h3_approx, h_t_minus_3, atol=tolerance), \
        f"F_optimal @ h_t_minus_4 is not close to h_t_minus_3! Difference: {np.linalg.norm(h3_approx - h_t_minus_3)}"
    
    assert np.allclose(h2_approx, h_t_minus_2, atol=tolerance), \
        f"F_optimal @ h_t_minus_3 is not close to h_t_minus_2! Difference: {np.linalg.norm(h2_approx - h_t_minus_2)}"
    
    assert np.allclose(h1_approx, h_t_minus_1, atol=tolerance), \
        f"F_optimal @ h_t_minus_2 is not close to h_t_minus_1! Difference: {np.linalg.norm(h1_approx - h_t_minus_1)}"

    return F_optimal, H_optimal, h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4

def proposed_kalman_filter(h_t_minus_1_post, F_t_minus_1_optimal, Q_t_minus_1, Sigma_t_minus_1_post, 
                  H_t_optimal, R_t_minus_1, y_t):
    """
    Perform Kalman filter prediction and correction steps.

    Parameters:
        h_t_minus_1_post (numpy.ndarray): h_{t-1}^+ (a posteriori estimate of the hidden state at t-1)
        F_t_minus_1_optimal (numpy.ndarray): F_{t-1}^* (optimal evolution matrix at t-1)
        Q_t_minus_1 (numpy.ndarray): Q_{t-1} (process noise covariance matrix at t-1)
        Sigma_t_minus_1_post (numpy.ndarray): Sigma_{t-1}^+ (a posteriori estimate of the covariance at t-1)
        H_t_optimal (numpy.ndarray): H_t^* (optimal measurement matrix at t)
        R_t_minus_1 (numpy.ndarray): R_{t-1} (measurement noise covariance matrix at t-1)
        y_t (numpy.ndarray): y_t (observation at time t)

    Returns:
        h_t_post (numpy.ndarray): h_t^+ (a posteriori estimate of the hidden state at t)
        y_t_post (numpy.ndarray): y_t^+ (predicted observation at time t)
    """

    y_t = y_t.reshape(R_t_minus_1.shape[0], 1)

    # print("Previous hidden state: h_t_minus_1_post")
    # printPretty(h_t_minus_1_post)
    # print("Incoming measurement: y_t")
    # print(y_t)

    # PREDICTION
    h_t_prior = F_t_minus_1_optimal @ h_t_minus_1_post 
    Sigma_t_prior = (F_t_minus_1_optimal @ Sigma_t_minus_1_post @ F_t_minus_1_optimal.T) + Q_t_minus_1
    y_t_prior = H_t_optimal @ h_t_prior  
    # print("Predicted measurement: y_t_prior")
    # print(y_t_prior)
    S_t = (H_t_optimal @ Sigma_t_prior @ H_t_optimal.T) + R_t_minus_1  
    
    # If in ground truth H scenario, S_t should be rank-deficient -- add a small value to the diagonal
    if np.linalg.matrix_rank(H_t_optimal) < H_t_optimal.shape[0]:
        # print("Forced to add small value to S_t diagonal")
        S_t += 1e-6 * np.eye(S_t.shape[0])
        rank_S_t = np.linalg.matrix_rank(S_t)
        # check work
        if rank_S_t < S_t.shape[0]:
            print(f"Warning: S_t is rank-deficient ({rank_S_t}/{S_t.shape[0]})")
    K_t = Sigma_t_prior @ (H_t_optimal.T) @ np.linalg.inv(S_t)  
    

    # CORRECTION
    innovation = y_t - y_t_prior 
    h_t_post = h_t_prior + K_t @ innovation 
    Sigma_t_post = (np.eye(Sigma_t_prior.shape[0]) - K_t @ H_t_optimal) @ Sigma_t_prior @ (np.eye(Sigma_t_prior.shape[0]) - K_t @ H_t_optimal).T + K_t @ R_t_minus_1 @ K_t.T 
    y_t_post = H_t_optimal @ h_t_post

    # POSTPROCESSING
    # if the values in h_t_post are too small, then set them to zero
    # makes more robust to noise
    h_t_post[np.abs(h_t_post) < 1e-4] = 0

    # print("h_t_post")
    # printPretty(h_t_post)
    # print("y_t_post")
    # printPretty(y_t_post)

    return h_t_post, y_t_post, K_t, Sigma_t_prior, y_t_prior, Sigma_t_post

def constrained_kalman_filter(h_t_minus_1_post, F_t_minus_1_optimal, Q_t_minus_1, Sigma_t_minus_1_post, 
                              H_t_optimal, R_t_minus_1, y_t, epsilon=0.1, THRESH_SCALE=0.1):
    """
    Perform a Constrained Kalman filter update where the state h_t changes minimially wrt h_t_minus_1.

    Parameters:
        h_t_minus_1_post (numpy.ndarray): Previous posterior estimate of the state h_{t-1}^+.
        F_t_minus_1_optimal (numpy.ndarray): Optimal evolution matrix F_{t-1}^*.
        Q_t_minus_1 (numpy.ndarray): Process noise covariance Q_{t-1}.
        Sigma_t_minus_1_post (numpy.ndarray): Previous covariance estimate Σ_{t-1}^+.
        H_t_optimal (numpy.ndarray): Optimal measurement matrix H_t^*.
        R_t_minus_1 (numpy.ndarray): Measurement noise covariance R_{t-1}.
        y_t (numpy.ndarray): Observation at time t.
        epsilon (float): Constraint parameter to limit changes in h_t.

    Returns:
        h_t_post (numpy.ndarray): Constrained a posteriori estimate of the hidden state h_t^+.
        y_t_post (numpy.ndarray): Predicted observation y_t^+.
        K_t (numpy.ndarray): Kalman gain matrix.
        Sigma_t_prior (numpy.ndarray): A priori covariance estimate.
        y_t_prior (numpy.ndarray): Predicted measurement before correction.
        Sigma_t_post (numpy.ndarray): A posteriori covariance estimate.
    """

    # epsilon should be a function of how much noise is on the graph signal
    # IDEA: from R obtain a notion of the origional noise on the signal
    # TODO do it as a function of the estimated noise on the signal

    # Prediction step
    h_t_prior = F_t_minus_1_optimal @ h_t_minus_1_post  
    # print("shape h_t_prior: ", h_t_prior.shape)

    Sigma_t_prior = (F_t_minus_1_optimal @ Sigma_t_minus_1_post @ F_t_minus_1_optimal.T) + Q_t_minus_1

    y_t_prior = H_t_optimal @ h_t_prior  
    # print("shape y_t_prior: ", y_t_prior.shape)

    y_t = y_t.reshape(R_t_minus_1.shape[0], 1)
    # print("shape y_t: ", y_t.shape)

    S_t = (H_t_optimal @ Sigma_t_prior @ H_t_optimal.T) + R_t_minus_1  

    # print("shape    S_t: ", S_t.shape)

    # Regularization for rank deficiency
    if np.linalg.matrix_rank(H_t_optimal) < H_t_optimal.shape[0]:
        S_t += 1e-6 * np.eye(S_t.shape[0])

    K_t = Sigma_t_prior @ H_t_optimal.T @ np.linalg.inv(S_t)  

    # print("shape    K_t: ", K_t.shape)

    innovation = y_t - y_t_prior  
    # print("shape innovation: ", innovation.shape)

    h_t_unconstrained = h_t_prior + K_t @ innovation  

    # print("shape h_t_unconstrained: ", h_t_unconstrained.shape)

    length = h_t_prior.shape[0]
    h_t = cp.Variable(length)  

    h_t = np.reshape(h_t, (length, 1)) # although says length, think N^2

    # print("shape h_t: ", h_t.shape)

    # Objective: Minimize the deviation from the standard Kalman update
    obj = cp.Minimize(cp.sum_squares(h_t - h_t_unconstrained))
    constraints = [cp.norm1(h_t - h_t_minus_1_post) <= epsilon]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    h_t_post = h_t.value.reshape(-1, 1)

    # postprocessing
    threshold = THRESH_SCALE * np.max(np.abs(h_t_post))
    h_t_post[np.abs(h_t_post) < threshold] = 0

    I = np.eye(Sigma_t_prior.shape[0])
    Sigma_t_post = (I - K_t @ H_t_optimal) @ Sigma_t_prior @ (I - K_t @ H_t_optimal).T + K_t @ R_t_minus_1 @ K_t.T 

    y_t_post = H_t_optimal @ h_t_post

    return h_t_post, y_t_post, K_t, Sigma_t_prior, y_t_prior, Sigma_t_post

def estimate_noise_covariances(Q_t_minus_1, alpha, K_t, Sigma_t_prior, 
                                y_t, y_t_prior, R_t_minus_1, y_t_post, H_t_optimal):
    """
    Estimate the hidden state noise covariance (Q_t) and observation noise covariance (R_t).

    Parameters:
        Q_t_minus_1 (numpy.ndarray): Q_{t-1} (hidden state noise covariance matrix at t-1)
        alpha (float): Weighting factor for the current and previous noise covariance.
        K_t (numpy.ndarray): K_t (Kalman gain at time t)
        Sigma_t_prior (numpy.ndarray): Sigma_{t-1}^+ (a posteriori estimate of the covariance at t-1)
        y_t (numpy.ndarray): y_t (observation at time t)
        y_t_prior (numpy.ndarray): y_t^+ (predicted observation at time t)
        R_t_minus_1 (numpy.ndarray): R_{t-1} (measurement noise covariance matrix at t-1)
        y_t_post (numpy.ndarray): y_t^+ (predicted observation at time t)
        H_t_optimal (numpy.ndarray): H_t^* (optimal measurement matrix at t)

    Returns:
        Q_t (numpy.ndarray): Q_t (updated hidden state noise covariance matrix)
        R_t (numpy.ndarray): R_t (updated observation noise covariance matrix)
    """
    innovation = y_t - y_t_prior
    Q_t = (alpha * Q_t_minus_1) + ( (1 - alpha) * (K_t @innovation @innovation.T @ K_t.T)) # kalman gain matrix uses to pull the innovation into the hidden state space, covariance of state estimation error
    
    R_t = np.zeros(R_t_minus_1.shape)
    R_t = (alpha * R_t_minus_1) + ( (1 - alpha) * ( ((y_t - y_t_post)@(y_t - y_t_post).T) + (H_t_optimal @ Sigma_t_prior @ H_t_optimal.T) ))
    # print("(y_t - y_t_post) @(y_t - y_t_post).T")
    # printPretty( (y_t - y_t_post) @(y_t - y_t_post).T)
    
    y_t = y_t.reshape(R_t.shape[0], 1)

    # print("y_t")
    # print(y_t)
    # print("y_t_post")
    # print(y_t_post)
    
    # print("y_t - y_t_post")
    # print(y_t - y_t_post)
    

    # print("H_t_optimal @ Sigma_t_prior @ H_t_optimal.T")
    # printPretty(H_t_optimal @ Sigma_t_prior @ H_t_optimal.T)

    

    # Values that are less than 10% of the largest value in the current R matrix are set to 0
    threshold = 0.1 * np.max(np.abs(R_t))
    R_t[np.abs(R_t) < threshold] = 0

    return Q_t, R_t

def freeze_part_1_test_part_2(Y, Y_nonoise, ground_truth_adjacency_tensor, lambda_param, alpha, Q_initial, R_initial, F_initial, H_initial, max_iter=100):
    '''
    Considering under the static scenario now. 
    Estimate the graph connectivity. Fix F to be I, Fix H_t to be y_t * h_t_pseudoinverse. 
    The graph signal is now with noise. Thus, each H_t differs. Additionally Q and R are nonzero.
    Focus on Part 2 with kalman filter


    Parameters:
        Y (numpy.ndarray): N x M matrix with the time-varying graph signal under a given data model.
        ground_truth_adjacency (numpy.ndarray): N x N matrix with the true graph connectivity.
        lambda_param (float): Regularization parameter.
        alpha (float): Weighting factor for the current and previous noise covariance.
        Q_initial (numpy.ndarray): Initial hidden state noise covariance matrix (N^2 x N^2).
        R_initial (numpy.ndarray): Initial observation noise covariance matrix (N x N).
        F_initial (numpy.ndarray): Initial evolution matrix (N^2 x N^2).
        H_initial (numpy.ndarray): Initial measurement matrix (N x N^2).
        max_iter (int): Maximum number of iterations.

    Returns:
        A (numpy.ndarray): N x N x M matrix with the estimated graph connectivity at each time step.
    '''
    # TODO adapt code to work with A_tensor rather than static A. A remains static but has noise which varies per time step.

    N, M = Y.shape 
    A_tensor = np.zeros((N, N, M)) 

    # Initialize
    Sigma_t_minus_1_post = np.eye(N*N) # (N^2 x N^2) -- fine for initialization

    # This still remains fixed for all time steps under the static scenario -- ground truth so this remains as Identity
    F_fixed_ground_truth = np.eye(N*N, N*N)

    # print("M is ...  ", M)
    all_H_optimals = np.zeros((N, N*N, M))

    for t in range(3, M):  # Has to start at the third time step
        # DEFINING INPUTS TO THE FUNCTIONS
        # print(f"Processing time step {t}...")

        last_3_timesteps_adj = ground_truth_adjacency_tensor[:, :, t-3:t]
        last_3_timesteps_hiddenst = last_3_timesteps_adj.reshape(N*N, 1, 3)
        last_3_timesteps_hiddenst = last_3_timesteps_hiddenst.reshape(N*N, 3)
        Q_estimation = (1/3) * last_3_timesteps_hiddenst @ last_3_timesteps_hiddenst.T
        # print("Q_estimation:")
        # printPretty(Q_estimation)

        # Q: Measurement covariance is nonzero if under the noise scenario. If data has no noise on it, the cov will eval to 0 anyway
        # From the last 3 timesteps, calculate a measure of the R_t
        last_3_timesteps_Y = Y[:, t-3:t]
        R_estimation = (1/3) * last_3_timesteps_Y @ last_3_timesteps_Y.T
        

        # PART 1: Calculate optimal matrices F_{t-1}^* and H_t^* -- bypass with ground truth knowledge
        cur_ground_truth_adjacency_tensor = ground_truth_adjacency_tensor[:, :, t]
        cur_h_ground_truth = cur_ground_truth_adjacency_tensor.reshape((N*N), 1)
        cur_h_pseudoinverse = np.linalg.pinv(cur_h_ground_truth)
        cur_y = Y_nonoise[:, t] # important tto be without noise
        cur_y = cur_y.reshape(N, 1)
        cur_h_pseudoinverse = cur_h_pseudoinverse.reshape(1, N*N)
        cur_H_ground_truth = np.multiply(cur_y, cur_h_pseudoinverse)

        # print("cur h ground truth")
        # printPretty(cur_h_ground_truth)
        # print("cur H ground truth")
        # printPretty(cur_H_ground_truth)

        # PART 2: Perform Kalman filter prediction and correction steps
        h_t_post, y_t_post, K_t, Sigma_t_prior, y_t_prior, Sigma_t_post = proposed_kalman_filter(cur_h_ground_truth, F_fixed_ground_truth, Q_estimation, Sigma_t_minus_1_post, cur_H_ground_truth, R_estimation, cur_y)

        # print("h_t_post")
        # printPretty(h_t_post)
        # PART 3: Estimate hidden state noise covariance (Q_t) and observation noise covariance (R_t)
        # defined above

        Sigma_t_minus_1_post = Sigma_t_post
        
        # h_t_post is the vectorized version of the adjacency matrix
        # unvectorize h_t_post to get the adjacency matrix
        A_t_post = np.reshape(h_t_post, (N, N)) # from N^2x1 to N x N
        A_tensor[:, :, t] = A_t_post

        all_H_optimals[:, :, t] = cur_H_ground_truth

    return A_tensor, all_H_optimals

def estimate_graph_connectivity(Y, Y_nonoise, lambda_param, alpha, Q_initial, R_initial, F_initial, H_initial, max_iter=100, ground_truth_H=None, ground_truth_adjacency_tensor=None, constrained=False):
    """
    Estimate graph connectivity (A_t) for a time-varying graph signal.

    Parameters:
        Y (numpy.ndarray): N x M matrix with the time-varying graph signal under a given data model.
        lambda_param (float): Regularization parameter.
        alpha (float): Weighting factor for the current and previous noise covariance.
        Q_initial (numpy.ndarray): Initial hidden state noise covariance matrix (N^2 x N^2).
        R_initial (numpy.ndarray): Initial observation noise covariance matrix (N x N).
        F_initial (numpy.ndarray): Initial evolution matrix (N^2 x N^2).
        H_initial (numpy.ndarray): Initial measurement matrix (N x N^2).
        max_iter (int): Maximum number of iterations.

    Returns:
        A (numpy.ndarray): N x N x M matrix with the estimated graph connectivity at each time step.
    """

    N, M = Y.shape 
    A = np.zeros((N, N, M)) 

    Q_t_minus_1 = Q_initial # (N^2 x N^2)
    R_t_minus_1 = R_initial # (N x N)
    F_t_minus_1_optimal = F_initial # (N^2 x N^2)
    H_t_optimal = H_initial # (N x N^2)
    # print("Shape of H_initial: ", H_initial.shape)h_t_minus_1:
    Sigma_t_minus_1_post = np.eye(N*N) # (N^2 x N^2)

    # PRIORS on h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4 -- mostly zero and a few 1's
    # USE WHEN RUNNING WITH H PRIOR AS INITIAL GUESS FOR H
    h_t_minus_1 = (np.zeros(N*N)).reshape((N*N, 1))
    h_t_minus_2 = (np.zeros(N*N)).reshape((N*N, 1))
    h_t_minus_3 = (np.zeros(N*N)).reshape((N*N, 1))
    h_t_minus_4 = (np.zeros(N*N)).reshape((N*N, 1))
    h_t_post = np.zeros(N*N) # (N^2 x 1)


    # IF PASSING IN ORIG GUESSES FOR H
    # H_initial_guess = H_initial

    # INITIAL H WOULD BE BASED ON KNOWING THE GROUND TRUTH ADJACENCY MATRIX AT THE FIRST TIME STEP
    
    init_adj = ground_truth_adjacency_tensor[:, :, 0]
    vec_init_adj = init_adj.reshape((N*N, 1)) # initial for h 
    # pseudo inverse of vec_init_adj
    vec_init_adj_pseudoinverse = np.linalg.pinv(vec_init_adj)
    H_initial_guess = np.multiply(Y[:, 0].reshape(N, 1), vec_init_adj_pseudoinverse)
    # print("H_initial_guess in step 1")
    # printPretty(H_initial_guess)
    


    ''' 
    # ALL ZEROS
    H_initial_guess = np.zeros((N, N*N))

    # 20 percent are 1.0 valued, the rest are 0
    H_initial_guess = np.ones((N, N*N))  
    num_ones = int(0.2 * N * N * N)  
    indices = np.random.choice(N*N*N, num_ones, replace=False)  

    H_initial_guess = H_initial_guess.flatten()  
    H_initial_guess[indices] = 1.0  
    H_initial_guess = H_initial_guess.reshape(N, N*N)  

    print("H_initial_guess in step 1")
    printPretty(H_initial_guess)
    '''

    # AN OPTION WITHOUT USING PSEUDOINV
    # H_initial_guess = np.ones((N, N*N))
    # h_t_minus_1 = (np.ones(N*N)).reshape((N*N, 1))
    # h_t_minus_2 = (np.ones(N*N)).reshape((N*N, 1))
    # h_t_minus_3 = (np.ones(N*N)).reshape((N*N, 1))
    # h_t_minus_4 = (np.ones(N*N)).reshape((N*N, 1))

    for t in range(3, M):
        print(f"Processing time step {t}...")

        # Pull the time lagged graph signals from the data matrix Y
        y_t = Y[:, t]
        y_t_minus_1 = Y[:, t - 1]
        y_t_minus_2 = Y[:, t - 2]
        y_t_minus_3 = Y[:, t - 3]

        if ground_truth_H is not None:
            cur_groundtr_H = ground_truth_H[:, :, t]
        
        printPretty(H_initial_guess)
        

        # print("   Processing part 1")
        if ground_truth_H is not None: # IF WORKING WITH GROUND TRUTH FOR TESTING
            F_t_minus_1_optimal, H_t_optimal, \
                h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4 \
                    = calculate_optimal_F_H(y_t_minus_1, y_t_minus_2, y_t_minus_3, \
                        h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4, \
                            lambda_param, F_t_minus_1_optimal, H_t_optimal, ground_truth_H=cur_groundtr_H, H_t_minus_1=H_initial_guess)

        else:   # ELSE WORKING WITH H AS 0
            F_t_minus_1_optimal, H_t_optimal, \
                h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4 \
                    = calculate_optimal_F_H(y_t_minus_1, y_t_minus_2, y_t_minus_3, \
                        h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4, \
                            lambda_param, F_t_minus_1_optimal, H_t_optimal)
        
        # MAYBE ISSUE IS IN Q AND R ESTIMATION CAUSING ERRORS IN THE MATRICES
        def use_ground_truth_Q_and_R(ground_truth_adjacency_tensor, t, Y, Y_nonoise):
            last_3_timesteps_adj = ground_truth_adjacency_tensor[:, :, t-3:t]
            last_3_timesteps_hiddenst = last_3_timesteps_adj.reshape(N*N, 1, 3)
            last_3_timesteps_hiddenst = last_3_timesteps_hiddenst.reshape(N*N, 3)
            Q_estimation = (1/3) * last_3_timesteps_hiddenst @ last_3_timesteps_hiddenst.T
            last_3_timesteps_Y_onlynoise = Y[:, t-3:t] - Y_nonoise[:, t-3:t]
            R_estimation = (1/3) * last_3_timesteps_Y_onlynoise @ last_3_timesteps_Y_onlynoise.T
            return Q_estimation, R_estimation

        # print("   Processing part 2")
        if constrained:
                    h_t_post, y_t_post, K_t, Sigma_t_prior, y_t_prior, Sigma_t_post = constrained_kalman_filter(h_t_minus_1, F_t_minus_1_optimal, Q_t_minus_1, Sigma_t_minus_1_post, H_t_optimal, R_t_minus_1, y_t)
        else:
            h_t_post, y_t_post, K_t, Sigma_t_prior, y_t_prior, Sigma_t_post = proposed_kalman_filter(h_t_minus_1, F_t_minus_1_optimal, Q_t_minus_1, Sigma_t_minus_1_post, H_t_optimal, R_t_minus_1, y_t)


        # print("   Processing part 3")
        Q_t, R_t = estimate_noise_covariances(Q_t_minus_1, alpha, K_t, Sigma_t_prior, y_t, y_t_prior, R_t_minus_1, y_t_post, H_t_optimal)
        # print("Alpha value: ", alpha)
        Q_t_minus_1 = Q_t
        R_t_minus_1 = R_t

        # Get ground truth Q and R for debugging purposes
        Q_est, R_est = use_ground_truth_Q_and_R(ground_truth_adjacency_tensor, t, Y, Y_nonoise)
        # rint("Q_t as solved in part 3")
        # printPretty(Q_t)
        # print("R_t as solved in part 3")
        # printPretty(R_t)
        # print("Q_estimation using the ground truth signals (with and without noise) and adjacencies")
        # printPretty(Q_est)
        # print("R_estimation using the ground truth signals (with and without noise) and adjacencies")
        # printPretty(R_est)


        # Shift the time step -- throw away the oldest time step and use the new h from the kalman filter
        h_t_minus_4 = h_t_minus_3
        h_t_minus_3 = h_t_minus_2
        h_t_minus_2 = h_t_minus_1
        h_t_minus_1 = h_t_post

        H_initial_guess = H_t_optimal # set this for the next time step

        Sigma_t_minus_1_post = Sigma_t_post
        
        # h_t_post is the vectorized version of the adjacency matrix
        # unvectorize h_t_post to get the adjacency matrix
        A_t_post = np.reshape(h_t_post, (N, N)) # from N^2x1 to N x N
        A[:, :, t] = A_t_post

    return A

def printPretty(a):
    for row in a:
        for col in row:
            if col == 0.00:
                print("    0", end="    ")
            else:
                print("{:8.2f}".format(col), end=" ")
        print("")


############ MAIN #############
'''
if __name__ == "__main__":
    NODES = 5
    SAMPLES = 15
    K_PARAM = 1
    EPSILON = 1e-10
    VALUE_THRESHOLD = 1e10
    ALPHA = 0.9
    NOISE_MEAN, NOISE_COVAR = 5, 1
    INITIAL_WEIGHT_MEAN, INITIAL_WEIGHT_VAR = 5, 1
    INITIAL_SIGNAL_MEAN, INITIAL_SIGNAL_VAR = 10, 3
    DECREASE = 0.97
    INCREASE = 1.03

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
        Generated_Y_dynamic_piecewiseFast_SVARM = generate_data_matrices(N=NODES, M=SAMPLES, K=K_PARAM, EP=EPSILON, \
                                                                     VT=VALUE_THRESHOLD, AL=ALPHA, NM=NOISE_MEAN, NC=NOISE_COVAR,\
                                                                         IWM=INITIAL_WEIGHT_MEAN, IWV=INITIAL_WEIGHT_VAR, \
                                                                            ISM=INITIAL_SIGNAL_MEAN, ISV=INITIAL_SIGNAL_VAR, DF=DECREASE, IF=INCREASE)

    Q_initial = np.eye(NODES*NODES) # correct dimensions
    R_initial = np.eye(NODES)
    H_initial = np.random.randn(NODES, NODES*NODES) 
    F_initial = np.random.randn(NODES*NODES, NODES*NODES)

    lambda_param = 0.1
    alpha = 0.95

    # Estimate graph connectivity
    A_estimated_static_SEM = estimate_graph_connectivity(Generated_Y_static_SEM, lambda_param, alpha, Q_initial, R_initial, F_initial, H_initial)
    # A_estimated_static_SVARM = estimate_graph_connectivity(Generated_Y_static_SVARM, lambda_param, alpha, Q_initial, R_initial, F_initial, H_initial)

    # Compare it with true adjacency from SEM
    print("Solved Adjacency Matrix from SEM")
    for i in range(SAMPLES):
        print(f"Time Step {i}")
        printPretty(A_estimated_static_SEM[:, :, i])

    # BASELINE: COMPARE TO GLASSO
    # Get the empirical covariace matrix for Y_static_SEM
    emp_cov = np.cov(Generated_Y_static_SEM)

    # invert it to get the empirical precision matrix
    # emp_prec = np.linalg.inv(emp_cov)

    # Checking the shape
    print("Shape of empirical covariance matrix: ", emp_cov.shape)
    print("Shape of sample array: ", Generated_Y_static_SEM.shape)

    P = glasso_problem(emp_cov, NODES, reg_params = {'lambda1': 0.05}, latent = False, do_scaling = False)

    lambda1_range = np.logspace(0, -3, 30)
    modelselect_params = {'lambda1_range': lambda1_range}

    P.model_selection(modelselect_params = modelselect_params, method = 'eBIC', gamma = 0.1)

    # regularization parameters are set to the best ones found during model selection
    print(P.reg_params)

    sol = P.solution.precision_
    P.solution.calc_adjacency(t = 1e-4)

    printPretty(P.solution.adjacency_)
'''
