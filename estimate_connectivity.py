# Proposed Method File
# Takes as input a matrix Y (N x M) with the time varying graph signal under a given data model
# Returns the estimated graph connectivity (A_t) of dimension NxN per time step
# TODO: Plot the adjacency matrix answers as a heatmap

import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from MyDataGen import generate_data_matrices 
import time
from MyDataGen_GGLASSOSolver import time_varying_graphical_lasso
from gglasso.problem import glasso_problem
import seaborn as sns
from scipy.stats import rankdata 
import sys
import math

# FOR A SINGLE TIME STEP
'''
GET OPTIMAL EVOLUTION AND MEASUREMENT MATRICES
'''
def calculate_optimal_F_H(y_t_minus_1, y_t_minus_2, y_t_minus_3, 
                          h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4,
                          lambda_param, F_t_minus_1_optimal, H_t_optimal, ground_truth_H=None, THRESH_SCALE=0.1, H_t_minus_1=None):
    """
    Calculate optimal matrices F_{t-1}^* and H_t^* for a given set of inputs.

    Parameters:
        y_t_minus_1, y_t_minus_2, y_t_minus_3: Graph signals at previous timesteps.
        h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4: Historical state vectors.
        lambda_param (float): Regularization parameter.

    Returns:
        F_t_minus_1_optimal (numpy.ndarray): Optimized matrix F_{t-1}^*.
        H_t_optimal (numpy.ndarray): Optimized matrix H_t^*.
    """
    print("Calculating optimal F and H...")
    n = len(y_t_minus_1)
    Nodes = int(math.sqrt(h_t_minus_1.shape[0]))

    epsilon = 0.1

    y_t_minus_1 = y_t_minus_1.reshape(n, 1)
    y_t_minus_2 = y_t_minus_2.reshape(n, 1)
    y_t_minus_3 = y_t_minus_3.reshape(n, 1)

    def objective_H(H_flat):
        H = H_flat.reshape(n, n*n)
        # part1 = (1/8)*np.linalg.norm(y_t_minus_3 - H @ h_t_minus_3, 2)
        part1 = np.linalg.norm(y_t_minus_3 - H @ h_t_minus_3, 2)
        part2 = np.linalg.norm(y_t_minus_2 - H @ h_t_minus_2, 2)
        part3 = np.linalg.norm(y_t_minus_1 - H @ h_t_minus_1, 2)
        # reg = lambda_param * np.linalg.norm(H.flatten(), 1) # l1 norm

        reg = 0
        # group sparsity constraint by column on H
        for col in range(H.shape[1]):
            norm_col = np.linalg.norm(H[:, col], 2)
            if norm_col > 0:
                reg += norm_col
        reg = lambda_param * reg


        # return (8/7)*(part1 + part2 + part3) + reg # does not work
        # return (8/7)*(part1 + part2 + part3)  # does not work, adds extra entries
        # return part1+reg -- # whole columns getting lost
        # exp = (1/2)*part1 + (1/2)*part2
        # exp = part1 + part2 + part3 
        # exp = (part1 + part2 + part3)
        exp = (part1 + part2 + part3)
        # exp = part1

        # Temporal consistency regularization: ||H_t - H_t_minus_1||_1
        if H_t_minus_1 is not None:
            reg_temporal = np.linalg.norm(H - H_t_minus_1, 2)
        else:
            reg_temporal = 0  # No penalty for the first iteration
        
        return exp + 10*reg + lambda_param*30*reg_temporal

    def objective_F(F_flat):
        F = F_flat.reshape(n*n, n*n)
        # part1 = (1/8)*np.linalg.norm(h_t_minus_3 - F @ h_t_minus_4, 2)
        # part2 = (1/4)*np.linalg.norm(h_t_minus_2 - F @ h_t_minus_3, 2)
        # part3 = (1/2)*np.linalg.norm(h_t_minus_1 - F @ h_t_minus_2, 2)
        part1 = np.linalg.norm(h_t_minus_3 - F @ h_t_minus_4, 2)
        part2 = np.linalg.norm(h_t_minus_2 - F @ h_t_minus_3, 2)
        part3 = np.linalg.norm(h_t_minus_1 - F @ h_t_minus_2, 2)
        # h1_1 - h2_1 <= ep, h1_2 = h2_2 <= ep, h1_3 = h2_3 <= ep, h1_4 = h2_4 <= ep ... h1_N^2 = h2_N^2 <= ep
    
        # F_vectorized = F.flatten()
        # reg = lambda_param * np.linalg.norm(F_vectorized, 1)
        reg = lambda_param * np.linalg.norm(F, 'fro')
        # reg = lambda_param * np.linalg.norm(F, 1) # forcing a good fit on the data so i want to add regularization to prevent strong overfitting
        reg_sparsity = lambda_param * np.linalg.norm(F, 1)
        return (1/3)*(part1 + part2 + part3) + reg + reg_sparsity
        # return (8/7)*(part1 + part2 + part3) + reg
    
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

    
    # scipy minimize function needs to take in a flattened array
    # H_initial = np.zeros(n, n*n).flatten() if H_t_optimal is None else H_t_optimal.flatten()
    # H_initial = np.random.randn(n, n*n).flatten() if H_t_optimal is None else H_t_optimal.flatten()
    # F_initial = np.random.randn(n*n, n*n).flatten() if F_t_minus_1_optimal is None else F_t_minus_1_optimal.flatten()

    # print("H_initial is given here as ground truth")
    # print("Ground truth for H_optimal")
    # printPretty(ground_truth_H)
    
    # THRESHOLDING THE H_INITIAL PREPROCESSING
    # select the maximum value in H
    max_H_matrix = np.amax(np.abs(ground_truth_H))
    threshold = THRESH_SCALE * max_H_matrix
    ground_truth_H[np.abs(ground_truth_H) < threshold] = 0
    # print("H initial as based on thresholded ground truth")
    # printPretty(ground_truth_H)

    # H_initial = ground_truth_H.flatten() if ground_truth_H is not None else np.zeros(n, n*n).flatten()
    H_initial = H_t_minus_1.flatten() if H_t_minus_1 is not None else np.zeros(n, n*n).flatten() # first instance via moore-penrose pseudoinverse, later, last value

    # rather than setting H_initial to the ground truth, set H initial to the last solved H
    # If it is the first iteration, then set H initial to the solution to the moore-penrose pseudoinverse



    
    
    print("F_initial is ground truth (identity because static adjacency).")
    F_initial = np.eye(n*n, n*n).flatten() if F_t_minus_1_optimal is None else F_t_minus_1_optimal.flatten()
    constraints_F = {'type': 'ineq', 'fun': constraint_funct}

    print("Performing optimization...")
    
    H_optimization = minimize(objective_H, H_initial, method='L-BFGS-B', options={'ftol': 1e-20, 'disp': False}) 
    H_optimal = H_optimization.x.reshape(n, n*n)
    # THRESHOLDING the H MATRIX POSTPROCESSING
    print("Thresholding the H matrix...")
    # select the maximum value in H
    max_H_matrix = np.amax(np.abs(H_optimal))
    threshold = THRESH_SCALE * max_H_matrix
    H_optimal[np.abs(H_optimal) < threshold] = 0
    # print("Solved answer for H_optimal")
    # printPretty(H_optimal)

    # DEBUGGING H OPTIMIZATION

    print("IN STATIC: Thus, H_optimal should be equal to H_solved")
    equal = np.allclose(H_optimal, ground_truth_H, atol=1e-1)
    if not equal:
        print("\033[91mH_optimal is NOT equal to ground truth H!\033[0m")  # Prints in red
        print("H optimal has been solved to be")
        printPretty(H_optimal)
    else:
        print("H_optimal is equal to ground truth H.")
        print("The algorithm solved H_optimial to be")
        printPretty(H_optimal)
    
    # print(np.linalg.norm(y_t_minus_1 - H_optimal @ h_t_minus_1, 2))
    # print(np.linalg.norm(y_t_minus_2 - H_optimal @ h_t_minus_2, 2))
    # print(np.linalg.norm(y_t_minus_3 - H_optimal @ h_t_minus_3, 2))
    

    # IF UNDER STATIC SETTING, F SHOULD BE IDENTITY - for testing # OTHERWISE SOLVE FOR H HERE
    F_optimal = np.eye(n*n, n*n)
    # F_optimization = minimize(objective_F, F_initial, method='SLSQP', constraints=constraints_F)
    # F_optimal = F_optimization.x.reshape(n*n, n*n)

    # IN THE STATIC CASE, THIS SHOULD BE THE 'ANSWER' FOR F
    # F_optimal = np.eye(n*n, n*n) # for debugging purposes -- make it what it should be theoretically

    # Is my solved F orthonormal? based on the behavior it seems like it but the frob norm is quite large
    # print("checking orthonomality of F:")
    # printPretty(F_optimal @ F_optimal.T)

    # DEBUGGING FOR H
    # multiply H by h_t_minus_1 to see if you get the origional sample back
    
    # print("H_t_optimal @ h_t_minus_1: ", H_optimal @ h_t_minus_1)
    # check for other samples too
    # print("H_t_optimal @ h_t_minus_2: ", H_optimal @ h_t_minus_2)
    # print("H_t_optimal @ h_t_minus_3: ", H_optimal @ h_t_minus_3) 

    # if H_optimal @ h_t_minus_1 not close to y_t_minus_1, then there is a problem etc.
    tolerance = 20

    y1_approx = H_optimal @ h_t_minus_1
    y2_approx = H_optimal @ h_t_minus_2
    y3_approx = H_optimal @ h_t_minus_3

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

    # Step 3: Compare the rankings directly with the samples in y 

    # Debugging
    # print("y_t_minus_1:")
    # print(y_t_minus_1)
    # print("H_optimal:")
    # printPretty(H_optimal)
    # print("F_optimal:")
    # printPretty(F_optimal)
    # print("frobenius norm F")
    # print(np.linalg.norm(F_optimal, 'fro'))

    # This ranking does not work -- why?
    # H_row_means = np.mean(np.abs(H_optimal), axis=1)
    # print("H row means:")
    # print(H_row_means)
    # H_magnitude_ranking = rankdata(H_row_means, method='ordinal') # does increasing

    # y_magnitude_ranking = rankdata(y_t_minus_1, method='ordinal') # does increasing

    # print("Ranking of rows in H:", H_magnitude_ranking)
    # print("Ranking of y sample entries:", y_magnitude_ranking)

    # Assert that the rankings match
    # print("Rankings match between H and y?", np.array_equal(H_magnitude_ranking, y_magnitude_ranking))

    # print("The ||h_t_minus_1||_1 is: ", np.linalg.norm(h_t_minus_1, 1))

    # The rank of F and H need to be full rank in order for the system to be observable
    # print("Rank of F: ", np.linalg.matrix_rank(F_optimal))
    # print("Rank of H: ", np.linalg.matrix_rank(H_optimal))

    

    return F_optimal, H_optimal, h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4

'''
GET HIDDEN STATE ESTIMATE AND OBSERVATION ESTIMATE
'''
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
    # print("h t minus 1 post")
    # printPretty(h_t_minus_1_post)

    # PREDICTION
    h_t_prior = F_t_minus_1_optimal @ h_t_minus_1_post 
    print("h t prior")
    printPretty(h_t_prior)

    # print("h t prior")
    # printPretty(h_t_prior)

    # F: N^2 x N^2, Sigma: N^2 x N^2, Q: N^2 x N^2
    Sigma_t_prior = (F_t_minus_1_optimal @ Sigma_t_minus_1_post @ F_t_minus_1_optimal.T) + Q_t_minus_1

    # print("Sigma t prior")
    # printPretty(Sigma_t_prior)

    y_t_prior = H_t_optimal @ h_t_prior  

    # print("y t prior")
    # printPretty(y_t_prior)

    # K CALCULATION
    S_t = (H_t_optimal @ Sigma_t_prior @ H_t_optimal.T) + R_t_minus_1  

    # print("S t")
    # printPretty(S_t)
    
    # If in ground truth H scenario, S_t should be rank-deficient -- add a small value to the diagonal
    if np.linalg.matrix_rank(H_t_optimal) < H_t_optimal.shape[0]:
        # print("In ground truth H scenario so adding small value to S_t diagonal")
        S_t += 1e-6 * np.eye(S_t.shape[0])
        rank_S_t = np.linalg.matrix_rank(S_t)
        # check work
        if rank_S_t < S_t.shape[0]:
            print(f"Warning: S_t is rank-deficient ({rank_S_t}/{S_t.shape[0]})")

    K_t = Sigma_t_prior @ (H_t_optimal.T) @ np.linalg.inv(S_t)  

    print("K t")
    printPretty(K_t)

    y_t = y_t.reshape(R_t_minus_1.shape[0], 1)

    # print("y t")
    # printPretty(y_t)

    # CORRECTION
    innovation = y_t - y_t_prior 

    # print("innovation")
    # printPretty(innovation)

    h_t_post = h_t_prior + K_t @ innovation 

    # print("h t post")
    # printPretty(h_t_post)

    Sigma_t_post = (np.eye(Sigma_t_prior.shape[0]) - K_t @ H_t_optimal) @ Sigma_t_prior @ (np.eye(Sigma_t_prior.shape[0]) - K_t @ H_t_optimal).T + K_t @ R_t_minus_1 @ K_t.T 

    # print("Sigma t post")
    # printPretty(Sigma_t_post)

    y_t_post = H_t_optimal @ h_t_post

    # if the values in h_t_post are too small, then set them to zero
    h_t_post[np.abs(h_t_post) < 1e-4] = 0

    # print("y t post")
    # printPretty(y_t_post)

    return h_t_post, y_t_post, K_t, Sigma_t_prior, y_t_prior, Sigma_t_post


'''
CONSTRAINED KF '''

import numpy as np
import cvxpy as cp

def constrained_kalman_filter(h_t_minus_1_post, F_t_minus_1_optimal, Q_t_minus_1, Sigma_t_minus_1_post, 
                              H_t_optimal, R_t_minus_1, y_t, epsilon=0.01, THRESH_SCALE=0.1):
    """
    Perform a Constrained Kalman filter update where the state h_t does not change too much from h_t_minus_1.

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
    print("shape h_t_prior: ", h_t_prior.shape)

    Sigma_t_prior = (F_t_minus_1_optimal @ Sigma_t_minus_1_post @ F_t_minus_1_optimal.T) + Q_t_minus_1

    y_t_prior = H_t_optimal @ h_t_prior  
    print("shape y_t_prior: ", y_t_prior.shape)

    
    y_t = y_t.reshape(R_t_minus_1.shape[0], 1)
    print("shape y_t: ", y_t.shape)

    # Compute Kalman gain
    S_t = (H_t_optimal @ Sigma_t_prior @ H_t_optimal.T) + R_t_minus_1  

    print("shape    S_t: ", S_t.shape)

    # Regularization for rank deficiency
    if np.linalg.matrix_rank(H_t_optimal) < H_t_optimal.shape[0]:
        S_t += 1e-6 * np.eye(S_t.shape[0])

    K_t = Sigma_t_prior @ H_t_optimal.T @ np.linalg.inv(S_t)  

    print("shape    K_t: ", K_t.shape)

    # Compute innovation (difference between measurement and predicted observation)
    innovation = y_t - y_t_prior  
    print("shape innovation: ", innovation.shape)

    # Standard unconstrained Kalman update (for reference)
    h_t_unconstrained = h_t_prior + K_t @ innovation  

    print("shape h_t_unconstrained: ", h_t_unconstrained.shape)

    # **Constrained Update Using Quadratic Programming**
    N = h_t_prior.shape[0]
    h_t = cp.Variable(N)  # Define the constrained variable

    h_t = np.reshape(h_t, (N, 1)) # although says N, think N^2

    print("shape h_t: ", h_t.shape)

    # Objective: Minimize the deviation from the standard Kalman update
    obj = cp.Minimize(cp.sum_squares(h_t - h_t_unconstrained))

    # Constraint: Ensure ||h_t - h_t_minus_1_post||_1 <= epsilon
    constraints = [cp.norm1(h_t - h_t_minus_1_post) <= epsilon]

    # Solve QP problem
    prob = cp.Problem(obj, constraints)
    prob.solve()

    # Extract constrained solution
    h_t_post = h_t.value.reshape(-1, 1)

    # Postprocess setting values smaller than a THRESH_SCALE to 0
    threshold = THRESH_SCALE * np.max(np.abs(h_t_post))
    h_t_post[np.abs(h_t_post) < threshold] = 0


    # Compute posterior covariance
    I = np.eye(Sigma_t_prior.shape[0])
    Sigma_t_post = (I - K_t @ H_t_optimal) @ Sigma_t_prior @ (I - K_t @ H_t_optimal).T + K_t @ R_t_minus_1 @ K_t.T 

    y_t_post = H_t_optimal @ h_t_post

    return h_t_post, y_t_post, K_t, Sigma_t_prior, y_t_prior, Sigma_t_post

'''
ESTIMATE NOISE COVARIANCES
'''
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

    Q_t = (alpha * Q_t_minus_1) + ( (1 - alpha) * (K_t @innovation @innovation.T @ K_t.T))
    print("Q_t")
    printPretty(Q_t)

    R_t = (alpha * R_t_minus_1) + ( (1 - alpha) * ( (y_t - y_t_post) @(y_t - y_t_post).T + \
          (H_t_optimal @ Sigma_t_prior @ H_t_optimal.T) ))
    print("R_t")
    printPretty(R_t)

    return Q_t, R_t

# TESTING INDIVIDUAL COMPONENTS
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
        print(f"Processing time step {t}...")

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

        print("cur h ground truth")
        printPretty(cur_h_ground_truth)
        print("cur H ground truth")
        printPretty(cur_H_ground_truth)

        # PART 2: Perform Kalman filter prediction and correction steps
        h_t_post, y_t_post, K_t, Sigma_t_prior, y_t_prior, Sigma_t_post = proposed_kalman_filter(cur_h_ground_truth, F_fixed_ground_truth, Q_estimation, Sigma_t_minus_1_post, cur_H_ground_truth, R_estimation, cur_y)

        print("h_t_post")
        printPretty(h_t_post)
        # PART 3: Estimate hidden state noise covariance (Q_t) and observation noise covariance (R_t)
        # defined above

        Sigma_t_minus_1_post = Sigma_t_post
        
        # h_t_post is the vectorized version of the adjacency matrix
        # unvectorize h_t_post to get the adjacency matrix
        A_t_post = np.reshape(h_t_post, (N, N)) # from N^2x1 to N x N
        A_tensor[:, :, t] = A_t_post

        all_H_optimals[:, :, t] = cur_H_ground_truth

    return A_tensor, all_H_optimals


# CONTROL CODE PROCESSING WHOLE DATA MATRIX
def estimate_graph_connectivity(Y, lambda_param, alpha, Q_initial, R_initial, F_initial, H_initial, max_iter=100, ground_truth_H=None, ground_truth_adjacency_tensor=None):
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
    print("-----------------EST GR CONNECTIVITY-----------------")
    # print("Shape of Y: ", Y.shape)
    A = np.zeros((N, N, M)) 

    Q_t_minus_1 = Q_initial # (N^2 x N^2)
    R_t_minus_1 = R_initial # (N x N)
    F_t_minus_1_optimal = F_initial # (N^2 x N^2)
    H_t_optimal = H_initial # (N x N^2)
    # print("Shape of H_initial: ", H_initial.shape)h_t_minus_1:
    Sigma_t_minus_1_post = np.eye(N*N) # (N^2 x N^2)

    # TODO: Consdier alternative intializations -- not expected to work well on the first few iterations but thereafter should be able to adjust
    # h_t_minus_4 = np.random.randn(N*N, 1)  # (N^2 x 1)
    # h_t_minus_3 = np.random.randn(N*N, 1)
    # h_t_minus_2 = np.random.randn(N*N, 1)
    # h_t_minus_1 = np.random.randn(N*N, 1)

    # PRIORS on h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4 -- mostly zero and a few 1's
    h_t_minus_1 = (np.zeros(N*N)).reshape((N*N, 1))
    h_t_minus_2 = (np.zeros(N*N)).reshape((N*N, 1))
    h_t_minus_3 = (np.zeros(N*N)).reshape((N*N, 1))
    h_t_minus_4 = (np.zeros(N*N)).reshape((N*N, 1))


    # print("type of h_t_minus_4 elements", type(h_t_minus_4[0]))

    h_t_post = np.zeros(N*N) # (N^2 x 1)

    print("M is ...  ", M)

    init_adj = ground_truth_adjacency_tensor[:, :, 0]
    vec_init_adj = init_adj.reshape((N*N, 1)) # initial for h 
    # pseudo inverse of vec_init_adj
    vec_init_adj_pseudoinverse = np.linalg.pinv(vec_init_adj)
    H_initial_guess = np.multiply(Y[:, 0].reshape(N, 1), vec_init_adj_pseudoinverse)
    print("H_initial_guess in step 1")
    printPretty(H_initial_guess)


    for t in range(3, M):  # Has to start at the third time step
        # DEFINING INPUTS TO THE FUNCTIONS
        print(f"Processing time step {t}...")

        # Pull the time lagged graph signals from the data matrix Y
        y_t = Y[:, t]
        y_t_minus_1 = Y[:, t - 1]
        y_t_minus_2 = Y[:, t - 2]
        y_t_minus_3 = Y[:, t - 3]

        if ground_truth_H is not None:
            cur_groundtr_H = ground_truth_H[:, :, t]

        print("   Processing part 1")
        # PART 1: Calculate optimal matrices F_{t-1}^* and H_t^*
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
        last_3_timesteps_adj = ground_truth_adjacency_tensor[:, :, t-3:t]
        last_3_timesteps_hiddenst = last_3_timesteps_adj.reshape(N*N, 1, 3)
        last_3_timesteps_hiddenst = last_3_timesteps_hiddenst.reshape(N*N, 3)
        Q_estimation = (1/3) * last_3_timesteps_hiddenst @ last_3_timesteps_hiddenst.T
        print("Q_estimation:")
        printPretty(Q_estimation)

        # Q: Measurement covariance is nonzero if under the noise scenario. If data has no noise on it, the cov will eval to 0 anyway
        # From the last 3 timesteps, calculate a measure of the R_t
        last_3_timesteps_Y = Y[:, t-3:t]
        R_estimation = (1/3) * last_3_timesteps_Y @ last_3_timesteps_Y.T
        print("R_estimation:")
        printPretty(R_estimation)

        print("   Processing part 2")
        # Check that F_t_minus_1_optimal is still the identity matrix by comparing it to the identity
        print("F_t_minus_1_optimal is identity matrix?", np.array_equal(F_t_minus_1_optimal, np.eye(N*N, N*N)))
        # Check that H_t_optimal is equal to the ground truth H
        if cur_groundtr_H is not None:
            print("H_t_optimal is equal to ground truth H?", np.array_equal(H_t_optimal, cur_groundtr_H))
        # PART 2: Perform Kalman filter prediction and correction steps
        # print("h_t_minus_1")
        # printPretty(h_t_minus_1)
        # time.sleep(2)
        # h_t_post, y_t_post, K_t, Sigma_t_prior, y_t_prior, Sigma_t_post = proposed_kalman_filter(h_t_minus_1, F_t_minus_1_optimal, Q_t_minus_1, Sigma_t_minus_1_post, H_t_optimal, R_t_minus_1, y_t)
        h_t_post, y_t_post, K_t, Sigma_t_prior, y_t_prior, Sigma_t_post = proposed_kalman_filter(h_t_minus_1, F_t_minus_1_optimal, Q_estimation, Sigma_t_minus_1_post, H_t_optimal, R_estimation, y_t)
        print("h_t_post")
        printPretty(h_t_post)

        # print("h t post shape", h_t_post.shape)

        print("   Processing part 3") # see above
        # PART 3: Estimate hidden state noise covariance (Q_t) and observation noise covariance (R_t)
        # Q_t_minus_1 = Q_t
        # R_t_minus_1 = R_t

        # print(" --------- ")
        # print("for time step ", t)
        # print("h_t_minus_1: ", h_t_minus_1) # need to make sure they are updating. as time goes on this should move closer to the true adjacency matrix
        # print("h_t_minus_2: ", h_t_minus_2)
        # print("h_t_minus_3: ", h_t_minus_3) 


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


def estimate_graph_connectivity_constrainedKF(Y, lambda_param, alpha, Q_initial, R_initial, F_initial, H_initial, max_iter=100, ground_truth_H=None, ground_truth_adjacency_tensor=None):
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
    print("-----------------EST GR CONNECTIVITY-----------------")
    # print("Shape of Y: ", Y.shape)
    A = np.zeros((N, N, M)) 

    Q_t_minus_1 = Q_initial # (N^2 x N^2)
    R_t_minus_1 = R_initial # (N x N)
    F_t_minus_1_optimal = F_initial # (N^2 x N^2)
    H_t_optimal = H_initial # (N x N^2)
    # print("Shape of H_initial: ", H_initial.shape)h_t_minus_1:
    Sigma_t_minus_1_post = np.eye(N*N) # (N^2 x N^2)

    # PRIORS on h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4 -- mostly zero and a few 1's
    h_t_minus_1 = (np.zeros(N*N)).reshape((N*N, 1))
    h_t_minus_2 = (np.zeros(N*N)).reshape((N*N, 1))
    h_t_minus_3 = (np.zeros(N*N)).reshape((N*N, 1))
    h_t_minus_4 = (np.zeros(N*N)).reshape((N*N, 1))


    # print("type of h_t_minus_4 elements", type(h_t_minus_4[0]))

    h_t_post = np.zeros(N*N) # (N^2 x 1)

    print("M is ...  ", M)

    init_adj = ground_truth_adjacency_tensor[:, :, 0]
    vec_init_adj = init_adj.reshape((N*N, 1)) # initial for h 
    # pseudo inverse of vec_init_adj
    vec_init_adj_pseudoinverse = np.linalg.pinv(vec_init_adj)
    H_initial_guess = np.multiply(Y[:, 0].reshape(N, 1), vec_init_adj_pseudoinverse)
    print("H_initial_guess in step 1")
    printPretty(H_initial_guess)


    for t in range(3, M):  # Has to start at the third time step
        # DEFINING INPUTS TO THE FUNCTIONS
        print(f"Processing time step {t}...")

        # Pull the time lagged graph signals from the data matrix Y
        y_t = Y[:, t]
        y_t_minus_1 = Y[:, t - 1]
        y_t_minus_2 = Y[:, t - 2]
        y_t_minus_3 = Y[:, t - 3]

        if ground_truth_H is not None:
            cur_groundtr_H = ground_truth_H[:, :, t]

        print("   Processing part 1")
        # PART 1: Calculate optimal matrices F_{t-1}^* and H_t^*
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
        last_3_timesteps_adj = ground_truth_adjacency_tensor[:, :, t-3:t]
        last_3_timesteps_hiddenst = last_3_timesteps_adj.reshape(N*N, 1, 3)
        last_3_timesteps_hiddenst = last_3_timesteps_hiddenst.reshape(N*N, 3)
        Q_estimation = (1/3) * last_3_timesteps_hiddenst @ last_3_timesteps_hiddenst.T
        print("Q_estimation:")
        printPretty(Q_estimation)

        # Q: Measurement covariance is nonzero if under the noise scenario. If data has no noise on it, the cov will eval to 0 anyway
        # From the last 3 timesteps, calculate a measure of the R_t
        last_3_timesteps_Y = Y[:, t-3:t]
        R_estimation = (1/3) * last_3_timesteps_Y @ last_3_timesteps_Y.T
        print("R_estimation:")
        printPretty(R_estimation)

        print("   Processing part 2")
        # Check that F_t_minus_1_optimal is still the identity matrix by comparing it to the identity
        print("F_t_minus_1_optimal is identity matrix?", np.array_equal(F_t_minus_1_optimal, np.eye(N*N, N*N)))
        # Check that H_t_optimal is equal to the ground truth H
        if cur_groundtr_H is not None:
            print("H_t_optimal is equal to ground truth H?", np.array_equal(H_t_optimal, cur_groundtr_H))
        # PART 2: Perform Kalman filter prediction and correction steps
        # print("h_t_minus_1")
        # printPretty(h_t_minus_1)
        # time.sleep(2)
        # h_t_post, y_t_post, K_t, Sigma_t_prior, y_t_prior, Sigma_t_post = proposed_kalman_filter(h_t_minus_1, F_t_minus_1_optimal, Q_t_minus_1, Sigma_t_minus_1_post, H_t_optimal, R_t_minus_1, y_t)

        # using constrained KF to set the prior that the h solutions should start at 1 and that they should remain there mostly

        h_t_post, y_t_post, K_t, Sigma_t_prior, y_t_prior, Sigma_t_post = constrained_kalman_filter(h_t_minus_1, F_t_minus_1_optimal, Q_estimation, Sigma_t_minus_1_post, H_t_optimal, R_estimation, y_t)
        print("h_t_post")
        printPretty(h_t_post)

        # Postprocessing of h for the first time step
        # if value in h is nonzero, set it to 1.0
        if t == 3:
            h_t_post[h_t_post != 0.0] = 1.0


        # print("h t post shape", h_t_post.shape)

        print("   Processing part 3")
        # PART 3: Estimate hidden state noise covariance (Q_t) and observation noise covariance (R_t)
        Q_t, R_t = estimate_noise_covariances(Q_t_minus_1, alpha, K_t, Sigma_t_prior, y_t, y_t_prior, R_t_minus_1, y_t_post, H_t_optimal)

        # print(" --------- ")
        # print("for time step ", t)
        # print("h_t_minus_1: ", h_t_minus_1) # need to make sure they are updating. as time goes on this should move closer to the true adjacency matrix
        # print("h_t_minus_2: ", h_t_minus_2)
        # print("h_t_minus_3: ", h_t_minus_3) 


        # Shift the time step -- throw away the oldest time step and use the new h from the kalman filter
        h_t_minus_4 = h_t_minus_3
        h_t_minus_3 = h_t_minus_2
        h_t_minus_2 = h_t_minus_1
        h_t_minus_1 = h_t_post

        H_initial_guess = H_t_optimal # set this for the next time step


        Q_t_minus_1 = Q_t
        R_t_minus_1 = R_t
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

    



