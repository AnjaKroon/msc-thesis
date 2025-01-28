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

# FOR A SINGLE TIME STEP
'''
GET OPTIMAL EVOLUTION AND MEASUREMENT MATRICES
'''
def calculate_optimal_F_H(y_t_minus_1, y_t_minus_2, y_t_minus_3, 
                          h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4,
                          lambda_param, F_t_minus_1_optimal, H_t_optimal):
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

    epsilon = 0.1

    y_t_minus_1 = y_t_minus_1.reshape(n, 1)
    y_t_minus_2 = y_t_minus_2.reshape(n, 1)
    y_t_minus_3 = y_t_minus_3.reshape(n, 1)


    
    def objective_H(H_flat):
        H = H_flat.reshape(n, n*n)
        part1 = (1/8)*np.linalg.norm(y_t_minus_3 - H @ h_t_minus_3, 2)
        part2 = (1/4)*np.linalg.norm(y_t_minus_2 - H @ h_t_minus_2, 2)
        part3 = (1/2)*np.linalg.norm(y_t_minus_1 - H @ h_t_minus_1, 2)
        reg = lambda_param * np.linalg.norm(H, 1)
        return (8/7)*(part1 + part2 + part3) + reg

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

        return (1/3)*(part1 + part2 + part3) + reg
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
    

    # scipy minimize function needs to take in a flattened array
    H_initial = np.random.randn(n, n*n).flatten() if H_t_optimal is None else H_t_optimal.flatten()
    F_initial = np.random.randn(n*n, n*n).flatten() if F_t_minus_1_optimal is None else F_t_minus_1_optimal.flatten()


    constraints = {'type': 'ineq', 'fun': constraint_funct}

    H_optimization = minimize(objective_H, H_initial, method='L-BFGS-B', options={'ftol': 1e-20, 'disp': False}) 
    # options={'ftol': 1e-9, 'disp': True}
    F_optimization = minimize(objective_F, F_initial, method='SLSQP', constraints=constraints)

    H_optimal = H_optimization.x.reshape(n, n*n)
    F_optimal = F_optimization.x.reshape(n*n, n*n)


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
    tolerance = 10

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

    print("h3_approx vs h_t_minus_3: ", np.linalg.norm(h3_approx - h_t_minus_3))
    print("h1_approx vs h_t_minus_1: ", np.linalg.norm(h1_approx - h_t_minus_1))

    # PRIOR: There shouldn't be too much difference between h2 and h3, h1 and h2 , check
    print("h_t_minus_3:")
    printPretty(h_t_minus_3)
    print("h_t_minus_2:")
    printPretty(h_t_minus_2)
    print("h_t_minus_1:")
    printPretty(h_t_minus_1)
    print("h_t_minus_3 vs h_t_minus_2: ", np.linalg.norm(h_t_minus_3 - h_t_minus_2))
    print("h_t_minus_2 vs h_t_minus_1: ", np.linalg.norm(h_t_minus_2 - h_t_minus_1))

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
    print("H_optimal:")
    printPretty(H_optimal)
    # print("F_optimal:")
    # printPretty(F_optimal)
    # print("frobenius norm F")
    # print(np.linalg.norm(F_optimal, 'fro'))

    

    # This ranking does not work -- why?
    H_row_means = np.mean(np.abs(H_optimal), axis=1)
    # print("H row means:")
    # print(H_row_means)
    H_magnitude_ranking = rankdata(H_row_means, method='ordinal') # does increasing

    y_magnitude_ranking = rankdata(y_t_minus_1, method='ordinal') # does increasing

    print("Ranking of rows in H:", H_magnitude_ranking)
    print("Ranking of y sample entries:", y_magnitude_ranking)

    # Assert that the rankings match
    print("Rankings match between H and y?", np.array_equal(H_magnitude_ranking, y_magnitude_ranking))


    # As soon as h becomes all 0's, break the program and stop everything
    if np.allclose(h_t_minus_1, np.zeros(n*n), atol=1e-10):
       print("h_t_minus_1 has become all zeros!")
       quit()

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
    # PREDICTION
    h_t_prior = F_t_minus_1_optimal @ h_t_minus_1_post 
    # F: N^2 x N^2, Sigma: N^2 x N^2, Q: N^2 x N^2
    Sigma_t_prior = (F_t_minus_1_optimal @ Sigma_t_minus_1_post @ F_t_minus_1_optimal.T) + Q_t_minus_1  
    y_t_prior = H_t_optimal @ h_t_prior  

    # K CALCULATION
    S_t = (H_t_optimal @ Sigma_t_prior @ H_t_optimal.T) + R_t_minus_1  
    K_t = Sigma_t_prior @ (H_t_optimal.T) @ np.linalg.inv(S_t)  

    
    y_t = y_t.reshape(R_t_minus_1.shape[0], 1)
    # CORRECTION
    innovation = y_t - y_t_prior 
    h_t_post = h_t_prior + K_t @ innovation 
    Sigma_t_post = (np.eye(Sigma_t_prior.shape[0]) - K_t @ H_t_optimal) @ Sigma_t_prior @ (np.eye(Sigma_t_prior.shape[0]) - K_t @ H_t_optimal).T + K_t @ R_t_minus_1 @ K_t.T 

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

    R_t = (alpha * R_t_minus_1) + ( (1 - alpha) * ( (y_t - y_t_post) @(y_t - y_t_post).T + \
          (H_t_optimal @ Sigma_t_prior @ H_t_optimal.T) ))

    return Q_t, R_t


# CONTROL CODE PROCESSING WHOLE DATA MATRIX
def estimate_graph_connectivity(Y, lambda_param, alpha, Q_initial, R_initial, F_initial, H_initial, max_iter=100):
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
    print("Shape of Y: ", Y.shape)
    A = np.zeros((N, N, M)) 

    Q_t_minus_1 = Q_initial # (N^2 x N^2)
    R_t_minus_1 = R_initial # (N x N)
    F_t_minus_1_optimal = F_initial # (N^2 x N^2)
    H_t_optimal = H_initial # (N x N^2)
    print("Shape of H_initial: ", H_initial.shape)
    Sigma_t_minus_1_post = np.eye(N*N) # (N^2 x N^2)

    # TODO: Consdier alternative intializations -- not expected to work well on the first few iterations but thereafter should be able to adjust
    h_t_minus_4 = np.random.randn(N*N, 1)  # (N^2 x 1)
    h_t_minus_3 = np.random.randn(N*N, 1)
    h_t_minus_2 = np.random.randn(N*N, 1)
    h_t_minus_1 = np.random.randn(N*N, 1)

    print("type of h_t_minus_4 elements", type(h_t_minus_4[0]))

    h_t_post = np.zeros(N*N) # (N^2 x 1)

    print("M is ...  ", M)

    for t in range(3, M):  # Has to start at the third time step
        # DEFINING INPUTS TO THE FUNCTIONS
        print(f"Processing time step {t}...")

        # Pull the time lagged graph signals from the data matrix Y
        y_t = Y[:, t]
        y_t_minus_1 = Y[:, t - 1]
        y_t_minus_2 = Y[:, t - 2]
        y_t_minus_3 = Y[:, t - 3]

        print("   Processing part 1")
        # PART 1: Calculate optimal matrices F_{t-1}^* and H_t^*
        F_t_minus_1_optimal, H_t_optimal, \
            h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4 \
                 = calculate_optimal_F_H(y_t_minus_1, y_t_minus_2, y_t_minus_3, \
                    h_t_minus_1, h_t_minus_2, h_t_minus_3, h_t_minus_4, \
                        lambda_param, F_t_minus_1_optimal, H_t_optimal)

        print("   Processing part 2")
        # PART 2: Perform Kalman filter prediction and correction steps
        h_t_post, y_t_post, K_t, Sigma_t_prior, y_t_prior, Sigma_t_post = proposed_kalman_filter(h_t_minus_1, F_t_minus_1_optimal, Q_t_minus_1, Sigma_t_minus_1_post, H_t_optimal, R_t_minus_1, y_t)

        print("h t post shape", h_t_post.shape)

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



    # Plot the estimated graph connectivity
    '''
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(A[:, :, -1], cmap='binary')
    ax[0].set_title('True Graph Connectivity')
    ax[1].imshow(A_estimated[:, :, -1], cmap='binary')
    ax[1].set_title('Estimated Graph Connectivity')
    plt.show()
    '''
    
    '''
    # Compare it with the baseline methods
    As_GLASSO = time_varying_graphical_lasso(Generated_Y_static_SEM, alpha=alpha, beta=0.05)
    
    # for item in As_GLASSO: print the item pretty
    print("-------------------- SOLUTION WITH GLASSO --------------------")
    for i in range(SAMPLES):
        print(f"Time Step {i}")
        printPretty(As_GLASSO[i])'''



