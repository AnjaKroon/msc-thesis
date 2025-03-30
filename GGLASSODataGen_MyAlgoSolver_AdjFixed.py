import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
import time
import argparse

from gglasso.helper.data_generation import generate_precision_matrix, group_power_network, sample_covariance_matrix
from gglasso.problem import glasso_problem
from gglasso.helper.basic_linalg import adjacency_matrix
from estimate_connectivity import estimate_graph_connectivity, printPretty, freeze_part_1_test_part_2
from estimate_connectivity import estimate_graph_connectivity
from datetime import datetime
from scipy.stats import rankdata 
from io import StringIO


def sample_covariance_matrix(Sigma, N, seed=None, deterministic=False):
    """
    Samples data for a given covariance matrix Sigma (with K layers).
    Returns: sample covariance matrix S and the sampled data.
    """
    if deterministic:
        rng = None
    else:
        rng = np.random.default_rng(seed)

    if len(Sigma.shape) == 2:
        # Case for a single covariance matrix
        assert abs(Sigma - Sigma.T).max() <= 1e-10
        (p, p) = Sigma.shape
        
        if deterministic:
            # Generate predefined deterministic samples
            predefined_samples = np.tile(np.arange(p), (N, 1)).T  # Shape: (p, N)
            sample = np.linalg.cholesky(Sigma) @ predefined_samples
        else:
            sample = rng.multivariate_normal(np.zeros(p), Sigma, N).T
            
        S = np.cov(sample, bias=True)
        
    else:
        # Case for multiple covariance matrices
        assert abs(Sigma - np.transpose(Sigma, axes=(0, 2, 1))).max() <= 1e-10
        (K, p, p) = Sigma.shape

        sample = np.zeros((K, p, N))
        for k in np.arange(K):
            if deterministic:
                # Generate predefined deterministic samples for each layer
                predefined_samples = np.tile(np.arange(p), (N, 1)).T  # Shape: (p, N)
                sample[k, :, :] = np.linalg.cholesky(Sigma[k, :, :]) @ predefined_samples
            else:
                sample[k, :, :] = rng.multivariate_normal(np.zeros(p), Sigma[k, :, :], N).T
    
        S = np.zeros((K, p, p))
        for k in np.arange(K):
            S[k, :, :] = np.cov(sample[k, :, :], bias=True)
            
    return S, sample

def add_noise_to_y(data):
    # generate a noise vector for the graph signal, a N x 1 vector with gaussian noise 0 mean 1 variance for every sample

    N , M = data.shape

    for i in range(M):
        cur_sample = data[:, i]
        # calculate the power of the signal
        signal_power = np.linalg.norm(cur_sample, 2)**2 / N

        sigma_noise = signal_power * (10**(-SNR_dB / 10))

        noise = np.random.normal(0, sigma_noise, N)
        
        data[:, i] = data[:, i] + noise


    return data

def dynamic_scaling_y(data_dyn, scaling_factor=1.0001):
    # generate a noise vector for the graph signal, a N x 1 vector with gaussian noise 0 mean 1 variance for every sample

    N , M = data_dyn.shape

    # expponential scaling factor, see if this does not blow up

    for i in range(M):
        if i == 0:
            pass
        else:
            # last_sample = data_dyn[:, i-1]
            # last_sample = last_sample * scaling_factor
            data_dyn[:, i] = data_dyn[:, i] * scaling_factor**i

    return data_dyn

def draw_adjacency(Theta, Y):
    A = adjacency_matrix(Theta)

    printPretty(A)

    G = nx.from_numpy_array(A)
    pos = nx.drawing.layout.spring_layout(G, seed = 1234)

    labels = {i: round(Y[i, 0],2) for i in range(len(Y))}

    plt.figure()
    nx.draw_networkx(G, pos = pos, node_color = "darkblue", edge_color = "darkblue", font_color = 'red', font_size=20, labels=labels, with_labels = True)

def create_A_tensor_noNoise(A):
    # create the A_tensor
    for i in range(SAMPLES):
        if i == 0:
            A_tensor = A.astype(float)  
        else:
            A_tensor = np.dstack((A_tensor, A))
    # print(A_tensor.shape)
    return A_tensor

def add_noise_to_A_tensor(sample, A):
    # Turn the adjacency matrix into a tensor, one adjacency per M

    N, M = sample.shape

    for i in range(M):
        # A = adjacency_matrix(Theta)
        if i == 0:
            A_tensor = A.astype(float)  
        else:
            A_tensor = np.dstack((A_tensor, A))

    # check the shape of the tensor
    # print("A_tensor shape: ", A_tensor.shape)

    # Add noise to the state matrix -- the adjacency based on the SNR previously
    for i in range(M):
        cur_A = A_tensor[:, :, i]
        
        signal_power = np.linalg.norm(cur_A, 2)**2 / N

        sigma_noise = signal_power * (10**(-SNR_dB / 10))

        noise = np.random.normal(0, sigma_noise, (N,N))

        A_tensor[:, :, i] = A_tensor[:, :, i] + noise

        printPretty(A_tensor[:, :, i])
    
    return A_tensor

def setup_logging():
    """Sets up logging and redirects print statements to log output."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # Ensure the logs directory exists

    # Generate a timestamped log filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"log_{timestamp}.txt")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",  # Remove timestamps/log levels for clean output
        handlers=[
            logging.FileHandler(log_filename, mode="w"),  # Save logs to file
            logging.StreamHandler(sys.__stdout__),  # Print logs to terminal
        ],
        force=True,
    )

    class PrintLogger:
        def __init__(self):
            self.buffer = StringIO()  # Temporary buffer for multi-line prints

        def write(self, message):
            self.buffer.write(message)
            if message.endswith("\n"):  # Only log when a full line is ready
                full_message = self.buffer.getvalue().rstrip()  # Get full buffered line
                self.buffer = StringIO()  # Reset buffer
                
                if full_message.strip():  # Avoid logging empty lines
                    logging.info(full_message)  
                    for handler in logging.getLogger().handlers:
                        handler.flush()  # Force write to file

        def flush(self):
            pass  # Required for compatibility with sys.stdout

    sys.stdout = PrintLogger()  
    sys.stderr = PrintLogger()  

def meanvar_evaluation(SAMPLES, A_estimated):
    total_average = 0
    tally = 0
    all_values = []  # Store all nonzero values for variance calculation

    for i in range(SAMPLES):
        # Extract nonzero elements of A_estimated
        nonzero_values = A_estimated[:, :, i][A_estimated[:, :, i] != 0]

        if nonzero_values.size > 0:
            tally += 1
            avg_A = np.mean(nonzero_values)
            total_average += avg_A
            all_values.extend(nonzero_values)  # Store values for variance calculation

    # Prevent division by zero
    if tally > 0:
        overall_avg = total_average / tally
        variance = np.var(all_values) if all_values else 0  # Compute variance of all nonzero values
        print("Total Average:", overall_avg)
        print("Variance:", variance)
    else:
        print("No nonzero values found in A_estimated, cannot compute total average or variance.")

def plot_differences(SAMPLES, A_tensor, A_estimated, name_of_test):
    # COMPARISON OF THE SOLUTION TO THE GROUND TRUTH VIA THE L2 NORM
    all_differences = []
    for i in range(SAMPLES):
        # print("L2 Norm of the difference between the estimated and ground truth adjacency matrix")
        diff = np.linalg.norm(A_tensor[:, :, i] - A_estimated[:, :, i], 2)
        # print(diff)
        all_differences.append(diff)

    # produce a plot of all differences in relation to their iteration number
    # clear plt
    plt.clf()
    plt.plot(all_differences)
    plt.xlabel("Iteration")
    plt.ylabel("L2 Norm of Difference")
    plt.title(f"Difference from G.T.: {name_of_test}")
    plt.show()

def print_A_comparison(SAMPLES, A_estimated, A_tensor):
    for i in range(SAMPLES):
        if i%1 == 0:
            print(f"Time Step {i}")
            printPretty(A_estimated[:, :, i])

            print("Comparing to the given input adjacency")
            printPretty(A_tensor[:, :, i])


############ MAIN #############

if __name__ == "__main__":
    setup_logging()

    p = 3 # nodes
    N = 2 # samples
    SNR_dB = 20

    print("Script name:", sys.argv[0])
    if len(sys.argv) < 4:
        print("Usage: python script.py <nodes> <samples> <SNR_dB>")
        sys.exit(1)
    p = int(sys.argv[1])
    N = int(sys.argv[2])
    SNR_dB = float(sys.argv[3])
    print(f"p = {p}")
    print(f"N = {N}")
    print(f"SNR_dB = {SNR_dB}")

    Sigma, Theta = generate_precision_matrix(p=p, M=1, style='erdos', prob=0.2, seed=1234)

    NODES = p
    SAMPLES = N
    EPSILON = 1e-10
    VALUE_THRESHOLD = 1e10
    ALPHA = 0.5
    SCALING = 1.01 # scaling factor for dynamic scaling of graph signals
    lambda_param = 0.1
    alpha = 0.05

    noise_on_y = True # ALWAYS TRUE otherwise singular matrix
    noise_on_Adj = False # SHOULD NEVER BE TRUE
    dynamic_scaling_of_data = False # If false, then in static scenario

    def startup():
        Q_initial = np.zeros(((NODES*NODES), (NODES*NODES)))
        R_initial = np.zeros((NODES, NODES))
        # R_initial = np.eye(NODES)
        # H_initial = np.zeros((NODES, NODES*NODES)) 
        H_initial = np.ones((NODES, NODES*NODES)) 
        F_initial = np.eye(NODES*NODES)

        Y = np.zeros((NODES, SAMPLES))  # Initialize the graph signal matrix to 0, a reset

        # Generate the samples
        S, sample = sample_covariance_matrix(Sigma, N, deterministic=True)

        Y = sample[:NODES, :SAMPLES] # should already be of these dimensions but just to check

        # Display the true adjacency 
        # draw_adjacency(Theta, Y)

        # print("True Adjacency Matrix:")
        # get the true adjacency
        A = adjacency_matrix(Theta)

        # print("UNDER DYNAMIC? ", dynamic_scaling_of_data)

        # print("Origional Samples Prior to Any Processing:")
        # printPretty(sample)
        # printPretty(Y)

        if noise_on_Adj:
            # print("NOISE ON ADJ: adding noise to adjacency")
            A_tensor = add_noise_to_A_tensor(sample, A)
        else:
            # print("NOISE ON ADJ: there is no noise on the adjacency")
            A_tensor = create_A_tensor_noNoise(A)

        Y_nonoise = Y.copy()
        # printPretty(Y_nonoise)

        if noise_on_y:
            # print("NOISE ON Y: adding noise to the graph signal on basis of the provided SNR")
            for i in range(SAMPLES):
                # print(Y[:, i])
                signal_power = np.linalg.norm(Y[:, i], 2)**2 / NODES # get signal power of the current sample, because it can change
                sigma_noise = signal_power * (10**(-SNR_dB / 10))
                noise = np.random.normal(0, sigma_noise, (NODES))
                # Covariance matrix of the nodes for the current time step

                Y[:, i] = Y[:, i] + noise
                # print(Y[:, i])
            # Y = Y + 0.1*np.random.randn(p, N)
            # printPretty(Y)

        return Q_initial, R_initial, H_initial, F_initial, Y, A_tensor, Y_nonoise

    # TEST 1
    
    name_of_test = "FREEZE PART 1 TESTING PART 2"
    print(f"------------------- {name_of_test} -------------------")
    Q_initial, R_initial, H_initial, F_initial, Y, A_tensor, Y_nonoise = startup()

    if dynamic_scaling_of_data:
        # Dynamic scaling of the samples if that is the experiment being run
        # print("Scaling the dynamic samples: ")
        Y = dynamic_scaling_y(Y, SCALING)
        # printPretty(Y)
        # print("Dynamic scaling of Y data, parts 1 and 3 are fixed. Part 2 is estimated.")
        scal_fac = SCALING
        # A_estimated = freeze_part_1_test_part_2_dynamicScalingGrSignal(Y, A_tensor, scal_fac, lambda_param, alpha, Q_initial, R_initial, F_initial, H_initial)
        A_estimated, groundtr_H_optimals = freeze_part_1_test_part_2(Y, Y_nonoise, A_tensor, lambda_param, alpha, Q_initial, R_initial, F_initial, H_initial)
    else:
        A_estimated, groundtr_H_optimals = freeze_part_1_test_part_2(Y, Y_nonoise, A_tensor, lambda_param, alpha, Q_initial, R_initial, F_initial, H_initial)

    meanvar_evaluation(SAMPLES, A_estimated)
    plot_differences(SAMPLES, A_tensor, A_estimated, name_of_test)
    print_A_comparison(SAMPLES, A_estimated, A_tensor)

    # TEST 2

    name_of_test = "FULL ALGORITHM TESTING with H initial solved via pseudoinv, Htmin1 as H_init in H* calculations"
    print(f"------------------- {name_of_test} -------------------")
    
    Q_initial, R_initial, H_initial, F_initial, Y, A_tensor, Y_nonoise = startup()

    A_estimated = estimate_graph_connectivity(Y, Y_nonoise, lambda_param, alpha, Q_initial, R_initial, F_initial, H_initial, \
        ground_truth_H=groundtr_H_optimals, ground_truth_adjacency_tensor=A_tensor)
    
    # A_estimated = estimate_graph_connectivity(Y, Y_nonoise, lambda_param, alpha, Q_initial, R_initial, F_initial, H_initial, ground_truth_adjacency_tensor=A_tensor)

    
    plot_differences(SAMPLES, A_tensor, A_estimated, name_of_test)
    print_A_comparison(SAMPLES, A_estimated, A_tensor)
    meanvar_evaluation(SAMPLES, A_estimated)

    # TEST 3
    '''
    name_of_test = "FULL ALGORITHM TESTING using pseudo for init with CONSTRAINED kalman filter for h control"
    print(f"------------------- {name_of_test} -------------------")    
    Q_initial, R_initial, H_initial, F_initial, Y, A_tensor, Y_nonoise = startup()
    A_estimated = estimate_graph_connectivity(Y, Y_nonoise, lambda_param, alpha, Q_initial, R_initial, F_initial, H_initial, \
        ground_truth_H=groundtr_H_optimals, ground_truth_adjacency_tensor=A_tensor, constrained=True)

    meanvar_evaluation(SAMPLES, A_estimated)
    plot_differences(SAMPLES, A_tensor, A_estimated, name_of_test)
    '''
    
