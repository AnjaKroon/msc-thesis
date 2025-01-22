# Initialize random BA graph with weighted links distributed gaussianly

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import os
import sys

def printPretty(a):
    for row in a:
        for col in row:
            if col == 0.00:
                print("    0", end="    ")
            else:
                print("{:8.2f}".format(col), end=" ")
        print("")

def initialize_graph(N, K=1, exp_rate=5, gauss_mean=0.2, gauss_std=0.1):
    """
    Initialize a Barabási-Albert graph with weighted edges and Gaussian-distributed node values.
    
    Parameters:
        N (int): Number of nodes in the graph.
        K (int): Number of edges to attach from a new node to existing nodes.
        exp_rate (float): Rate parameter for the exponential distribution for edge weights.
        gauss_mean (float): Mean of the Gaussian distribution for initial node values.
        gauss_std (float): Standard deviation of the Gaussian distribution for initial node values.
    
    Returns:
        G (networkx.Graph): The initialized graph with weighted edges and node values.
        y_init (list): List of initial node values.
        A (numpy.matrix): The weighted adjacency matrix of the graph.
    """
    # Initialize the graph
    G = nx.barabasi_albert_graph(N, K)
    pos = nx.spring_layout(G)
    
    # Assign weights to edges using exponential distribution
    for (u, v) in G.edges():
        G[u][v]['weight'] = abs(round(random.expovariate(exp_rate), 2))
    
    # Initialize node values with Gaussian distribution
    y_init = []
    for n in G.nodes():
        value = round(random.gauss(gauss_mean, gauss_std), 2)
        G.nodes[n]['value'] = value
        y_init.append(value)
    
    # Get the weighted adjacency matrix
    A = nx.adjacency_matrix(G, weight='weight').todense()
    
    # Print adjacency matrix
    print("Adjacency matrix:")
    printPretty(A)
    
    # Check the eigenvalues of A to detect potential instability
    eigenvalues = np.linalg.eigvals(A)
    if any(abs(eigenvalues) > 1):
        print("\033[91mThe system will blow up!\033[0m")
    else:
        print("The system is stable.")
    
    return G, pos, y_init, A


# genY_staticSEM()
# Generate the data matrix Y according to the static undirected SEM model
# y_t = A * y_{t-1} + e_t

def genY_staticSEM(y_init, A, M, N, G, pos):
    """
    Generate the data matrix Y according to the static undirected SEM model.
    
    Parameters:
        y_init (list): List of initial node values.
        A (numpy.matrix): The weighted adjacency matrix of the graph.
        M (int): Number of samples.
    
    Returns:
        Y (numpy.matrix): The data matrix Y with M samples.
    """
    # Initialize the data matrix Y
    Y = np.array(y_init)  # y_init is a column vector
    Y = Y.reshape(-1, 1)  # convert to column vector
    
    # Generate the data matrix Y
    for m in range(1, M):
        noise = np.random.normal(0, 1, N)
        y_m = np.round(np.dot(Y[:, -1], A) + noise, 2)
        y_m = y_m.reshape(-1, 1)
        Y = np.column_stack((Y, y_m))
    
    return Y

def genY_dynamicSEM_piecewiseSlow(y_init, A, M, N, G, pos, step_interval=10, decrease_factor=0.8, increase_factor=1.2):
    """
    Generate the data matrix Y for a dynamic SEM model by adjusting edge weights in a periodic pattern.
    
    Parameters:
        y_init (list): List of initial node values.
        A (numpy.matrix): The initial weighted adjacency matrix of the graph.
        M (int): Number of samples.
        N (int): Number of nodes.
        step_interval (int): Interval at which to switch the flag for weight adjustment.
        decrease_factor (float): Factor to decrease weights by when the flag is True.
        increase_factor (float): Factor to increase weights by when the flag is False.
    
    Returns:
        Y (numpy.matrix): The data matrix Y with M samples.
    """
    # Initialize the data matrix Y with y_init as the first column
    Y = np.array(y_init).reshape(-1, 1)
    A_prev = A.copy()  # Start with the initial adjacency matrix
    flag = False  # Start with the flag set to False

    for m in range(1, M):
        # Switch the flag based on the step interval
        if m % step_interval == 0:
            flag = not flag
            print("\033[91mSwitching flag\033[0m")

        # Set the adjustment matrix based on the flag
        if flag:
            F_prev = decrease_factor * np.eye(N)
        else:
            F_prev = increase_factor * np.eye(N)

        # Update adjacency matrix with scaling factor
        A_cur = np.round(np.matmul(F_prev, A_prev), 2)
        print("A_cur:")
        printPretty(A_cur)

        # Generate the next y vector with noise
        noise = np.random.normal(0, 1, N)
        y_m = np.round(np.dot(Y[:, -1], A_cur) + noise, 2).reshape(-1, 1)
        
        # Append the new y vector to Y
        Y = np.column_stack((Y, y_m))
        
        # Set A_prev for the next iteration
        A_prev = A_cur

    return Y

def genY_dynamicSEM_randomSelectSmooth(y_init, A, M, N, G, pos, select_percent=0.1, weight_variation=0.05):
    """
    Generate the data matrix Y according to a dynamic SEM model with smooth weight changes.
    
    Parameters:
        y_init (list): List of initial node values.
        A (numpy.matrix): The weighted adjacency matrix of the graph.
        M (int): Number of samples.
        select_percent (float): Percentage of edges to modify each iteration.
        weight_variation (float): Amount to vary the weights by, positive or negative.
    
    Returns:
        Y (numpy.matrix): The data matrix Y with M samples.
    """
    Y = np.array(y_init).reshape(-1, 1)  # Initialize the data matrix Y
    A_prev = A.copy()  # Start with the initial adjacency matrix
    
    # Get edges and compute selection probabilities based on node degrees
    edges = np.array(list(G.edges()))
    degrees = np.array([deg for _, deg in G.degree()])
    edge_probs = np.array([degrees[u] + degrees[v] for u, v in edges])  # Sum of degrees for each edge
    edge_probs = edge_probs / edge_probs.sum()  # Normalize to get probabilities

    for m in range(1, M):
        # Select a subset of edges to adjust based on their probabilities
        num_to_select = max(1, int(select_percent * edges.shape[0]))  # Minimum of 1 edge
        selected_indices = np.random.choice(edges.shape[0], size=num_to_select, p=edge_probs, replace=False)
        selected_edges = edges[selected_indices]

        # Modify selected edges' weights
        for u, v in selected_edges:
            weight_change = random.choice([-weight_variation, weight_variation])
            new_weight = max(0, A_prev[u, v] + weight_change)  # Ensure weight remains non-negative
            A_prev[u, v] = new_weight
            A_prev[v, u] = new_weight  # Keep the graph undirected

        # Calculate y_m based on updated adjacency matrix
        noise = np.random.normal(0, 1, N)
        y_m = np.round(np.dot(Y[:, -1], A_prev) + noise, 2)
        y_m = y_m.reshape(-1, 1)
        Y = np.column_stack((Y, y_m))
    
    return Y


def plot_graph(G, pos):
    plt.figure()
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    labels = {n: f"{n}\n{G.nodes[n]['value']}" for n in G.nodes()}
    nx.draw(G, pos, with_labels=True, node_size=50, labels=labels, node_color="red", font_size=8)
    plt.show()

def checkSamples(Y):
    """
    Print the first, middle, and last columns of matrix Y.
    
    Parameters:
        Y (numpy.matrix): The data matrix to print columns from.
    """
    print("First column of Y:")
    print(Y[:, 0])

    print("Middle column of Y:")
    print(Y[:, Y.shape[1] // 2])

    print("Last column of Y:")
    print(Y[:, -1])


def generate_data_matrices(N=10, M=50, K=1):
    """
    Initialize a Barabási-Albert graph and generate data matrices for static and dynamic SEM models.

    Parameters:
        N (int): Number of nodes in the graph.
        M (int): Number of samples.
        K (int): Number of edges to attach from a new node to existing nodes.

    Returns:
        Y_static (numpy.matrix): Data matrix for the static SEM model.
        Y_dynamic_piecewiseSlow (numpy.matrix): Data matrix for the dynamic SEM model with piecewise slow adjustment.
        Y_dynamic_smoothSlow (numpy.matrix): Data matrix for the dynamic SEM model with smooth random selection adjustment.
    """
    # Initialize
    G, pos, y_init, A = initialize_graph(N, K)
    # plot_graph(G, pos)

    # Generate data for the static SEM model
    Y_static = genY_staticSEM(y_init, A, M, N, G, pos)
    # checkSamples(Y_static)

    # Generate data for the dynamic SEM model with piecewise slow adjustment
    Y_dynamic_piecewiseSlow = genY_dynamicSEM_piecewiseSlow(y_init, A, M, N, G, pos)
    # checkSamples(Y_dynamic_piecewiseSlow)

    # Generate data for the dynamic SEM model with smooth random selection adjustment
    Y_dynamic_smoothSlow = genY_dynamicSEM_randomSelectSmooth(y_init, A, M, N, G, pos)
    # checkSamples(Y_dynamic_smoothSlow)
    ''' 
    print("Statistics for Y_dynamic_smoothSlow:")
    print("Mean difference:", np.mean(np.abs(Y_dynamic_smoothSlow[:, 1:] - Y_dynamic_smoothSlow[:, :-1])))
    print("Max difference:", np.max(np.abs(Y_dynamic_smoothSlow[:, 1:] - Y_dynamic_smoothSlow[:, :-1])))
    print("Min difference:", np.min(np.abs(Y_dynamic_smoothSlow[:, 1:] - Y_dynamic_smoothSlow[:, :-1])))
    print("Standard deviation of differences:", np.std(np.abs(Y_dynamic_smoothSlow[:, 1:] - Y_dynamic_smoothSlow[:, :-1])))
    '''

    return Y_static, Y_dynamic_piecewiseSlow, Y_dynamic_smoothSlow

############ MAIN #############

Y_static, Y_dynamic_piecewiseSlow, Y_dynamic_smoothSlow = generate_data_matrices()














''' 
# Parameters
N = 10      # number of nodes
M = 50      # number of samples -- recall this is static so we don't yet have time steps
K = 1       # number of neighbors, visually tended to match my expectations regarding network sparsity

G, pos, y_init, A = initialize_graph(N, K)
plot_graph(G, pos)

# Generate data for the static SEM model
Y_static = genY_staticSEM(y_init, A, M)
# checkSamples(Y_static)


##  Generate data for the dynamic SEM model
# y_t = A_{t-1} * y_{t-1} + e_t

# OPTION 1: Decrease all edge weights for 10 steps by a very little amount, then increase them back for 10 steps, continue this pattern
Y_dynamic_piecewiseSlow = genY_dynamicSEM_piecewiseSlow(y_init, A, M, N)
# checkSamples(Y_dynamic_piecewiseSlow)



# OPTION 2: randomly selects 10% of the entries in F, with a preference for edges linked to high-degree nodes
# randomly adjusts the weights up or down by a small amount. Simulates the smooth variation in edge weights over time in a "smoothness graph"
Y_dynamic_smooth = genY_dynamicSEM_randomSelectSmooth(y_init, A, M)
# checkSamples(Y_dynamic_smooth)
print("Mean difference:", np.mean(np.abs(Y_dynamic_smooth[:, 1:] - Y_dynamic_smooth[:, :-1])))
print("Max difference:", np.max(np.abs(Y_dynamic_smooth[:, 1:] - Y_dynamic_smooth[:, :-1])))
print("Min difference:", np.min(np.abs(Y_dynamic_smooth[:, 1:] - Y_dynamic_smooth[:, :-1])))
print("Standard deviation of differences:", np.std(np.abs(Y_dynamic_smooth[:, 1:] - Y_dynamic_smooth[:, :-1])))

'''