# Returns syntehtically generated matrices for five different test scenarios.
# Tests conducted to ensure stability and reasonable values.

# TODO: 
# Save the set of ground truth A matrices not F matrices
# Does the data remain gaussian with the same mu and sigma across 10 samples? It should be somewhat stationary across small time intervals.
# Return the ground truth adjacency matrices for each time step too in order to check it against what the proposed method is creating

# AXIS Seems to be being made based on a single time period and is not consistent across all the colors in the plot making it difficult to compare

# TODO Later: check that real world systems exhibit stability in having the eigenvalue less than 1
# TODO Later: I choose to use A_norm rather than A for stability reasons -- merits of this decision?? 

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import os
import sys

'''
PRIMARY FUNCTIONS
'''
def initialize_graph(N=10, M=50, K=1, EP=1e-10, VT=1e10, AL=0.95,  NM=1, NC=0.5, IWM=1, IWV=0.01, ISM=1, ISV=0.01, DF=0.95, IF=0.99):
    """
    Initialize a Barabási-Albert graph with weighted edges and Gaussian-distributed graph signal. Mean and std of graph signal are inputs.
    
    Parameters:
        N (int): Number of nodes in the graph.
        K (int): Number of edges to attach from a new node to existing nodes.
        gauss_mean (float): Mean of the Gaussian distribution for initial graph signal.
        gauss_std (float): Standard deviation of the Gaussian distribution for initial graph signal.
    
    Returns:
        G (networkx.Graph): The initialized graph with weighted edges and graph signal.
        pos (dict): Dictionary of node positions for plotting.
        y_init (list): List of initial graph signal.
        A (numpy.matrix): The weighted adjacency matrix of the graph.
        test_flag (bool): Flag indicating if the graph is connected and stable.
    """

    print()
    print()
    print("\033[94mInitializing graph...\033[0m")
    test_flag = True
    G = nx.barabasi_albert_graph(N, K)
    pos = nx.spring_layout(G)

    # Assign weights to edges using exponential distribution
    for (u, v) in G.edges():
        G[u][v]['weight'] = abs(round(random.gauss(IWM, IWV), 2))
    
    # Initialize graph signal with Gaussian distribution
    
    y_init = []
    for n in G.nodes():
        value = abs(round(random.gauss(ISM, ISV), 2)) # made them all positive
        G.nodes[n]['value'] = value
        y_init.append(value)    
    
    Adj = nx.adjacency_matrix(G, weight='weight').todense()

    # printPretty(A)

    # Check what the largest eigenvalue of A is
    # print("Largest eigenvalue of A: ", np.max(np.linalg.eigvals(Adj))) # should be less than 1 for stability, TODO LATER, check that real world systems exhibit this property

    # Is A symmetric?
    # print("FIRST CREATION OF A: Is A symmetric? ", np.allclose(A, A.T, atol=1e-8))

    # What is the current graph sparsity
    print("Graph sparsity: ", 1 - nx.density(G))

    # Check if graph connected
    if nx.is_connected(G):
        print("Graph is connected.")
    else:
        print("\033[91mGraph is not connected.\033[0m")
        test_flag = False

    
    # normalize the adjacency and check the largest eigenvalue
    # A_norm = Adj / (Adj.sum(axis=1, keepdims=True) + EP)
    # print the largest eigenvalue of A_norm
    #print("Largest eigenvalue of A_norm: ", np.max(np.linalg.eigvals(A_norm))) # should be close to 1 - it is

    print("Graph initialized.")

    return G, pos, y_init, Adj, test_flag


'''
SEM DATA MODEL
'''
# Colors remain same distribution across samples, colors are moving, values are stable (-300 to 300)
# Min and max values are consistently large, average remains close to NM
def genY_staticSEM(y_init, A, G, pos, N=10, M=50, K=1, EP=1e-10, VT=1e10, AL=0.95,  NM=1, NC=0.5, IWM=1, IWV=0.01, ISM=1, ISV=0.01, DF=0.95, IF=0.99):
    """
    Generate the data matrix Y according to the static undirected SEM model.
    
    Parameters:
        y_init (list): List of initial graph signal.
        A (numpy.matrix): The weighted adjacency matrix of the graph.
        M (int): Number of samples.
    
    Returns:
        Y (numpy.matrix): The data matrix Y with M samples.
        A_static (numpy.matrix): The adjacency matrix of the graph.
    """

    Y = np.array(y_init).reshape(-1, 1) # yes 
    A_static = A.copy()

    '''
    # Check that the largest eigenvalue of A_static is real and less than 1, if not, break and print error message in red
    if np.max(np.linalg.eigvals(A_static)) > 1:
        print("\033[91mLargest eigenvalue of A_static is greater than 1.\033[0m")
        return
    '''

    for m in range(1, M):
        noise_vector = np.random.normal(NM, NC, (N, 1)) # of dimension N x 1
        # noise_vector = np.ones((N, 1)) # of dimension N x 1

        y_m = np.matmul(  np.linalg.inv(np.eye(N) - 0.9*A_static) , noise_vector) # non weighted
        # y_m = np.matmul(  np.linalg.inv(np.eye(N) - (AL)*A_static) , noise_vector) # weighted
        Y = np.concatenate((Y, y_m), axis=1)
    
    # If value of the entry is super close to 0, set it to 0 for sparsity reasons
    Y[abs(Y) < EP] = 0
    
    return Y, A_static

# Seems good visually and stays around same magnitude
# When looking at all grtaphs together, looks very monotonous -- due to scale and memoryless I think
def genY_dynamicSEM_FixedF(y_init, A, G, pos, N=10, M=50, K=1, EP=1e-10, VT=1e10, AL=0.95,  NM=1, NC=0.5, IWM=1, IWV=0.01, ISM=1, ISV=0.01, DF=0.95, IF=0.99):
    """
    Generate the data matrix Y for a dynamic SEM model with fixed F matrices.
    
    Parameters:
        y_init (list): List of initial graph signal.
        A (numpy.matrix): The initial weighted adjacency matrix of the graph.
        M (int): Number of samples.
        N (int): Number of nodes.
        step_interval (int): Interval at which to switch the flag for weight adjustment.
        
    Returns:
        Y (numpy.matrix): The data matrix Y with M samples.
        F_matrices (numpy.matrix): List of F matrices for each time step.
    """

    Y = np.array(y_init).reshape(-1, 1)

    F = genRandomOrthogonalMatrix(N) # gets random orthonormal matrix

    # is F orthonormal? yes
    # print("Is F orthonormal? ", np.allclose(np.matmul(F, F.T), np.eye(N), atol=1e-8))

    A_prev_fixed = A.copy()

    # Is A_prev symmetric?
    # print("INITIAL FIXED F: Is A_prev symmetric? ", np.allclose(A_prev, A_prev.T, atol=1e-8))
    # Eigenvalue check
    # print("Eigenvalues of A_prev: ", np.max(np.linalg.eigvals(A_prev_fixed)))
    # print("Frobenius norm of A_norm: ", np.linalg.norm(A_prev_fixed))

    for m in range(1, M):
        # Calculate A_cur and enforce symmetry
        A_cur_fixed = np.round((np.matmul( F, A_prev_fixed) + np.matmul(F, A_prev_fixed).T) / 2, 2)
        
        # Generate noise and calculate the next signal
        noise_vector = np.random.normal(NM, NC, (N, 1))
        y_m = np.matmul(np.linalg.inv(np.eye(N) - 0.85*A_cur_fixed), noise_vector)
        Y = np.column_stack((Y, y_m))
        
        # Update A_prev with the symmetric A_cur for the next iteration
        A_prev_fixed = A_cur_fixed


    # If value of the entry is super close to 0, set it to 0 for sparsity reasons
    Y[abs(Y) < EP] = 0

    return Y

# Slow -- seems initially stable and then seems to explode to larger values and larger extremes
# Med -- seems to be workig but then again does start to swing to larger and smaller values
# Fast -- swings even faster, avg, min and max seem relatively stable, distributions of node colors seems to be fine, not consistent
def genY_dynamicSEM_piecewise(y_init, A, G, pos, N=10, M=50, K=1, EP=1e-10, VT=1e10, AL=0.95,  NM=1, NC=0.5, IWM=1, IWV=0.01, ISM=1, ISV=0.01, DF=0.95, IF=0.99, step_interval=10):
    """
    Generate the data matrix Y for a dynamic SEM model by adjusting edge weights in a periodic pattern.
    
    Parameters:
        y_init (list): List of initial graph signal.
        A (numpy.matrix): The initial weighted adjacency matrix of the graph.
        M (int): Number of samples.
        N (int): Number of nodes.
        step_interval (int): Interval at which to switch the flag for weight adjustment.
        decrease_factor (float): Factor to decrease adjacency weights by.
        increase_factor (float): Factor to increase adjacency weights by.
    
    Returns:
        Y (numpy.matrix): The data matrix Y with M samples.
        F_matrices (numpy.matrix): List of F matrices for each time step.
    """
    Y = np.array(y_init).reshape(-1, 1)

    A_prev_piece = A.copy()

    assert isinstance(A_prev_piece, np.ndarray), "A_prev_piece must be a numpy array"

    A_prev_piece = np.round((A_prev_piece + A_prev_piece.T) / 2, 2)

    piecewise_flag = False  

    initial_F = IF * np.eye(N)
    set_F_matrices = np.array([initial_F])

    for m in range(1, M):
        # Switch the flag based on the step interval (10 per)
        assert isinstance(step_interval, int), "step_interval must be an integer"

        if m % step_interval == 0:
            piecewise_flag = not piecewise_flag

        # Set F_prev based on the flag
        if piecewise_flag:
            F_prev = DF * np.eye(N)
        elif not piecewise_flag:
            F_prev = IF * np.eye(N)
        else:
            print("\033[91mFlag error\033[0m")
            break
        
        # Stack F_prev along the first axis
        set_F_matrices = np.concatenate((set_F_matrices, F_prev[np.newaxis, :, :]), axis=0)

        # Calculate A_cur and enforce symmetry by averaging with its transpose
        assert isinstance(A_prev_piece, np.ndarray), "A_prev_piece must be a numpy array"
        assert isinstance(F_prev, np.ndarray), "F_prev must be a numpy array"

        A_cur_piece = (np.matmul(F_prev, A_prev_piece) + np.matmul(F_prev, A_prev_piece).T) / 2

        # Generate noise and calculate the next signal
        noise_vector = np.random.normal(NM, NC, (N, 1))
        assert isinstance(noise_vector, np.ndarray), "noise_vector must be a numpy array"
        y_m = np.matmul(np.linalg.inv(np.eye(N) - 0.85*A_cur_piece), noise_vector)
        Y = np.column_stack((Y, y_m))

        # Update A_prev with the symmetric A_cur for the next iteration
        A_prev_piece = A_cur_piece

    # If value of the entry is super close to 0, set it to 0 for sparsity reasons
    Y[abs(Y) < EP] = 0

    return Y



'''
SVARM DATA MODEL
''' 
# After 20 samples stays the same plot
# When looking at all grtaphs together, looks very monotonous
def genY_staticSVARM(y_init, A, G, pos, N=10, M=50, K=1, EP=1e-10, VT=1e10, AL=0.95,  NM=1, NC=0.5, IWM=1, IWV=0.01, ISM=1, ISV=0.01, DF=0.95, IF=0.99):
    """
    Generate the data matrix Y according to the static undirected SVARM model.
    
    Parameters:
        y_init (list): List of initial graph signal.
        A (numpy.matrix): The weighted adjacency matrix of the graph.
        M (int): Number of samples.
    
    Returns:
        Y (numpy.matrix): The data matrix Y with M samples.
        A_static (numpy.matrix): The adjacency matrix of the graph.
    """

    Y = np.array(y_init).reshape(-1, 1)
    y_init = np.array(y_init).reshape(-1, 1)
    y_prev = y_init

    A_static = A.copy()

    print("Static SVARM")

    for m in range(1, M):
        noise_vector = np.random.normal(NM, NC, (N, 1))
        # noise_vector = np.ones((N, 1))
        ''' 
        print("I - A")
        printPretty( np.eye(N) - A_static)
        print("Inverse of I - A")
        printPretty( np.linalg.inv(np.eye(N) - A_static))
        print("Inverse of I - A * A")
        printPretty( np.matmul(  np.linalg.inv(np.eye(N) - A_static)  , A_static))
        print("y_prev")
        printPretty(y_prev)
        print("Inverse of I - A * A * y_prev")
        printPretty( np.matmul(np.matmul(  np.linalg.inv(np.eye(N) - A_static)  , A_static),  y_prev))
        print("Everything")
        printPretty(np.matmul(np.matmul(  np.linalg.inv(np.eye(N) - A_static)  , A_static),  y_prev) + noise_vector)
        '''
        # SVARM: y_t = [ (I - A_{t})^(-1) ] * [ A_{t-1}*y_{t-1} + noise ]
        first_term =  np.linalg.inv(np.eye(N) - (AL)*A_static) 
        second_term = A_static @ y_prev + noise_vector
        y_m = first_term @ second_term

        # y_m = np.matmul(np.matmul(  np.linalg.inv(np.eye(N) - (AL)*A_static)  , (1 - AL)*A_static), y_prev) + noise_vector # weighted
        # y_m = np.matmul(np.matmul(  np.linalg.inv(np.eye(N) - A_static)  , A_static),   y_prev) + noise_vector # without weighted

        # print("y_m")
        # printPretty(y_m)
        # Add y_m as column to data matrix Y if Y is a np array
        Y = np.concatenate((Y, y_m), axis=1)

        # print("Y last column")
        # printPretty(Y[:, -1].reshape(-1, 1))
        y_prev = Y[:, -1].reshape(-1, 1)
    
    # If value of the entry is super close to 0, set it to 0 for sparsity reasons
    Y[abs(Y) < EP] = 0
    
    return Y, A_static

# Looks good visually but NOT same magnitude -- much larger
# When looking at all grtaphs together, looks very monotonous
def genY_dynamicSVARM_FixedF(y_init, A, G, pos, N=10, M=50, K=1, EP=1e-10, VT=1e10, AL=0.95,  NM=1, NC=0.5, IWM=1, IWV=0.01, ISM=1, ISV=0.01, DF=0.95, IF=0.99):
    """
    Generate the data matrix Y for a dynamic SEM model with fixed F matrices.
    
    Parameters:
        y_init (list): List of initial graph signal.
        A (numpy.matrix): The initial weighted adjacency matrix of the graph.
        M (int): Number of samples.
        N (int): Number of nodes.
        step_interval (int): Interval at which to switch the flag for weight adjustment.
        
    Returns:
        Y (numpy.matrix): The data matrix Y with M samples.
        F_matrices (numpy.matrix): List of F matrices for each time step.
    """
    
    Y = np.array(y_init).reshape(-1, 1)
    y_init = np.array(y_init).reshape(-1, 1)
    y_prev = y_init

    F = abs(genRandomOrthogonalMatrix(N))
    # make F symmetric
    F = (F + F.T) / 2

    A_prev_fixed = A.copy()

    for m in range(1, M):
        # print("Adjacency matrix importing: ")
        # printPretty(A_prev_fixed)

        A_cur_fixed = np.round( F @ A_prev_fixed, 2)

        # give A_cur_fixed the same sparsity mask as A_prev_fixed -- fixed a lot of issues
        A_cur_fixed[abs(A_prev_fixed) < EP] = 0
        # But also fixed the edges in place, no dynamic edges


        # Generate noise and calculate the next signal
        noise_vector = np.random.normal(NM, NC, (N, 1))

        '''
        print("Adjacency matrix: ")
        printPretty(A_cur_fixed)
        print("I - 0.85*A: ")
        printPretty(np.eye(N) - (0.85)*A_cur_fixed)
        print("Inverse of I - 0.85*A: ")
        printPretty(np.linalg.inv(np.eye(N) - (0.85)**A_cur_fixed))
        print("Inverse of I - 0.85*A * A: ")
        printPretty(np.matmul(  np.linalg.inv(np.eye(N) - (0.85)*A_cur_fixed)  , A_cur_fixed))
        print("Inverse of I - 0.85*A * A * y_prev: ")
        printPretty(np.matmul(np.matmul(  np.linalg.inv(np.eye(N) - (0.85)*A_cur_fixed)  , A_cur_fixed),  y_prev))
        print("Everything: ")
        printPretty(np.matmul(np.matmul(  np.linalg.inv(np.eye(N) - (0.85)*A_cur_fixed)  , A_cur_fixed),   y_prev) + noise_vector)
        '''

        y_m = np.matmul(np.matmul(  np.linalg.inv(np.eye(N) - (AL)*A_cur_fixed)  , (1-AL)*A_cur_fixed),   y_prev) + noise_vector # weighted
        # y_m = np.matmul(np.matmul(  np.linalg.inv(np.eye(N) - (0.85)*A_cur_fixed)  , (1 - 0.85)*A_cur_fixed),   y_prev) + noise_vector # weighted
        Y = np.concatenate((Y, y_m), axis=1)
        y_prev = y_m
        
        # Update A_prev with the symmetric A_cur for the next iteration
        A_prev_fixed = A_cur_fixed

    # If value of the entry is super close to 0, set it to 0 for sparsity reasons
    Y[abs(Y) < EP] = 0
    
    return Y

# Slow -- seems initially stable and then seems to explode to larger values and larger extremes
# Med -- seems to be workig but then again does start to swing to larger and smaller values
# Fast -- swings even faster
def genY_dynamicSVARM_piecewise(y_init, A, G, pos, N=10, M=50, K=1, EP=1e-10, VT=1e10, AL=0.95,  NM=1, NC=0.5, IWM=1, IWV=0.01, ISM=1, ISV=0.01, DF=0.95, IF=0.99, step_interval=10):
    """
    Generate the data matrix Y for a dynamic SEM model by adjusting edge weights in a periodic pattern.
    
    Parameters:
        y_init (list): List of initial graph signal.
        A (numpy.matrix): The initial weighted adjacency matrix of the graph.
        M (int): Number of samples.
        N (int): Number of nodes.
        step_interval (int): Interval at which to switch the flag for weight adjustment.
        decrease_factor (float): Factor to decrease adjacency weights by.
        increase_factor (float): Factor to increase adjacency weights by.
    
    Returns:
        Y (numpy.matrix): The data matrix Y with M samples.
        F_matrices (numpy.matrix): List of F matrices for each time step.
    """
    Y = np.array(y_init).reshape(-1, 1)
    y_init = np.array(y_init).reshape(-1, 1)
    y_prev = y_init

    A_prev_piece = A.copy()

    piecewise_flag = False  


    for m in range(1, M):
        # Switch the flag based on the step interval (10 per)
        if m % step_interval == 0:
            piecewise_flag = not piecewise_flag

        # Set F_prev based on the flag
        if piecewise_flag:
            F_prev = DF * np.eye(N)
        elif not piecewise_flag:
            F_prev = IF * np.eye(N)
        else:
            print("\033[91mFlag error\033[0m")
            break

        # Calculate A_cur and enforce symmetry by averaging with its transpose
        A_cur_piece = (np.matmul(F_prev, A_prev_piece) + np.matmul(F_prev, A_prev_piece).T) / 2

        # Enforce same sparsity mask as in A_prev_piece
        A_cur_piece[abs(A_prev_piece) < EP] = 0

        # Generate noise and calculate the next signal
        noise_vector = np.random.normal(NM, NC, (N, 1))
        y_m = np.matmul(np.matmul(  np.linalg.inv(np.eye(N) - (AL)*A_cur_piece)  , (1 - AL)*A_cur_piece),   y_prev) + noise_vector # weighted
        Y = np.column_stack((Y, y_m))

        # Update A_prev with the symmetric A_cur for the next iteration
        A_prev_piece = A_cur_piece

        y_prev = y_m

    # If value of the entry is super close to 0, set it to 0 for sparsity reasons
    Y[abs(Y) < EP] = 0

    return Y



'''
HELPER FUNCTIONS
'''
def genRandomOrthogonalMatrix(N):
    """
    Generate a random orthogonal matrix of size NxN.
    
    Parameters:
        N (int): Size of the orthogonal matrix.
    
    Returns:
        Q (numpy.matrix): Random orthogonal matrix of size NxN.
    """

    H = np.random.randn(N, N)
    Q, _ = np.linalg.qr(H)
    return Q

def printPretty(a):
    for row in a:
        for col in row:
            if col == 0.00:
                print("    0", end="    ")
            else:
                print("{:8.2f}".format(col), end=" ")
        print("")



'''
PLOTTING FUNCTIONS
'''
def plot_graph(G, pos, title="Original Graph Visualization"):
    '''
    Plot the graph with node labels, edge weights, and a consistent color scale.
    
    Parameters:
        G (networkx.Graph): The graph to plot. Node values should be stored in the 'value' attribute of each node.
        pos (dict): Dictionary of node positions for plotting.
        title (str): Title of the plot.
    '''
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and axes object

    # Extract node values from the graph
    node_values = np.array([G.nodes[n]['value'] for n in G.nodes()])

    # Calculate the 5th and 95th percentiles for consistent color scaling
    vmin = np.percentile(node_values, 5)
    vmax = np.percentile(node_values, 95)

    # Draw edge labels with weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    # Draw the graph nodes with consistent color scaling
    nodes = nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=20,
        node_color=node_values,
        cmap=plt.cm.viridis,
        vmin=vmin,
        vmax=vmax,
        edge_color="gray",
        ax=ax
    )

    # Annotate each node with its value
    labels = {n: f"{G.nodes[n]['value']:.2f}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    # Calculate statistics: min, max, and average
    min_val = np.min(node_values)
    max_val = np.max(node_values)
    avg_val = np.mean(node_values)

    # Display statistics below the plot
    fig.text(0.5, 0.01, f"Min: {min_val:.2f}, Max: {max_val:.2f}, Avg: {avg_val:.2f}", 
             fontsize=10, ha="center", color="black")

    # Add a colorbar linked to the nodes
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
    cbar.set_label("Node Values")

    # Add title and show the plot
    plt.title(title, fontsize=14)
    folder_name = "plots"
    os.makedirs(folder_name, exist_ok=True)  
    file_path = os.path.join(folder_name, f"{title}.png")
    plt.savefig(file_path)
    plt.show()

def plot_data_from_synthetic_method(Y, title, G, pos, show_flag = False):
    """
    Plot the synthetic graph signal data using a graph representation.

    Parameters:
        Y (numpy.matrix): The data matrix of graph signals (nodes x time steps).
        title (str): Title for the plot.
        G (networkx.Graph): The graph structure.
        pos (dict): Node positions for plotting the graph.
    """
    # Determine the number of time steps and evenly sample 10 or fewer time steps
    num_time_steps = Y.shape[1]
    if num_time_steps <= 10:
        sampled_indices = range(num_time_steps)
    else:
        step_size = max(num_time_steps // 10, 1)
        sampled_indices = range(0, num_time_steps, step_size)

    # Create subplots for each sampled time step
    num_samples = len(sampled_indices)
    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 4))
    
    if num_samples == 1:  # Ensure axes is iterable even with one plot
        axes = [axes]

    vmin = np.percentile(Y, 1)  
    vmax = np.percentile(Y, 99) 


    for ax, time_idx in zip(axes, sampled_indices):
        # Get current value for node 1 and store it in node_1_cur_value
        node_1_cur_value = Y[0, time_idx]

        # Plot graph with node colors based on signal values at the given time step
        node_colors = Y[:, time_idx]

        nx.draw(
            G,
            pos,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            with_labels=False,
            node_size=10,
            ax=ax,
            vmin=vmin,  # Use 10th percentile as vmin
            vmax=vmax   # Use 90th percentile as vmax
        )

        for node, (x, y) in pos.items():
            node_value = node_colors[node]
            ax.text(
                x, y + 0.05,  # Slightly above the node position
                f"{node_value:.2f}",
                fontsize=10,
                color="black",
                ha="center"
            )
        ''' 
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[1],  
            node_color='red',
            node_size=10,
            ax=ax
        )
        ax.text(
            pos[1][0] + 0.001,  # x position of the node
            pos[1][1] + 0.001,  # y position slightly above the node
            f"{node_1_cur_value:.2f}",  # Format the value with 2 decimal places
            fontsize=18,
            color="red",
            ha="center"
        )'''
        # at the bottom of each subplot, tell me the max and min values of the current graph signal 
        ax.text(
            0.5, 0.01,
            f"min: {np.min(node_colors):.2f}, max: {np.max(node_colors):.2f}",
            fontsize=10,
            ha='center',
            transform=ax.transAxes
        )
        # just under the previous text, tell me the average of the node values in the current time step
        ax.text(
            0.5, 0.05,
            f"avg: {np.mean(node_colors):.2f}",
            fontsize=10,
            ha='center',
            transform=ax.transAxes
        )
        ax.set_title(f"M= {time_idx}")

    # Add a colorbar for the signal values, based on the percentile range
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.1)
    # Set the main title
    plt.suptitle(title, fontsize=16)
    folder_name = "plots"
    os.makedirs(folder_name, exist_ok=True)  # Create folder if it doesn't exist
    file_path = os.path.join(folder_name, f"{title.replace(' ', '_')}.png")
    plt.savefig(file_path)
    if show_flag:
        plt.show()

def plot_final_graphs(G, pos, all_Y_datamatrices):
    '''
    Plot all final graph signals for each test case using the graph initially created.

    Parameters:
        G (networkx.Graph): The graph structure.
        pos (dict): Node positions for plotting the graph.
        all_Y_datamatrices (dict): Dictionary containing various graph signal data (key: name, value: signal array).
    '''
    num_signals = len(all_Y_datamatrices)
    num_plots = num_signals + 1  
    num_cols = 3 
    num_rows = (num_plots + num_cols - 1) // num_cols  

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 3 * num_rows))
    axes = axes.flatten() 

    all_values = np.concatenate([values.flatten() for values in all_Y_datamatrices.values()])
    vmin = np.percentile(all_values, 5)  
    vmax = np.percentile(all_values, 95) 

    # Plot the origional signal
    first_signal_name, first_signal_values = next(iter(all_Y_datamatrices.items()))
    nx.draw(
        G, pos, 
        node_color=first_signal_values[:, 0], 
        cmap=plt.cm.viridis, 
        with_labels=False, 
        node_size=10, 
        ax=axes[0], 
        vmin=vmin, 
        vmax=vmax
    )
    axes[0].set_title("(ORIG)")
    
    # Plot the rest of the signals
    for i, (signal_name, signal_values) in enumerate(all_Y_datamatrices.items(), start=1):
        if i >= num_plots:  
            break
        ax = axes[i]
        node_1_cur_value = signal_values[0, -1]
        nx.draw(
            G, pos, 
            node_color=signal_values[:, -1], 
            cmap=plt.cm.viridis, 
            with_labels=False, 
            node_size=10, 
            ax=ax, 
            vmin=vmin, 
            vmax=vmax
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[1],  
            node_color='red',
            node_size=10,
            ax=ax
        )
        ax.text(
            pos[1][0] + 0.3,  # x position of the node
            pos[1][1] + 0.3,  # y position slightly above the node
            f"{node_1_cur_value:.2f}",  # Format the value with 2 decimal places
            fontsize=18,
            color="red",
            ha="center"
        )
        ax.set_title(signal_name)

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  
    fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1)
    folder_name = "plots"
    os.makedirs(folder_name, exist_ok=True)  # Create folder if it doesn't exist
    file_path = os.path.join(folder_name, "comparing_final_graphs.png")
    plt.savefig(file_path)
    plt.show()

def checkSamples(Y):
    """
    Print the first, middle, and last columns of matrix Y. To be used for testing and debugging.
    
    Parameters:
        Y (numpy.matrix): The data matrix to print columns from.
    """
    
    for i in range(0, Y.shape[1], 50):
        print(np.round(Y[:, i], 2))

    if np.isnan(Y).any():
        nan_indices = np.argwhere(np.isnan(Y))
        nan_columns = np.unique(nan_indices[:, 1])
        first_nan_column = nan_columns[0]
        print(Y[:, first_nan_column-1])
        print("\033[91mNaN values found in Y matrix.\033[0m")
        print(f"First column with NaN values: Column {first_nan_column}")
        return



'''
DATA GENERATION FUNCTION
'''
def generate_data_matrices(N=10, M=50, K=1, EP=1e-10, VT=1e10, AL=0.95,  NM=1, NC=0.5, IWM=1, IWV=0.01, ISM=1, ISV=0.01, DF=0.95, IF=0.99):
    """
    Initialize a Barabási-Albert graph and generate data matrices for static and dynamic SEM models.
    If any test fails, restart the function.
    
    Parameters:
        N (int): Number of nodes in the graph.
        M (int): Number of samples.
        K (int): Number of edges to attach from a new node to existing nodes.
    
    Returns:
        TEST 1
        Y_static (numpy.matrix): Data matrix for the static SEM model. F is the identity matrix implying adjacency remains same across timesteps.

        TEST 2
        Y_static_SVARM (numpy.matrix): Data matrix for the static SVARM model. F is the identity matrix implying adjacency remains same across timesteps.

        TEST 3
        Y_dynamic_fixedF (numpy.matrix): Data matrix for the dynamic SEM model with fixed F matrices.

        TEST 4
        Y_dynamicSVARM_fixedF (numpy.matrix): Data matrix for the dynamic SVARM model with fixed F matrices.

        TEST 5
        Y_dynamic_piecewiseSlow_SEM (numpy.matrix): Data matrix for the dynamic SEM model with piecewise slow weight adjustment.

        TEST 6
        Y_dynamic_piecewiseMedium_SEM (numpy.matrix): Data matrix for the dynamic SEM model with piecewise medium weight adjustment.

        TEST 7
        Y_dynamic_piecewiseFast_SEM (numpy.matrix): Data matrix for the dynamic SEM model with piecewise fast weight adjustment.

        TEST 8
        Y_dynamic_piecewiseSlow_SVARM (numpy.matrix): Data matrix for the dynamic SVARM model with piecewise slow weight adjustment.

        TEST 9
        Y_dynamic_piecewiseMedium_SVARM (numpy.matrix): Data matrix for the dynamic SVARM model with piecewise medium weight adjustment.

        TEST 10
        Y_dynamic_piecewiseFast_SVARM (numpy.matrix): Data matrix for the dynamic SVARM model with piecewise fast weight adjustment.

    """

    print("Nodes: ", N)
    print("Samples: ", M)

    stability_flag = True
    
    while True:
        try:
            G, pos, y_init, A, stability_flag = initialize_graph(N=N, M=M, K=K, EP=EP, VT=VT, AL=AL,  NM=NM, NC=NC, IWM=IWM, IWV=IWV, ISM=ISM, ISV=ISV, DF=DF, IF=IF) # pass all parameters
            
            if stability_flag is False:
                print("\033[91mTest failed: Issue in graph initialization. Restarting data gen.\033[0m")
                continue

            Y_static_SEM, GT_A_static = genY_staticSEM(y_init, A, G, pos, N=N, M=M, K=K, EP=EP, VT=VT, AL=AL,  NM=NM, NC=NC, IWM=IWM, IWV=IWV, ISM=ISM, ISV=ISV, DF=DF, IF=IF)

            Y_static_SVARM, GT_A_static_SVARM = genY_staticSVARM(y_init, A, G, pos, N=N, M=M, K=K, EP=EP, VT=VT, AL=AL,  NM=NM, NC=NC, IWM=IWM, IWV=IWV, ISM=ISM, ISV=ISV, DF=DF, IF=IF)
        
            Y_dynamicSEM_fixedF = genY_dynamicSEM_FixedF(y_init, A, G, pos, N=N, M=M, K=K, EP=EP, VT=VT, AL=AL,  NM=NM, NC=NC, IWM=IWM, IWV=IWV, ISM=ISM, ISV=ISV, DF=DF, IF=IF)
            
            Y_dynamicSVARM_fixedF = genY_dynamicSVARM_FixedF(y_init, A, G, pos, N=N, M=M, K=K, EP=EP, VT=VT, AL=AL,  NM=NM, NC=NC, IWM=IWM, IWV=IWV, ISM=ISM, ISV=ISV, DF=DF, IF=IF)
            
            Y_dynamic_piecewiseSlow_SEM = genY_dynamicSEM_piecewise(y_init, A, G, pos,  N=N, M=M, K=K, EP=EP, VT=VT, AL=AL,  NM=NM, NC=NC, IWM=IWM, IWV=IWV, ISM=ISM, ISV=ISV, DF=DF, IF=IF, step_interval=10)

            Y_dynamic_piecewiseMedium_SEM = genY_dynamicSEM_piecewise(y_init, A, G, pos,  N=N, M=M, K=K, EP=EP, VT=VT, AL=AL,  NM=NM, NC=NC, IWM=IWM, IWV=IWV, ISM=ISM, ISV=ISV, DF=DF, IF=IF, step_interval=5)

            Y_dynamic_piecewiseFast_SEM = genY_dynamicSEM_piecewise(y_init, A, G, pos,  N=N, M=M, K=K, EP=EP, VT=VT, AL=AL,  NM=NM, NC=NC, IWM=IWM, IWV=IWV, ISM=ISM, ISV=ISV, DF=DF, IF=IF, step_interval=3)

            Y_dynamic_piecewiseSlow_SVARM = genY_dynamicSVARM_piecewise(y_init, A, G, pos,  N=N, M=M, K=K, EP=EP, VT=VT, AL=AL,  NM=NM, NC=NC, IWM=IWM, IWV=IWV, ISM=ISM, ISV=ISV, DF=DF, IF=IF, step_interval=10)
            
            Y_dynamic_piecewiseMedium_SVARM = genY_dynamicSVARM_piecewise(y_init, A, G, pos,  N=N, M=M, K=K, EP=EP, VT=VT, AL=AL,  NM=NM, NC=NC, IWM=IWM, IWV=IWV, ISM=ISM, ISV=ISV, DF=DF, IF=IF, step_interval=5)

            Y_dynamic_piecewiseFast_SVARM = genY_dynamicSVARM_piecewise(y_init, A, G, pos,  N=N, M=M, K=K, EP=EP, VT=VT, AL=AL,  NM=NM, NC=NC, IWM=IWM, IWV=IWV, ISM=ISM, ISV=ISV, DF=DF, IF=IF, step_interval=3)

            data_gen_successful = True

            print("___________ SYN GEN ADJACENCY___________ ")
            printPretty(A)

            data_matrices = {
                "Y_static_SEM": Y_static_SEM,
                "Y_static_SVARM": Y_static_SVARM,
                "Y_dynamicSEM_fixedF": Y_dynamicSEM_fixedF,
                "Y_dynamicSVARM_fixedF": Y_dynamicSVARM_fixedF,
                "Y_dynamic_piecewiseSlow_SEM": Y_dynamic_piecewiseSlow_SEM,
                "Y_dynamic_piecewiseMedium_SEM": Y_dynamic_piecewiseMedium_SEM,
                "Y_dynamic_piecewiseFast_SEM": Y_dynamic_piecewiseFast_SEM,
                "Y_dynamic_piecewiseSlow_SVARM": Y_dynamic_piecewiseSlow_SVARM,
                "Y_dynamic_piecewiseMedium_SVARM": Y_dynamic_piecewiseMedium_SVARM,
                "Y_dynamic_piecewiseFast_SVARM": Y_dynamic_piecewiseFast_SVARM
            }
            
            print("l2 norm of initial graph signal for all tests:", np.linalg.norm(np.array(y_init)))
            for name, matrix in data_matrices.items():
                print()
                print("--------------------", name, "--------------------")
                # checkSamples(matrix)

                print("l2 norm of final graph signal for", name, ":", np.linalg.norm(matrix[:, -1]))

                if np.isnan(matrix).any():
                    print(f"\033[91mTest failed: {name} values are NaN. Restarting data gen.\033[0m")
                    data_gen_successful = False
                    break

                if np.any(matrix > VT):
                    print(f"\033[91mTest failed: {name} values above {VT}. Restarting data gen.\033[0m")
                    data_gen_successful = False
                    break

                if np.linalg.norm(matrix) > VT:
                    print(f"\033[91mTest failed: {name} L2 norm of final graph signal above {VT}. Restarting data gen.\033[0m")
                    data_gen_successful = False
                    break

            if not data_gen_successful or not stability_flag:
                continue 

            # PLOTTING
            plot_graph(G, pos)

            plot_final_graphs(G, pos, data_matrices)

            plot_data_from_synthetic_method(Y_static_SEM, "Static SEM", G, pos)
            # columns are not the same but they look the same visually 

            plot_data_from_synthetic_method(Y_static_SVARM, "Static SVARM", G, pos, show_flag = False)

            plot_data_from_synthetic_method(Y_dynamicSEM_fixedF, "Dynamic SEM with Fixed F", G, pos, show_flag = False)

            plot_data_from_synthetic_method(Y_dynamicSVARM_fixedF, "Dynamic SVARM with Fixed F", G, pos, show_flag = False)

            plot_data_from_synthetic_method(Y_dynamic_piecewiseSlow_SEM, "Dynamic SEM with Piecewise Slow", G, pos, show_flag = False)

            plot_data_from_synthetic_method(Y_dynamic_piecewiseMedium_SEM, "Dynamic SEM with Piecewise Medium", G, pos, show_flag = False)

            plot_data_from_synthetic_method(Y_dynamic_piecewiseFast_SEM, "Dynamic SEM with Piecewise Fast", G, pos, show_flag = False)

            plot_data_from_synthetic_method(Y_dynamic_piecewiseSlow_SVARM, "Dynamic SVARM with Piecewise Slow", G, pos, show_flag = False)

            plot_data_from_synthetic_method(Y_dynamic_piecewiseMedium_SVARM, "Dynamic SVARM with Piecewise Medium", G, pos, show_flag = False)

            plot_data_from_synthetic_method(Y_dynamic_piecewiseFast_SVARM, "Dynamic SVARM with Piecewise Fast", G, pos, show_flag = False)

            return Y_static_SEM, \
                Y_static_SVARM, \
                Y_dynamicSEM_fixedF, \
                Y_dynamicSVARM_fixedF, \
                Y_dynamic_piecewiseSlow_SEM, \
                Y_dynamic_piecewiseMedium_SEM, \
                Y_dynamic_piecewiseFast_SEM, \
                Y_dynamic_piecewiseSlow_SVARM, \
                Y_dynamic_piecewiseMedium_SVARM, \
                Y_dynamic_piecewiseFast_SVARM
                            

        except Exception as e:
            # Catch unexpected errors, print, and restart
            print(f"Error occurred: {e}")
            print("\033[91mAn error occurred; restarting function.\033[0m")
            continue


############ MAIN #############

if __name__ == "__main__":
    
    # TESTERS - the base case
    # NOISE_MEAN, NOISE_COVAR = 0.2, 0.01 -- needed to matrix inv. possible
    # INITIAL_WEIGHT_MEAN, INITIAL_WEIGHT_VAR = 1, 0
    # INITIAL_SIGNAL_MEAN, INITIAL_SIGNAL_VAR = 1, 0

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
                                                                         IWM=1, IWV=0.0001, \
                                                                            ISM=20, ISV=5, DF=0.97, IF=1.03)
    
    # Is the system observable??
    
    # N = nodes
    # M = samples
    # K = edges to attach from a new node to existing nodes
    # EP = epsilon
    # VT = value threshold
    # AL = alpha
    # NM = noise mean
    # NC = noise covariance
    # IWM = initial weight mean
    # IWV = initial weight variance
    # ISM = initial signal mean
    # ISV = initial signal variance
    # DF = decrease factor
    # IF = increase factor
    
    
    