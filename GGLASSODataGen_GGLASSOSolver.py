# Run GGLASSO after creating the initial graph with the initialize_graph function from syntheticDataGen.py

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from problem import glasso_problem

from sklearn.covariance import graphical_lasso
from MyDataGen import generate_data_matrices, initialize_graph, printPretty
from basic_linalg import trp

if __name__ == "__main__":
    
    NODES = 10
    SAMPLES = 100
    K_PARAM = 1
    EPSILON = 1e-10
    VALUE_THRESHOLD = 1e10
    ALPHA = 0.95
    NOISE_MEAN, NOISE_COVAR = 1, 0.5
    INITIAL_WEIGHT_MEAN, INITIAL_WEIGHT_VAR = 5, 0.01
    # INITIAL_SIGNAL_MEAN, INITIAL_SIGNAL_VAR = 1, 0.01
    INITIAL_SIGNAL_MEAN, INITIAL_SIGNAL_VAR = 20, 5
    DECREASE = 0.8
    INCREASE = 1.2

    # GENERATING SIGMA AND THETA

    Adj = np.zeros((NODES,NODES))
    Sigma = np.zeros((NODES,NODES))

    # Create the initial graph
    G, pos, y_init, Adj, test_flag = initialize_graph(NODES, SAMPLES, K_PARAM, EPSILON, VALUE_THRESHOLD, ALPHA, NOISE_MEAN, NOISE_COVAR, INITIAL_WEIGHT_MEAN, INITIAL_WEIGHT_VAR, INITIAL_SIGNAL_MEAN, INITIAL_SIGNAL_VAR, DECREASE, INCREASE)

    # symmetrize the adjacency matrix
    Adj = .5 * (Adj + Adj.T)

    print("Ground truth adjacency matrix")
    printPretty(Adj)

    # Calculate Sigma
    # There could be an alpha and beta parameter here 
    Sigma = np.linalg.pinv((np.eye(NODES) + 0.5* Adj), hermitian = True)
    Theta = np.linalg.pinv(Sigma, hermitian = True)

    # GENERATING SAMPLE COVARIANCE MATRIX -- RECOPY THIS
    rng = np.random.default_rng()

    # p are the number of nodes
    p = NODES
    # N are number of samples
    N = SAMPLES
    
        
    if len(Sigma.shape) == 2:
        assert abs(Sigma - Sigma.T).max() <= 1e-10
        (p,p) = Sigma.shape
        # Uses the random number generator rng to generate N samples from a multivariate normal distribution with: Mean vector: 0 and Covariance matrix: Sigma
        sample = rng.multivariate_normal(np.zeros(p), Sigma, N).T
        print("Shape of sample: ", sample.shape)
        S = np.cov(sample, bias = True)
        
    else:
        assert abs(Sigma - trp(Sigma)).max() <= 1e-10 # forcing symmetric
        print("K is, ", K)
        (K,p,p) = Sigma.shape

        sample = np.zeros((K,p,N))
        for k in np.arange(K):
            sample[k,:,:] = rng.multivariate_normal(np.zeros(p), Sigma[k,:,:], N).T
    
        S = np.zeros((K,p,p))
        for k in np.arange(K):
            # normalize with N --> bias = True
            S[k,:,:] = np.cov(sample[k,:,:], bias = True)
    
    print("Shape of empirical covariance matrix: ", S.shape)
    print("Shape of the sample array: ", sample.shape)

    # DRAW TRUE GRAPH

    G = nx.from_numpy_array(Adj)
    # pos = nx.drawing.layout.spring_layout(G, seed = 1234)

    # plt.figure()
    # nx.draw_networkx(G, pos = pos, node_color = "darkblue", edge_color = "darkblue", font_color = 'white', with_labels = True)

    # RUN GLASSO
    P = glasso_problem(S, N, reg_params = {'lambda1': 0.05}, latent = False, do_scaling = False)
    print(P)

    # DO MODEL SELECTION
    lambda1_range = np.logspace(0, -3, 30)
    modelselect_params = {'lambda1_range': lambda1_range}

    P.model_selection(modelselect_params = modelselect_params, method = 'eBIC', gamma = 0.5)

    # regularization parameters are set to the best ones found during model selection
    print(P.reg_params)

    # PLOT RECOVERED GRAPH
    #tmp = P.modelselect_stats
    sol = P.solution.precision_
    P.solution.calc_adjacency(t = 1e-4)

    # Sparsify solution based on setting values below 0.05 to 0
    sol = np.where(np.abs(sol) < 0.1, 0, sol)


    fig, axs = plt.subplots(2,2, figsize=(10,8))
    node_size = 100
    font_size = 9

    nx.draw_networkx(G, pos = pos, node_size = node_size, node_color = "darkblue", edge_color = "darkblue", \
                    font_size = font_size, font_color = 'white', with_labels = True, ax = axs[0,0])

    axs[0,0].axis('off')
    axs[0,0].set_title("True graph")

    print("Recovered adjacency matrix")
    printPretty(P.solution.adjacency_)

    G1 = nx.from_numpy_array(P.solution.adjacency_)
    nx.draw_networkx(G1, pos = pos, node_size = node_size, node_color = "peru", edge_color = "peru", \
                font_size = font_size, font_color = 'white', with_labels = True, ax = axs[0,1])
    
    axs[0,1].axis('off')
    axs[0,1].set_title("Recovered graph")

    sns.heatmap(Theta, cmap = "coolwarm", vmin = -1.0, vmax = 1.0, linewidth = .5, square = True, cbar = False, \
                xticklabels = [], yticklabels = [], ax = axs[1,0])
    axs[1,0].set_title("True precision matrix")

    sns.heatmap(sol, cmap = "coolwarm", vmin = -1.0, vmax = 1.0, linewidth = .5, square = True, cbar = False, \
                xticklabels = [], yticklabels = [], ax = axs[1,1])
    axs[1,1].set_title("Recovered precision matrix")

    plt.show()

    print("True precision matrix")
    printPretty(Theta)

    print("Recovered precision matrix")
    printPretty(sol)
    



