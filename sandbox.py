import numpy as np

matrix = [[3, 3, 3], [4, 4, 4], [5, 5, 5]]
connectivity = [[0, 0, 0],
                [1, 1, 1],
                [0, 0, 0],
                [1, 1, 1], 
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 0]]

# inverse not possible for connectivity due to non square
# do pseudoinverse of connectivity for least squares solution

pseudo_inv_conn = np.linalg.pinv(connectivity)

# calculate H
H = np.dot(matrix, pseudo_inv_conn)

# check that it gives back the original matrix
print(np.allclose(matrix, np.dot(H, connectivity)))


print(H)


single_sample = [[3], [4], [5]] 
single_connectivity = [[0],
                [1],
                [0],
                [1], 
                [0],
                [1],
                [0],
                [1],
                [0]]

single_pseudo_inv_conn = np.linalg.pinv(single_connectivity)

single_H = np.dot(single_sample, single_pseudo_inv_conn)

print(np.allclose(single_sample, np.dot(single_H, single_connectivity)))

print(single_H)

alphabet = np.array([['a','b','c'], ['d','e','f'], ['g','h','i']])
print(alphabet)

N = alphabet.shape[0]

alphabet = alphabet.reshape((N*N), 1)

print(alphabet)

# Do the same but the alphabet array is 4 x 4
alphabet = np.array([['a','b','c','d'], ['e','f','g','h'], ['i','j','k','l'], ['m','n','o','p']])
N = alphabet.shape[0]
print(alphabet)
alphabet = alphabet.reshape((N*N), 1)
print(alphabet)