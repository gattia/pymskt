import numpy as np
from scipy.linalg import svd

def pca_svd(data):
    """
    Adapted from:
    "A Tutorial on Principal Component Analysis by Jonathon Shlens"
    https://arxiv.org/abs/1404.1100
    Inputs
    data = MxN matrix (M dimensions, N trials)
    Returns
    PC - each column is a PC
    V - Mx1 matrix of variances
    """
    M, N = data.shape
    mn = np.mean(data, axis=1)
    data = data - mn[:, None]  # produce centered data. If already centered this shouldnt be harmful.

    Y = data.T / np.sqrt(N - 1)

    U, S, V = svd(Y)
    PC = V.T  # V are the principle components (PC)
    V = S ** 2  # The squared singular values are the variances (V)

    return PC, V