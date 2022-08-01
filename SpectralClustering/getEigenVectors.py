import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.linalg import fractional_matrix_power ,eig

def computeNormalizedLaplacian(adjMatrix):
    # computing laplacian matrix
    D = np.diag(np.sum(adjMatrix, axis=1))
    L = D - adjMatrix
    eigVal, eigVec = eig(L)
    sorted_indecies = eigVal.argsort()
    # normalize the obtained eigen vectors
    eigVal = np.real(eigVal[sorted_indecies])
    eigVec = np.real(eigVec[:, sorted_indecies])
    # computing Normalized Spectral Clustering according to Andrew Ng, Jordan, and Weiss
    D_pow_neg_half = fractional_matrix_power(D, -0.5)
    Lnorm = np.dot(np.dot(D_pow_neg_half, L), D_pow_neg_half)
    return Lnorm

def computeEigenVectors(adjMatrix):
    Lnorm = computeNormalizedLaplacian(adjMatrix)
    # compute eigen values and vectors from the normalized laplacian matrix
    eigVal, eigVec = eig(Lnorm)
    sorted_indecies = eigVal.argsort()
    # normalize the obtained eigen vectors
    eigVal = np.real(eigVal[sorted_indecies])
    eigVec = np.real(eigVec[:, sorted_indecies])
    Kvectors = eigVec[:, :self.K]
    norm = np.sum(np.sqrt(Kvectors ** 2), axis=1)
    norm = np.reshape(len(norm), 1)
    KvectorsNorm = Kvectors / norm
    return KvectorsNorm

matrix=np.array([[0,1,1,0,0],
                [1,0,1,0,0],
                [1,1,0,0,0],
                [0,0,0,0,1],
                [0,0,0,1,0]])

matrix2=np.array([[0,1,1,0,1,0],
                [1,0,1,0,0,0],
                [1,1,0,1,0,0],
                [0,0,1,0,1,1],
                [1,0,0,1,0,1],
                [0,0,0,1,1,0]])

matrix3=np.array([[0,0.8,0.6,0.1,0,0],
                [0.8,0,0.9,0,0,0],
                [0.6,0.9,0,0,0,0.2],
                [0.1,0,0,0,0.6,0.7],
                [0,0,0,0.6,0,0.8],
                [0,0,0.2,0.7,0.8,0]])

eigenvectors=computeEigenVectors(matrix)
print('------------')