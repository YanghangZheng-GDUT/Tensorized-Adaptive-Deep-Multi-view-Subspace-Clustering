
import numpy as np
from numpy.linalg import svd
import math


def t_SVD(G, tau):
    Yhat = np.zeros(np.shape(G), complex)
    n3 = G.shape[2]
    D = np.fft.fft(G)
    halfn3 = math.ceil((n3 + 1)/2)
    for i in range(halfn3):
        U, S, V = svd(D[:, :, i], full_matrices=False)
        for j in range(np.shape(S)[0]):
            S[j] = max(S[j] - tau, 0)
        w = np.dot(np.dot(U, np.diag(S)), V)
        Yhat[:, :, i] = w
        if i > 0:
            w = np.dot(np.dot(U.conjugate(), np.diag(S)), V.conjugate())
            Yhat[:, :, n3-i] = w

    return np.fft.ifft(Yhat).real