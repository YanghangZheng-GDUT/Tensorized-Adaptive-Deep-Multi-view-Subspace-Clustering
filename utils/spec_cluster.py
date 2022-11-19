import numpy as np
from munkres import Munkres
from sklearn import cluster
from sklearn import metrics
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

def thrC(C,ro=0.13):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C
    return Cp


def post_proC(C, K, d=9, alpha=21):
    try:
        C = 0.5*(C + C.T)
        r = d*K + 1
        U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
        U = U[:,::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis = 1)
        Z = U.dot(U.T)
        Z = Z * (Z>0)
        L = np.abs(Z ** alpha)
        L = L/L.max()
        L = 0.5 * (L + L.T)
        spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
        spectral.fit(L)
        grp = spectral.fit_predict(L) + 1
        return grp, L
    except:
        wrong_grep = np.zeros(C.shape[0])
        return wrong_grep, wrong_grep