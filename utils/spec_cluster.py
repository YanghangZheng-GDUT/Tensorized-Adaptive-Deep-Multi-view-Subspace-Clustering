import numpy as np
from munkres import Munkres
from sklearn import cluster
from sklearn import metrics
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

def pure_spectralclustering(C, K):
    C = 0.5*(C + C.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(C)
    grp = spectral.fit_predict(C) + 1
    return grp


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
        Z = U.dot(U.T)  # Z=USU'
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

def cluster_acc(y_true, y_pred):
    try:
        y_true = y_true - np.min(y_true)
        l1 = list(set(y_true))
        numclass1 = len(l1)
        l2 = list(set(y_pred))
        numclass2 = len(l2)
        ind = 0
        if numclass1 != numclass2:
            for i in l1:
                if i in l2:
                    pass
                else:
                    y_pred[ind] = i
                    ind += 1
        l2 = list(set(y_pred))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            # print('numclass1 != numclass2')
            return 0, 0, 0, 0, 0, 0, 0
        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
                cost[i][j] = len(mps_d)
        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()
        indexes = m.compute(cost)
        # get the match results
        new_predict = np.zeros(len(y_pred))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]
            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
            new_predict[ai] = c
        acc = metrics.accuracy_score(y_true, new_predict)
        f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
        precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
        recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
        f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
        precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
        recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
        # return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro
        return acc
    except:
        return 0, 0, 0, 0, 0, 0, 0