import numpy as np
from munkres import Munkres
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score

import utils.spec_cluster as cluster


def dev(C, n_clusters, y_ture):

    y_pre = c_cluster(C, n_clusters)
    res = res_rate(y_ture, y_pre)
    return res


def c_cluster(c, n_clusters):

    c = cluster.thrC(c)
    y_pre, _ = cluster.post_proC(c, n_clusters)
    return y_pre


def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def res_rate(y_true, y_pred, average='macro', decimals=4):
    y_pred = best_map(y_true, y_pred)
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    # accuracy = np.round(accuracy, decimals)
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    # nmi = np.round(nmi, decimals)
    # purity
    pur = purity_score(y_true, y_pred)
    # pur = np.round(pur, decimals)
    # AR
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    # ari = np.round(ari, decimals)

    return [accuracy, nmi, pur, ari]


def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

