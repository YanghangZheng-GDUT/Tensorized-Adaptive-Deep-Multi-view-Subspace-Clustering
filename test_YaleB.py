import logging

import numpy as np
from utils.Dataset import Dataset
import utils.verify as verify
import os
from utils.log import log
from multiprocessing import Process, Queue

from model import model

def main(data_name, Shuffle_sample_seed=None, q=None):
    data = Dataset(data_name)
    X, y_ture = data.load_data()
    v = len(X)
    sample_num = len(y_ture)

    if Shuffle_sample_seed is not None:
        print('Shuffling samples, random seed is {}'.format(Shuffle_sample_seed))
        np.random.seed(Shuffle_sample_seed)
        sample_order = np.random.permutation(sample_num)
        for vi in range(v):
            X[vi] = X[vi][sample_order]
        y_ture = y_ture[sample_order]

    n_input1 = [50, 50]
    n_input2 = [56, 59]
    n_input3 = [75, 90]
    n_inputs = [n_input1, n_input2, n_input3]
    for i in range(3):
        X[i] = X[i].reshape([sample_num, n_inputs[i][0], n_inputs[i][1], 1])
    channel_enc1 = [1, 6, 6, 10]
    channel_enc2 = [1, 6, 6, 10]
    channel_enc3 = [1, 8, 8, 12]
    channels = [channel_enc1, channel_enc2, channel_enc3]

    kernel1 = [5, 3, 3]
    kernel2 = [5, 5, 3]
    kernel3 = [5, 5, 5]
    kernels = [kernel1, kernel2, kernel3]
    n_clusters = len(set(y_ture))
    lr_pre = 1.0e-3
    lr_aesc = 1.0e-3
    lr_af = 1.0e-3
    lr = [lr_pre, lr_aesc, lr_af]

    para = dict()
    para['c_norm'] = 100
    para['c_express'] = 10
    para['cf'] = 0.001

    Cf = model(X, para, lr, n_inputs=n_inputs,channels=channels,kernels=kernels,data_name=data_name)

    if Shuffle_sample_seed is not None:
        res = verify.dev(Cf, n_clusters, y_ture)
        print("\ntime:{}"
              "\ndataset:{}"
              "\nmetrics: ACC\tNMI \tPUR \tAR"
              "\nresult:{}".format(int(Shuffle_sample_seed+1), data_name, np.round(res, 4)))
        q.put(res)
        return



if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/data/'
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    data_name = 'YaleB10_3_650'

    logger = log('logs', data_name, is_cover=True)
    ress = []
    for i in range(10):
        logging.info('the {} times testing'.format(i + 1))
        q = Queue()
        p = Process(target=main,  kwargs={'data_name':data_name,
                                          'Shuffle_sample_seed':i,
                                          'q':q})
        p.start()
        p.join()
        res = q.get()
        ress.append(res)
    m = np.round(np.mean(ress, axis=0), 4)
    logging.info("\ndataset:{}"
                 "\nmetrics: ACC\tNMI \tPUR \tAR"
                 "\nresult:{}".format(data_name, m))


