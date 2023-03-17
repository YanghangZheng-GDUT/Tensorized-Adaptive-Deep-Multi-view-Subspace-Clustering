import logging
import numpy as np
from utils.Dataset import Dataset
import utils.verify as verify
import os
from utils.log import log
import warnings
warnings.filterwarnings("ignore")
from model import model
from multiprocessing import Queue
from multiprocessing import Process

def main(data_name, q, Shuffle_sample_seed):
    data = Dataset(data_name)
    X, y_ture = data.load_data()
    sample_num = len(y_ture)
    n_clusters = len(set(y_ture))

    if Shuffle_sample_seed is not None:
        print('Shuffling samples, random seed is {}'.format(Shuffle_sample_seed))
        np.random.seed(Shuffle_sample_seed)
        sample_order = np.random.permutation(sample_num)
        for vi in range(len(X)):
            X[vi] = X[vi][sample_order]
        y_ture = y_ture[sample_order]

    dims_ae1 = [48, 50]
    dims_ae2 = [40, 40]
    dims_ae3 = [254, 100, 20]
    L_dims = [dims_ae1, dims_ae2, dims_ae3]

    n_input1 = [31, 64]
    n_input2 = [16, 32]
    n_input3 = [29, 32]
    n_inputs = [n_input1, n_input2, n_input3]
    for i in range(len(n_inputs)):
        k = i+len(L_dims)
        X[k] = X[k].reshape([sample_num, n_inputs[i][0], n_inputs[i][1], 1])

    channel_enc1 = [1, 3, 3, 5]
    channel_enc2 = [1, 3, 5]
    channel_enc3 = [1, 2, 2, 4]
    channels = [channel_enc1, channel_enc2, channel_enc3]

    kernel1 = [3, 3, 3]
    kernel2 = [3, 3]
    kernel3 = [3, 3, 3]
    kernels = [kernel1, kernel2, kernel3]

    lr_pre = 1.0e-3
    lr_aesc = 1.0e-3
    lr_af = 1.0e-3
    lr = [lr_pre, lr_aesc, lr_af]

    para = dict()
    para['c_norm'] = 1
    para['c_express'] = 0.01
    para['cf'] = 10

    Cf = model(X, para, lr, L_dims=L_dims, n_inputs=n_inputs, channels=channels, kernels=kernels,
               data_name=data_name)
    res = verify.dev(Cf, n_clusters, y_ture)
    print("\ntime:{}"
          "\ndataset:{}"
          "\nmetrics: ACC\tNMI \tPUR \tAR"
          "\nresult:{}".format(int(Shuffle_sample_seed+1), data_name, np.round(res, 4)))
    q.put(res)
    return

if __name__ == '__main__':
    data_name = 'Caltech101-7'
    logger = log('logs', data_name, is_cover=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    ress = []
    for i in range(10):
        logging.info('the {} times testing'.format(i + 1))
        q = Queue()
        p = Process(target=main, kwargs={'data_name': data_name,
                                         'q': q,
                                         'Shuffle_sample_seed': i})
        p.start()
        p.join()
        res = q.get()
        ress.append(res)
    m = np.round(np.mean(ress, axis=0), 4)

    logging.info("\ndataset:{}"
                 "\nmetrics: ACC\tNMI \tPUR \tAR"
                 "\nresult:{}".format(data_name, m))

