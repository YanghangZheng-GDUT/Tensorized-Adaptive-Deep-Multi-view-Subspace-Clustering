import logging
import numpy as np
from utils.Dataset import Dataset
import utils.verify as verify
import os
from utils.log import log
import warnings
warnings.filterwarnings("ignore")
'''
each net has its own learning_rate(lr_xx), activation_function(act_xx), nodes_of_layers(dims_xx)
ae net need pretraining before the whole optimization
'''
from model import model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    pro_path = './'
    data_name = 'YaleB10_3_650'
    logger = log(pro_path, 'logs', data_name, is_cover=True)
    data = Dataset(data_name, pro_path)
    X, y_ture = data.load_data()
    sample_num = len(y_ture)
    n_clusters = len(set(y_ture))
    n_input1 = [50, 50]  # 165*4096 64x64
    n_input2 = [56, 59]  # 165*3304 56x59
    n_input3 = [75, 90]  # 165*6750 75x90
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

    para = dict()

    para['c_norm'] = 1
    para['c_express'] = 1
    para['cf'] = 1

    lr_pre = 1.0e-3
    lr_aesc = 1.0e-3
    lr_af = 1.0e-3
    lr = [lr_pre, lr_aesc, lr_af]


    Cf = model(pro_path, X, para, lr,n_inputs=n_inputs, channels=channels, kernels=kernels,
               data_name=data_name)
    res = verify.dev(Cf, n_clusters, y_ture)

    logging.info("\ndataset:{}"
                 "\nmetrics: ACC\tNMI \tPUR \tAR"
                 "\nresult:{}".format(data_name, np.round(res, 4)))




