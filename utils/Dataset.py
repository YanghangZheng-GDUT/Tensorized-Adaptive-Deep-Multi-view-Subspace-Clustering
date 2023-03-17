
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import scipy.io

class Dataset():
    def __init__(self, name):
        self.path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), './..')) + '/dataset/'
        self.name = name

    def load_data(self):
        data_path = self.path + self.name + '.mat'

        dataset = scipy.io.loadmat(data_path)

        if self.name == 'YaleB10_3_650':
            data = dataset['data'][0]
            y = dataset['label'][0]
            X = list()
            for i in range(data.shape[0]):
                data[i] = data[i].transpose()
                data[i] = self.normalize(data[i])
                X.append(data[i])
        elif self.name =='Caltech101-7':
            X = list()
            for i in range(len(dataset['X'])):
                x = dataset['X'][i][0]
                x = self.normalize_all(x)
                X.append(x)
            y = np.squeeze(dataset['Y'])
        else:
            X = None
            y = None

        return X, y

    def normalize(self, x, min=0):

        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def normalize_all(self, x):
        norm_x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return norm_x

    def l2_normalize(self, x, axis=0, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True),epsilon))
        return output

