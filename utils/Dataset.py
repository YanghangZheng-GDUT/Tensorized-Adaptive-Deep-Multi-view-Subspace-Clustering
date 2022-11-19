import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.io


class Dataset():
    def __init__(self, name, pro_path):
        self.path = pro_path + 'dataset/'
        self.name = name

    def load_data(self):
        data_path = self.path + self.name + '.mat'
        dataset = scipy.io.loadmat(data_path)

        if self.name =='Caltech101-7':
            X = list()
            for i in range(len(dataset['X'])):
                x = dataset['X'][i][0]
                x = self.normalize_all(x)
                X.append(x)
            y = np.squeeze(dataset['Y'])
        elif self.name == 'YaleB10_3_650':
            data = dataset['data'][0]
            y = dataset['label'][0]
            X = list()
            for i in range(data.shape[0]):
                data[i] = data[i].transpose()
                data[i] = self.normalize(data[i])
                X.append(data[i])
        else:
            X = None
            y = None

        return X, y

    def normalize(self, x, min=0):
        if min == 0:
            scaler = MinMaxScaler([0, 1])
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def normalize_all(self, x):
        norm_x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return norm_x
