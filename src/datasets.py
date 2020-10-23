import numpy as np
import torch

from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset


def log_matrix(X, eps=1e-10):
    U, S, VT = np.linalg.svd(X)
    log_x = (U * np.log(np.maximum(S, eps))[:, None, :]) @ VT
    return log_x


# def log_euclidean_distance(x, y, eps=1e-10):
#     n = int(np.sqrt(np.prod(x.shape)))
#
#     def log_matrix(x):
#         U, S, VT = np.linalg.svd(x.reshape(n, n))
#
#         return log_x
#
#     dist = np.linalg.norm(log_matrix(x) - log_matrix(y), ord='fro')
#     return dist


class SimpleDataset(Dataset):
    def __init__(self, data_path, targets_path):
        self.data = np.load(data_path)
        self.targets = np.load(targets_path)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class TrainKNNDataset(Dataset):
    def __init__(self, data, targets, n_neighbors=5, n_jobs=None, log=False):
        self.data = data
        self.targets = targets

        self.n_neighbors = n_neighbors
        self.log_flag = log
        len_data = self.data.shape[0]

        self.log_data = log_matrix(self.data)
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
        self.knn.fit(self.log_data.reshape(len_data, -1))
        self.neighbors = np.hstack([
            np.arange(len_data)[..., None],
            self.knn.kneighbors(return_distance=False)
        ])

    def __getitem__(self, index):
        neighbors_index = self.neighbors[index]
        if self.log_flag:
            return self.log_data[neighbors_index], self.targets[index]
        else:
            return self.data[neighbors_index], self.targets[index]

    def __len__(self):
        return self.targets.shape[0]


class TestKNNDataset(Dataset):
    def __init__(self, data, targets, train_dataset):
        self.train_dataset = train_dataset
        self.data = data
        self.targets = targets

        len_data = self.data.shape[0]
        self.log_data = log_matrix(self.data)
        self.neighbors = self.train_dataset.knn.kneighbors(
            self.log_data.reshape(len_data, -1), n_neighbors=self.train_dataset.n_neighbors, return_distance=False
        )

    def __getitem__(self, index):
        neighbors_index = self.neighbors[index]

        if self.train_dataset.log_flag:
            values = np.vstack([
                self.log_data[index][None, ...],
                self.train_dataset.log_data[neighbors_index]
            ])
        else:
            values = np.vstack([
                self.data[index][None, ...],
                self.train_dataset.data[neighbors_index]
            ])

        return values, self.targets[index]

    def __len__(self):
        return self.targets.shape[0]
