import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        # Load and preprocess the training data
        data = pd.read_csv(os.path.join(data_path, 'train.csv')).values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        # Load and preprocess the test data if it exists
        if os.path.exists(os.path.join(data_path, 'test.csv')):
            test_data = pd.read_csv(os.path.join(data_path, 'test.csv')).values[:, 1:]
            test_data = np.nan_to_num(test_data)
            self.test = self.scaler.transform(test_data)
        else:
            self.test = None
        
        self.train = data
        self.val = self.test if self.test is not None else self.train

        print("test:", self.test.shape if self.test is not None else "None")
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val' and self.val is not None:
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test' and self.test is not None:
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return torch.tensor(self.train[index:index + self.win_size], dtype=torch.float32)
        elif self.mode == 'val' and self.val is not None:
            return torch.tensor(self.val[index:index + self.win_size], dtype=torch.float32)
        elif self.mode == 'test' and self.test is not None:
            return torch.tensor(self.test[index:index + self.win_size], dtype=torch.float32)
        else:
            return torch.tensor(self.train[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size], dtype=torch.float32)


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        # Load and preprocess the training data
        data = np.load(os.path.join(data_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        # Load and preprocess the test data if it exists
        if os.path.exists(os.path.join(data_path, "MSL_test.npy")):
            test_data = np.load(os.path.join(data_path, "MSL_test.npy"))
            self.test = self.scaler.transform(test_data)
        else:
            self.test = None
        
        self.train = data
        self.val = self.test if self.test is not None else self.train

        print("test:", self.test.shape if self.test is not None else "None")
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val' and self.val is not None:
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test' and self.test is not None:
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return torch.tensor(self.train[index:index + self.win_size], dtype=torch.float32)
        elif self.mode == 'val' and self.val is not None:
            return torch.tensor(self.val[index:index + self.win_size], dtype=torch.float32)
        elif self.mode == 'test' and self.test is not None:
            return torch.tensor(self.test[index:index + self.win_size], dtype=torch.float32)
        else:
            return torch.tensor(self.train[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size], dtype=torch.float32)


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        # Load and preprocess the training data
        data = np.load(os.path.join(data_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        # Load and preprocess the test data if it exists
        if os.path.exists(os.path.join(data_path, "SMAP_test.npy")):
            test_data = np.load(os.path.join(data_path, "SMAP_test.npy"))
            self.test = self.scaler.transform(test_data)
        else:
            self.test = None
        
        self.train = data
        self.val = self.test if self.test is not None else self.train

        print("test:", self.test.shape if self.test is not None else "None")
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val' and self.val is not None:
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test' and self.test is not None:
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return torch.tensor(self.train[index:index + self.win_size], dtype=torch.float32)
        elif self.mode == 'val' and self.val is not None:
            return torch.tensor(self.val[index:index + self.win_size], dtype=torch.float32)
        elif self.mode == 'test' and self.test is not None:
            return torch.tensor(self.test[index:index + self.win_size], dtype=torch.float32)
        else:
            return torch.tensor(self.train[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size], dtype=torch.float32)


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        # Load and preprocess the training data
        data = np.load(os.path.join(data_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        # Load and preprocess the test data if it exists
        if os.path.exists(os.path.join(data_path, "SMD_test.npy")):
            test_data = np.load(os.path.join(data_path, "SMD_test.npy"))
            self.test = self.scaler.transform(test_data)
        else:
            self.test = None
        
        self.train = data
        self.val = self.test if self.test is not None else self.train

        print("test:", self.test.shape if self.test is not None else "None")
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val' and self.val is not None:
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test' and self.test is not None:
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return torch.tensor(self.train[index:index + self.win_size], dtype=torch.float32)
        elif self.mode == 'val' and self.val is not None:
            return torch.tensor(self.val[index:index + self.win_size], dtype=torch.float32)
        elif self.mode == 'test' and self.test is not None:
            return torch.tensor(self.test[index:index + self.win_size], dtype=torch.float32)
        else:
            return torch.tensor(self.train[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size], dtype=torch.float32)


class AwakeLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        # Load and preprocess the awake.npy data
        data = np.load(os.path.join(data_path, "awake.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        self.train = data
        self.val = self.train  # No separate test/val data, using train data for val

        print("awake data:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode in ["train", "val"]:
            return torch.tensor(self.train[index:index + self.win_size], dtype=torch.float32)
        else:
            return torch.tensor(self.train[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size], dtype=torch.float32)


class sleepLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        # Load and preprocess the awake.npy data
        data = np.load(os.path.join(data_path, "sleep.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        self.train = data
        self.val = self.train  # No separate test/val data, using train data for val

        print("sleep data:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode in ["train", "val"]:
            return torch.tensor(self.train[index:index + self.win_size], dtype=torch.float32)
        else:
            return torch.tensor(self.train[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size], dtype=torch.float32)









def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset=''):
    if dataset == 'SMD':
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif dataset == 'MSL':
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'SMAP':
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'PSM':
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'AWAKE':
        dataset = AwakeLoader(data_path, win_size, step, mode)
    elif dataset == 'SLEEP':
            dataset = sleepLoader(data_path, win_size, step, mode)

    else:
        raise ValueError(f"Dataset '{dataset}' is not recognized.")

    shuffle = mode == 'train'

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
