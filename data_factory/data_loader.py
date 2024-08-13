
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class HealthDataLoader(Dataset):
    def __init__(self, data, win_size, step, mode="train", state="all"):
        self.mode = mode
        self.step = step
        self.win_size = win_size

        # Filter data based on the state
        if state == "sleep":
            self.data = data[data[:, -1] == 1]  # Only sleep data
        elif state == "awake":
            self.data = data[data[:, -1] == 0]  # Only awake data
        else:
            self.data = data  # Use all data

    def __len__(self):
        return (self.data.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        window_data = self.data[index:index + self.win_size]
        return np.float32(window_data[:, :-1])  # Exclude the sleep/awake label when returning the data

def get_loader_segment(data, batch_size, win_size=100, step=100, mode='train', state='all'):
    dataset = HealthDataLoader(data, win_size, step, mode, state)
    
    shuffle = mode == 'train'
    
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
