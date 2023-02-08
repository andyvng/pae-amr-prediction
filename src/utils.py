import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class JointDataset(Dataset):
    '''
    Creating data object
    '''

    def __init__(self, X, y, balanced=False):
        self.X = X
        self.y = y
        self.balanced = balanced

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DatasetFromDir(Dataset):
    '''
    Loading dataset from working directory and lable files
    '''
    def __init__(self, working_dir, ids_path, max_intensity=0.009055):
        self.working_dir = working_dir
        self.ids = pd.read_csv(ids_path, header=None).values
        self.max_intensity = max_intensity

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.code = self.ids.item(idx)
        self.spectra = np.loadtxt(os.path.join(self.working_dir, f"{self.code}.txt")) / self.max_intensity
        return torch.tensor(self.spectra)

class EarlyStopper:
    """
    Setting criteria for early stopping
    https://stackoverflow.com/a/73704579
    """
    def __init__(self, patience=10, min_delta=20):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False