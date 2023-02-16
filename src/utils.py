import os
import numpy as np
import pandas as pd
import torch
import scipy.signal
import scipy.stats

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
    def __init__(self, 
                 working_dir, 
                 ast_label_path, 
                 ids_path, 
                 max_intensity=0.009055,
                 antimicrobials=["piperacillin_tazobactam" ,
                                 "meropenem",
                                 "ciprofloxacin",
                                 "tobramycin"]):
        self.working_dir = working_dir
        self.ast_label_path = ast_label_path
        self.ids = pd.read_csv(ids_path, header=None).values
        self.max_intensity = max_intensity
        self.antimicrobials = antimicrobials

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.code = self.ids.item(idx)
        self.spectra = np.loadtxt(os.path.join(self.working_dir, f"{self.code}.txt")) / self.max_intensity
        ast_label_df = pd.read_csv(self.ast_label_path, header=0)
        self.labels = ast_label_df.copy().loc[ast_label_df['id']==self.code, self.antimicrobials].to_numpy().squeeze()
        return torch.tensor(self.spectra), torch.tensor(self.antimicrobials)

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


def load_and_preprocess_spectra(input_dir,
                                ast_df,
                                antimicrobial,
                                ids,
                                bins, 
                                statistic='max'):
    '''
    Loading spectra from directory and preprocess spectra
    '''

    X = None
    y = None
    
    for id in ids:
        masses = [ (2000 + i) for i in range(18000) ]
        intensities = np.loadtxt(os.path.join(input_dir, f"{id}.txt")).reshape(1, -1)

        # Bin intensities and return bin means
        binned_intensities = scipy.stats.binned_statistic(masses,
                                                          intensities,
                                                          statistic=statistic,
                                                          bins=bins).statistic
        np.nan_to_num(binned_intensities, copy=False, nan=0)

        if X is not None:
            X = np.concatenate([X, binned_intensities], axis=0)
        else:
            X = binned_intensities

    # getting Y label
    tmp_df = pd.DataFrame({'id': ids})
    y = tmp_df.merge(ast_df, how='left', on='id')[antimicrobial].values

    nan_mask = np.isnan(y)

    return X[~nan_mask], y[~nan_mask]