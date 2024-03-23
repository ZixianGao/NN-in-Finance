import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

INITIAL_DATA_RATIO = 0.4
FOLD_RATIO_STEP = 0.15
N_TIME_FOLD = 4

class EvalDataset(Dataset):
    def __init__(self, data, label=None, window_size=128):
        self.features = data
        self.labels = label
        self.window_size = window_size
        self.stride = 1

    def __len__(self):
        windows = (len(self.features) - self.window_size) // self.stride + 1
        if (len(self.features) - self.window_size) % self.stride != 0:
            windows += 1
        return windows

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = min(start_idx + self.window_size, len(self.features))
        features = torch.tensor(self.features[start_idx:end_idx], dtype=torch.float32)
        if self.labels is not None:
            label = torch.tensor(self.labels[start_idx:end_idx], dtype=torch.float32)
            return features, label
        else:
            return features
