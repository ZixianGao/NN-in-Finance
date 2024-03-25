import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class EvalDataset(Dataset):
    def __init__(self, data, label=None, window_size=128):
        self.features = data
        self.labels = label
        self.window_size = window_size

    def __len__(self):
        windows = (len(self.features) - self.window_size) + 1
        return windows

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = min(start_idx + self.window_size, len(self.features))
        features = torch.tensor(self.features[start_idx:end_idx], dtype=torch.float32)
        if self.labels is not None:
            label = torch.tensor(self.labels[end_idx - 1], dtype=torch.float32)
            return features, label
        else:
            return features