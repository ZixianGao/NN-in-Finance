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
        
class StockDataset(Dataset):
    def __init__(self, data, label, window_size=128, stride=10):
        self.features = data
        self.labels = label
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        # 计算所有样本可以构成的窗口数量
        total_windows = (len(self.features) - self.window_size) // self.stride + 1
        return total_windows

    def __getitem__(self, idx):
        # 计算当前窗口的起始索引
        start_idx = idx * self.stride
        # 如果窗口结束索引超过数据集末尾，则将结束索引置为数据集末尾
        end_idx = min(start_idx + self.window_size, len(self.features))
        # 提取特征和标签
        features = torch.tensor(self.features[start_idx:end_idx], dtype=torch.float32)
        label = torch.tensor(self.labels[start_idx:end_idx], dtype=torch.float32)
        # label = torch.tensor(self.labels[end_idx], dtype=torch.float32)
        return features, label

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'input': self.data[idx], 'label': self.labels[idx]}
        return self.data[idx],self.labels[idx]
