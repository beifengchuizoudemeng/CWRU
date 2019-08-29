from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# __len__ 这样就可以通过len(dataset)返回数据集的大小。
# __getitem__ 支持索引，以便dataset[i]可以用来获取样本i
class SeriesDataset(Dataset):
    """series dataset."""

    def __init__(self, train=True,  transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        datasetName = "CWRU_1797"
        if train:
            tsv_file = "C:/Users/dreamby/Desktop/CWRU/Series/Single_CWRU_4/" + datasetName + "/" + "TRAIN.tsv"
        
        else:
            tsv_file = "C:/Users/dreamby/Desktop/CWRU/Series/Single_CWRU_4/" + datasetName + "/" + "TEST.tsv"

        self.data_frame = pd.read_csv(tsv_file, sep = '\t', header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        series = self.data_frame.iloc[idx-1][1:]
        series = np.array(series)
        series = series.astype(np.float32)
        series = torch.from_numpy(series)
        series = series.reshape(1, 400, -1)
        label = self.data_frame.iloc[idx-1][0]
        label = int(label) 

        if self.transform:
            series = self.transform(series)

        return series, label
