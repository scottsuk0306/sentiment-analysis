from torch.utils.data.dataset import Dataset
import pandas as pd
import os
import torch
import numpy as np

class IMDBDataset(Dataset):
    """IMDB Movie Reviews dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imdb_reviews = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.imdb_reviews)

    def __getitem__(self, idx):
        pass