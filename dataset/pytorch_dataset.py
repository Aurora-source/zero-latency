import os
from torch.utils.data import Dataset

class NuScenesDataset(Dataset):
    def __init__(self, data_path="dataset/v1.0-mini/v1.0-mini"):
        self.data_path = data_path
        self.files = os.listdir(data_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]