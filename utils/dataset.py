import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BrainDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        image = data['image']
        mask = data['mask']

        # take middle slice
        slice_idx = image.shape[1] // 2

        image = image[:, slice_idx, :, :]
        mask = mask[slice_idx]

        # label: tumor or not
        label = 1 if mask.sum() > 0 else 0

        image = torch.tensor(image, dtype=torch.float32)

        return image, torch.tensor(label)
