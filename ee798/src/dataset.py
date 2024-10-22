import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class IntrinsicDataset(Dataset):
    def __init__(self, image_paths, intrinsic_paths, transform=None):
        self.image_paths = image_paths
        self.intrinsic_paths = intrinsic_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        intrinsic = np.load(self.intrinsic_paths[idx])
        
        if self.transform:
            image = self.transform(image)
        
        intrinsic = torch.tensor(intrinsic, dtype=torch.float32)
        return image, intrinsic
