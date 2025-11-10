import torch
import numpy as np

from typing import Union
from monai.data import Dataset

from src.utils.ssim import load_grayscale_image

# This class defines a custom dataset for the DeepSSIM approach.
# The dataset handles image loading and transformation using a specified pipeline.
# Each sample contains an image (transformed) and its corresponding UID.
# Author: Antonio Scardace

class DeepSsimDataset(Dataset):

    def __init__(self, data: list[dict[str, str]], transforms) -> None:
        self.data = data
        self.transforms = transforms
    
    def __getitem__(self, idx: int) -> dict[str, Union[str, torch.Tensor]]:
        sample = self.data[idx]
        image = load_grayscale_image(sample['image'], normalise=True)
        image = self.transforms(np.expand_dims(image, axis=0)).repeat(3, 1, 1)
        return { 'uid': sample['uid'], 'image': image }