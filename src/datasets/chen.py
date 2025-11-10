import torch
import PIL.Image
import numpy as np

from typing import Union
from einops import rearrange
from monai.data import Dataset

# This class defines a custom dataset for the Chen et al. approach.
# The dataset handles image loading and transformation using a specified pipeline.
# Each sample contains an image (transformed) and its corresponding UID.
# Author: Antonio Scardace

class ChenDataset(Dataset):

    def __init__(self, data: list[dict[str, str]], transforms) -> None:
        self.data = data
        self.transforms = transforms
    
    def __getitem__(self, idx: int) -> dict[str, Union[str, torch.Tensor]]:
        sample = self.data[idx]
        image = np.array(PIL.Image.open(sample['image']).convert('RGB'))
        image = rearrange(image, 'H W C -> C H W')
        image = self.transforms(image)
        return { 'uid': sample['uid'], 'image': image }