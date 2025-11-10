import torch
import PIL.Image
import numpy as np

from typing import Union
from einops import rearrange
from monai.data import Dataset

# This class defines a custom dataset for the SemDeDup approach.
# The dataset handles image loading and transformation using a specified pipeline.
# Each sample contains an image (transformed) and its corresponding UID.
# Author: Antonio Scardace

class SemDeDupDataset(Dataset):

    def __init__(self, data: list[dict[str, str]], preprocess, augmentations) -> None:
        self.data = data
        self.preprocess = preprocess
        self.augmentations = augmentations
    
    def __getitem__(self, idx: int) -> dict[str, Union[str, torch.Tensor]]:
        sample = self.data[idx]
        image = np.array(PIL.Image.open(sample['image']).convert('RGB'))
        image = rearrange(image, 'H W C -> C H W')
        image = self.augmentations(image)
        image = rearrange(image, 'C H W -> H W C')
        image = self.preprocess(PIL.Image.fromarray(image.numpy().astype('uint8')))
        return { 'uid': sample['uid'], 'image': image }