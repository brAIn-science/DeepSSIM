import torch

from typing import Union
from monai.data import Dataset

# This class defines a custom dataset for the Dar et al. approach.
# The dataset handles image loading and transformation using a specified pipeline.
# Each sample contains an image (transformed) and its corresponding UID.
# Author: Antonio Scardace

class DarDataset(Dataset):

    def __init__(self, data: list[dict[str, str]], transforms) -> None:
        self.data = data
        self.transforms = transforms

    def __getitem__(self, idx: int) -> dict[str, Union[str, torch.Tensor]]:
        sample = self.data[idx]
        image = self.transforms(sample['image'])
        return { 'uid': sample['uid'], 'image': image }