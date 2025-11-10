import torch
import numpy as np
import pandas as pd
import monai.transforms

from monai.data import Dataset

from src.utils.utils import get_image_path
from src.utils.ssim import load_grayscale_image

# This class is a custom PyTorch Dataset for loading grayscale image pairs along with their SSIM labels.
# Each sample consists of a "real" and a "synthetic" image, along with the related SSIM score.
# It applies light data augmentation using MONAI transforms: Random flips, rotation, and contrast shift.
# Author: Antonio Scardace

class ImagePairDataset(Dataset):

    def __init__(self, data: pd.DataFrame, base_path: str) -> None:
        self.data = data
        self.base_path = base_path
        self.transforms = monai.transforms.Compose([
            monai.transforms.RandFlip(spatial_axis=0, prob=0.33),
            monai.transforms.RandFlip(spatial_axis=1, prob=0.33),
            monai.transforms.RandRotate(range_x=0.17, prob=0.33),
            monai.transforms.RandAdjustContrast(gamma=(0.5, 1.5), prob=0.33),
            monai.transforms.ToTensor()
        ])
    
    # Loads a grayscale pair of images and normalizes them by lightness.
    # Adds a channel dimension to each image: [H, W] -> [1, H, W].
    # Applies MONAI transforms, then repeats the channel to convert from [1, H, W] to [3, H, W].
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.data.iloc[idx]
        img1_path = get_image_path(sample['real_key'], self.base_path)
        img2_path = get_image_path(sample['synth_key'], self.base_path)
        img1 = load_grayscale_image(img1_path, normalise=True)
        img2 = load_grayscale_image(img2_path, normalise=True)
        img1 = self.transforms(np.expand_dims(img1, axis=0)).repeat(3, 1, 1)
        img2 = self.transforms(np.expand_dims(img2, axis=0)).repeat(3, 1, 1)

        return {
            'img1': img1,
            'img2': img2,
            'ssim': torch.tensor(sample['ssim'], dtype=torch.float16)
        }