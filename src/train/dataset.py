import torch
import numpy as np
import pandas as pd
import monai.transforms

from monai.data import Dataset

from src.utils.utils import get_image_path
from src.utils.utils import load_grayscale_image

# This class implements a custom PyTorch Dataset for loading real and synthetic grayscale image pairs along with their ground-truth SSIM scores.
# It leverages MONAI to apply anatomy-preserving spatial and intensity augmentations, improving overall model generalization.
# Author: Antonio Scardace

class ImagePairDataset(Dataset):

    def __init__(self, data: pd.DataFrame, base_path: str) -> None:
        self.data = data
        self.base_path = base_path
        self.transforms = monai.transforms.Compose([
            monai.transforms.RandFlip(spatial_axis=0, prob=0.25),
            monai.transforms.RandFlip(spatial_axis=1, prob=0.25),
            monai.transforms.RandAffine(rotate_range=0.09, translate_range=(4.0, 4.0), padding_mode='zeros', prob=0.75),
            monai.transforms.RandZoom(min_zoom=0.9, max_zoom=1.05, prob=0.75),
            monai.transforms.RandBiasField(degree=3, coeff_range=(0.0, 0.1), prob=0.5),
            monai.transforms.RandAdjustContrast(gamma=(0.5, 1.5), prob=0.25),
            monai.transforms.RandGaussianSmooth(sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.50), prob=0.25),
            monai.transforms.ToTensor()
        ])

    # Loads and normalizes the requested image pair, injecting a unitary channel dimension for MONAI compatibility.
    # Following the augmentation pipeline, the grayscale channel is duplicated to satisfy the three-channel input requirement of the backbone.
    
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
            'ssim': torch.tensor(sample['ssim'], dtype=torch.float32)
        }