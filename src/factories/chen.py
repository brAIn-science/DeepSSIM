import torch
import numpy as np
import monai.transforms

from monai.data import Dataset

from src.datasets.chen import ChenDataset
from src.scorers.chen import ChenEmbeddingScorer
from src.extractors.chen import ChenFeatureExtractor
from src.factories.base import AbstractMetricFactory
from src.scorers.base import AbstractEmbeddingScorer
from src.extractors.base import AbstractFeatureExtractor

# This class implements a Concrete Factory for Chen et al. components.
# It creates datasets, feature extractors, and scorers of the Chen et al. apporach.
# Follows the Abstract Factory pattern for modular evaluation pipeline design.
# Author: Antonio Scardace

class ChenFactory(AbstractMetricFactory):

    def __init__(self, augment: bool) -> None:

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        prob = 0.8 if augment else 0.0

        self.transforms = monai.transforms.Compose([
            monai.transforms.Resize((288, 288)),
            monai.transforms.RandFlip(spatial_axis=0, prob=prob),
            monai.transforms.RandFlip(spatial_axis=1, prob=prob),
            monai.transforms.RandRotate(range_x=0.1745, prob=prob),
            monai.transforms.RandAdjustContrast(gamma=(0.5, 1.5), prob=prob),
            monai.transforms.EnsureType(dtype=np.float32),
            monai.transforms.Lambda(lambda x: x / 255.0),
            monai.transforms.NormalizeIntensity(mean, std, channel_wise=True),
            monai.transforms.ToTensor()
        ])

    def create_dataset(self, data: list[dict[str, str]]) -> Dataset:
        return ChenDataset(data, self.transforms)

    def create_feature_extractor(self, model_path: str, device: torch.device) -> AbstractFeatureExtractor:
        return ChenFeatureExtractor(model_path, device, self.transforms)

    def create_embedding_scorer(self) -> AbstractEmbeddingScorer:
        return ChenEmbeddingScorer()