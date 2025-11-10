import torch
import monai.transforms

from monai.data import Dataset

from src.datasets.deepssim import DeepSsimDataset
from src.factories.base import AbstractMetricFactory
from src.scorers.base import AbstractEmbeddingScorer
from src.extractors.base import AbstractFeatureExtractor
from src.scorers.deepssim import DeepSsimEmbeddingScorer
from src.extractors.deepssim import DeepSsimFeatureExtractor

# This class implements a Concrete Factory for DeepSSIM components.
# It creates datasets, feature extractors, and scorers for our approach.
# Follows the Abstract Factory pattern for modular evaluation pipeline design.
# Author: Antonio Scardace

class DeepSsimFactory(AbstractMetricFactory):

    def __init__(self, augment: bool) -> None:

        prob = 0.8 if augment else 0.0
        self.transforms = monai.transforms.Compose([
            monai.transforms.RandFlip(spatial_axis=0, prob=prob),
            monai.transforms.RandFlip(spatial_axis=1, prob=prob),
            monai.transforms.RandRotate(range_x=0.1745, prob=prob),
            monai.transforms.RandAdjustContrast(gamma=(0.5, 1.5), prob=prob),
            monai.transforms.ToTensor()
        ])

    def create_dataset(self, data: list[dict[str, str]]) -> Dataset:
        return DeepSsimDataset(data, self.transforms)

    def create_feature_extractor(self, model_path: str, device: torch.device) -> AbstractFeatureExtractor:
        return DeepSsimFeatureExtractor(model_path, device, self.transforms)

    def create_embedding_scorer(self) -> AbstractEmbeddingScorer:
        return DeepSsimEmbeddingScorer()