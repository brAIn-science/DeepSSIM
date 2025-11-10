import torch
import monai.transforms

from monai.data import Dataset

from src.datasets.semdedup import SemDeDupDataset
from src.scorers.semdedup import SemDeDupEmbeddingScorer
from src.extractors.semdedup import SemDeDupFeatureExtractor
from src.factories.base import AbstractMetricFactory
from src.scorers.base import AbstractEmbeddingScorer
from src.extractors.base import AbstractFeatureExtractor

# This class implements a Concrete Factory for SemDeDup components.
# It creates datasets, feature extractors, and scorers of the SemDeDup apporach.
# Follows the Abstract Factory pattern for modular evaluation pipeline design.
# Author: Antonio Scardace

class SemDeDupFactory(AbstractMetricFactory):

    def __init__(self, augment: bool) -> None:

        prob = 0.8 if augment else 0.0
        self.augmentations = monai.transforms.Compose([
            monai.transforms.RandFlip(spatial_axis=0, prob=prob),
            monai.transforms.RandFlip(spatial_axis=1, prob=prob),
            monai.transforms.RandRotate(range_x=0.1745, prob=prob),
            monai.transforms.RandAdjustContrast(gamma=(0.5, 1.5), prob=prob)
        ])

    def create_dataset(self, data: list[dict[str, str]]) -> Dataset:
        return SemDeDupDataset(data, self.preprocess, self.augmentations)

    def create_feature_extractor(self, model_path: str, device: torch.device) -> AbstractFeatureExtractor:
        extractor = SemDeDupFeatureExtractor(model_path, device)
        self.preprocess = extractor.preprocess
        return extractor

    def create_embedding_scorer(self) -> AbstractEmbeddingScorer:
        return SemDeDupEmbeddingScorer()