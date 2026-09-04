import torch
import numpy as np
import monai.transforms

from monai.data import Dataset

from src.scorers.ums_l2 import UmsL2EmbeddingScorer
from src.factories.base import AbstractMetricFactory
from src.scorers.base import AbstractEmbeddingScorer
from src.extractors.base import AbstractFeatureExtractor

# This class implements a Concrete Factory for Chen et al. (UMS (L2)) components.
# It creates datasets, feature extractors, and scorers for the approach proposed by Chen et al.
# It follows the Abstract Factory pattern to enable a modular evaluation pipeline.
# Author: Antonio Scardace

class UmsL2Factory(AbstractMetricFactory):

    def __init__(self, augment: bool) -> None:
        pass

    def create_dataset(self, data: list[dict[str, str]]) -> Dataset:
        return None

    def create_feature_extractor(self, model_path: str, device: torch.device) -> AbstractFeatureExtractor:
        return None

    def create_embedding_scorer(self) -> AbstractEmbeddingScorer:
        return UmsL2EmbeddingScorer()