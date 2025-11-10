import torch

from abc import ABC
from abc import abstractmethod
from monai.data import Dataset

from src.scorers.base import AbstractEmbeddingScorer
from src.extractors.base import AbstractFeatureExtractor

# This class defines an Abstract Factory for constructing components used in similarity-based evaluation.
# It declares methods to create datasets, feature extractors, and embedding scorers.
# Concrete factories must implement all creation methods.
# Author: Antonio Scardace

class AbstractMetricFactory(ABC):

    @abstractmethod
    def create_dataset(self, data: list[dict[str, str]]) -> Dataset:
        pass

    @abstractmethod
    def create_feature_extractor(self, model_path: str, device: torch.device) -> AbstractFeatureExtractor:
        pass

    @abstractmethod
    def create_embedding_scorer(self) -> AbstractEmbeddingScorer:
        pass