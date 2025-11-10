import torch

from abc import ABC
from abc import abstractmethod

# This class defines an Abstract Product in an Abstract Factory pattern for feature extraction.
# It specifies the interface for computing image embeddings for similarity-based metrics.
# Subclasses must implement batch and single image embedding methods.
# Author: Antonio Scardace

class AbstractFeatureExtractor(ABC):

    @abstractmethod
    def get_batch_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_image_embedding(self, path: str) -> torch.Tensor:
        pass