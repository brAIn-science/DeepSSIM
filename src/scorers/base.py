import torch
import numpy as np

from abc import ABC
from abc import abstractmethod

# This class defines an Abstract Product in the Abstract Factory pattern for embedding scoring.
# It specifies the interface to compute similarity matrices from embeddings and perform classification.
# Subclasses must implement methods to compute score matrices and classify similarity scores.
# Author: Antonio Scardace

class AbstractEmbeddingScorer(ABC):

    @abstractmethod
    def compute_matrix(self, embs1: torch.Tensor, embs2: torch.Tensor) -> np.ndarray:
        pass

    @abstractmethod
    def classify(self, score: float) -> int:
        pass