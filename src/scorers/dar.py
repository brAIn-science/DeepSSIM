import torch
import numpy as np

from scipy.spatial.distance import cdist

from src.scorers.base import AbstractEmbeddingScorer

# This class handles the scoring logic based on the Dar et al. approach.
# Intended for use in embedding-based similarity and deduplication tasks.
# Author: Antonio Scardace

class DarEmbeddingScorer(AbstractEmbeddingScorer):
        
    # Computes the correlation similarity matrix between two sets of embeddings.
    # Measures similarity (1 - distance) using the correlation metric.

    def compute_matrix(self, embs1: torch.Tensor, embs2: torch.Tensor) -> np.ndarray:
        return 1 - cdist(embs1, embs2, metric='correlation')
    
    # Assigns labels based on score and threshold value.
    # 0 = Different (score <= threshold)
    # 1 = Duplicate (score > threshold)

    def classify(self, score: float) -> int:
        if score <= 0.8030425: return 0
        else: return 1
