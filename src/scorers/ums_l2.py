import torch
import numpy as np

from src.scorers.base import AbstractEmbeddingScorer

# This class handles the scoring logic based on the Chen et al. (UMS (L2)) approach.
# It is intended for use in embedding-based similarity and deduplication tasks.
# Author: Antonio Scardace

class UmsL2EmbeddingScorer(AbstractEmbeddingScorer):

    # Computes the similarity matrix between two sets of embeddings.
    # Uses cosine similarity after normalizing the embeddings.

    def compute_matrix(self, embs1: torch.Tensor, embs2: torch.Tensor) -> np.ndarray:
        return None
    
    # Assigns labels based on score and threshold values.
    # 0 = Different (score <= low_threshold)
    # 1 = Duplicate (score > upper_threshold)
    # 2 = Similar (score in between thresholds)

    def classify(score: float) -> int:
        if score <= 1.84: return 0
        elif 1.84 < score <= 1.89: return 2
        else: return 1