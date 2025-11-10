import torch
import numpy as np

from src.scorers.base import AbstractEmbeddingScorer

# This class handles the scoring logic based on the SemDeDup approach.
# Intended for use in embedding-based similarity and deduplication tasks.
# Author: Antonio Scardace

class SemDeDupEmbeddingScorer(AbstractEmbeddingScorer):

    # Computes the similarity matrix between two sets of embeddings.
    # Uses cosine similarity after normalizing the embeddings.

    def compute_matrix(self, embs1: torch.Tensor, embs2: torch.Tensor) -> np.ndarray:
        embs1_norm = torch.nn.functional.normalize(embs1, p=2, dim=1)
        embs2_norm = torch.nn.functional.normalize(embs2, p=2, dim=1)
        return torch.mm(embs1_norm, embs2_norm.T).cpu().numpy()
    
    # Assigns labels based on score and threshold values.
    # 0 = Different (score < low_threshold)
    # 1 = Duplicate (score > upper_threshold)
    # 2 = Similar (score in between thresholds)

    def classify(self, score: float) -> int:
        if score < 0.97: return 0
        elif score >= 0.99905: return 1
        else: return 2