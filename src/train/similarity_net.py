import torch
import torch.nn as nn
import torch.nn.functional as F

from src.train.embedder_net import ImageEmbedder

# This class defines a Siamese neural network designed to approximate the SSIM score.
# It projects an image pair into a shared latent space using a shared feature extractor.
# The cosine similarity between the L2-normalized embeddings serves as the predicted continuous score.
# Author: Antonio Scardace

class SimilarityNet(nn.Module):

    def __init__(self, embedding_dim: int, dropout_prob: float) -> None:
        super().__init__()
        self.embedding_net = ImageEmbedder(embedding_dim, dropout_prob)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        emb1 = F.normalize(self.embedding_net(img1), p=2.0, dim=1)
        emb2 = F.normalize(self.embedding_net(img2), p=2.0, dim=1)
        cosine_sim = F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8)
        return cosine_sim.unsqueeze(1)