import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

from src.extractors.base import AbstractFeatureExtractor

# This class implements a Concrete Product in the Abstract Factory pattern for feature extraction.
# It performs inference using a Contrastive Learning approach to compute image embeddings.
# Reference: https://arxiv.org/pdf/2402.01054
# Author: Lemuel Puglisi

class DarFeatureExtractor(nn.Module, AbstractFeatureExtractor):

    # Initializes the model, modifies ResNet-50 for grayscale input, and loads weights.
    # Freezes model parameters to disable gradient updates.

    def __init__(self, model_path: str, device: torch.device, transforms) -> None:
        super(DarFeatureExtractor, self).__init__()
        self.transforms = transforms
        self.device = device
        
        self.in_channels = 1 
        self.n_features = 128
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=2048, out_features=self.n_features, bias=True)
        self.fc_end = nn.Linear(self.n_features, 1)
        self.load_state_dict(torch.load(model_path, weights_only=True))
        self.to(self.device)
        
    # Computes L2-normalized feature embeddings for a batch of images.
    # Processes the batch without gradient computation for efficiency.

    def get_batch_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            emb = self.model(batch)
            return F.normalize(emb, p=2, dim=1)
        
    # Loads an image from a given path and applies preprocessing.
    # Returns its feature embedding.
        
    def get_image_embedding(self, path: str) -> torch.Tensor:
        tensor = self.transforms(path).unsqueeze(0).to(self.device)
        return self.get_batch_embeddings(tensor)[0].unsqueeze(0).cpu().numpy()