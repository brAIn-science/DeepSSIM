import torch

from src.utils.ssim import load_grayscale_image
from src.extractors.base import AbstractFeatureExtractor

# This class implements a Concrete Product in the Abstract Factory pattern for feature extraction.
# It performs inference using the DeepSSIM approach to compute image embeddings.
# Reference: https://arxiv.org/pdf/2509.16582
# Author: Antonio Scardace

class DeepSsimFeatureExtractor(AbstractFeatureExtractor):

    # Loads the pre-trained TorchScript model and moves it to the specified device.
    # Freezes model parameters to disable gradient updates.

    def __init__(self, model_path: str, device: torch.device, transforms) -> None:
        self.transforms = transforms
        self.device = device
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    # Computes feature embeddings for a batch of images.
    # Runs inference with no gradient tracking for better efficiency.

    def get_batch_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(batch)
        
    # Loads an image from a given path and applies preprocessing.
    # Returns its feature embedding.
        
    def get_image_embedding(self, path: str) -> torch.Tensor:
        image = load_grayscale_image(path, normalise=True)
        image = self.transforms(image).repeat(3, 1, 1).float()
        image = image.unsqueeze(0).to(self.device)
        return self.get_batch_embeddings(image).squeeze(0)