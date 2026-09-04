import torch
import PIL.Image
import numpy as np

from einops import rearrange

from src.extractors.base import AbstractFeatureExtractor

# This class implements a Concrete Product in the Abstract Factory pattern for feature extraction.
# It performs inference with the approach proposed by Chen et al. to compute image embeddings, as described in
# "SIDE: Surrogate Conditional Data Extraction from Diffusion Models".
# Reference: https://ojs.aaai.org/index.php/AAAI/article/view/36972
# Author: Antonio Scardace

class UmsSscdFeatureExtractor(AbstractFeatureExtractor):

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
        image = np.array(PIL.Image.open(path).convert('RGB'))
        image = rearrange(image, 'H W C -> C H W')
        image = self.transforms(image)
        batch = self.transforms(image).unsqueeze(0).to(self.device)
        return self.model(batch)[0, :]