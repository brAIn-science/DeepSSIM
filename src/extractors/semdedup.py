import torch
import PIL.Image

from open_clip import create_model_from_pretrained

from src.extractors.base import AbstractFeatureExtractor

# This class implements a Concrete Product in the Abstract Factory pattern for feature extraction.
# It performs inference using the SemDeDup approach to compute image embeddings.
# Reference: https://openreview.net/forum?id=u96ZBg_Shna
# Author: Antonio Scardace

class SemDeDupFeatureExtractor(AbstractFeatureExtractor):

    # Loads the pre-trained TorchScript model and moves it to the specified device.
    # Freezes model parameters to disable gradient updates.

    def __init__(self, model_path: str, device: torch.device, transforms=None) -> None:
        self.transforms = transforms
        self.device = device
        self.model, self.preprocess = create_model_from_pretrained(model_path)
        self.model = self.model.to(device).eval()
    
    # Computes feature embeddings for a batch of images.
    # Runs inference with no gradient tracking for better efficiency.

    def get_batch_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode_image(batch.to(self.device))
        
    # Loads an image from a given path and applies preprocessing.
    # Returns its feature embedding.

    def get_image_embedding(self, path: str) -> torch.Tensor:
        image = PIL.Image.open(path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.encode_image(image_tensor)