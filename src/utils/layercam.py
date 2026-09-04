import cv2
import torch
import numpy as np
import torch.nn.functional as F

from src.utils.utils import load_grayscale_image

# This function loads a grayscale image and prepares it for both model inference and visualization.
# It normalizes the intensity, adds the batch and channel dimensions required by the three-channel
# ConvNeXt input, and returns both the processed tensor and a display-ready RGB array.
# Author: Antonio Scardace

def load_and_format(image_path: str, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    image_np = load_grayscale_image(image_path, normalise=True)
    tensor = torch.from_numpy(image_np).float()
    tensor = tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
    display = np.clip(np.stack((image_np,) * 3, axis=-1), 0, 1)
    return tensor, display

# This function blends a generated activation heatmap with the original input image.
# It applies a Turbo colormap to the spatial weights, converts the channels to standard RGB format,
# and performs alpha-weighted superimposition to create the final visualization.
# Author: Antonio Scardace

def create_overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_TURBO)
    heatmap_color = heatmap_color.astype(np.float32) / 255.0
    heatmap_color = heatmap_color[:, :, ::-1]
    overlay = (alpha * heatmap_color) + ((1.0 - alpha) * image_rgb)
    return np.clip(overlay, 0.0, 1.0)

# This function retrieves the intermediate convolutional stages from the embedder backbone.
# These targeted layers serve as hook attachment points for extracting multi-scale feature maps.
# Author: Antonio Scardace

def get_target_layers(embedder: torch.nn.Module) -> dict:
    return {
        'stage_a': embedder.stage_a,
        'stage_b': embedder.stage_b,
        'stage_c': embedder.stage_c,
        'stage_d': embedder.stage_d,
    }

# This function generates a multi-scale activation heatmap by fusing responses from different network depths.
# It extracts LayerCAM maps from each specified target layer, resizes them to the original spatial resolution,
# and computes their normalized average to capture both low-level details and high-level semantics.
# Author: Antonio Scardace

def generate_fused_heatmap(embedder: torch.nn.Module, img_1: torch.Tensor, img_2: torch.Tensor, target_layers: dict) -> np.ndarray:
    def _extract_cam(layer, w, h):
        with SimilarityLayerCAM(embedder, layer) as cam:
            return cv2.resize(cam.generate_heatmap(img_1, img_2), (w, h))

    h, w = img_1.shape[2:]
    heatmaps = [_extract_cam(layer, w, h) for layer in target_layers.values()]
    fused = np.mean(heatmaps, axis=0)
    return (fused - fused.min()) / (np.ptp(fused) + 1e-8)

# This class implements Gradient-weighted Class Activation Mapping (LayerCAM) tailored for similarity tasks.
# It utilizes forward and backward hooks to capture intermediate activations and gradients, highlighting the
# spatial regions in the source image that maximize the cosine similarity with a target embedding.
# Author: Antonio Scardace

class SimilarityLayerCAM:
    
    def __init__(self, embedder: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.embedder = embedder.eval()
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def __enter__(self) -> "SimilarityLayerCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def _save_activation(self, module, input, output) -> None:
        self.activations = output.detach()
        
    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, img_real: torch.Tensor, img_target: torch.Tensor) -> np.ndarray:
        self.embedder.zero_grad()
        with torch.no_grad():
            emb_target = self.embedder(img_target)
            
        emb_real = self.embedder(img_real.requires_grad_(True))
        similarity = F.cosine_similarity(emb_real, emb_target, dim=1)
        similarity.mean().backward()
        cam = F.relu(torch.sum(F.relu(self.gradients[0]) * self.activations[0], dim=0))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()