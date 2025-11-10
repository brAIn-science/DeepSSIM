import cv2
import numpy as np

from typing import Tuple

# This function estimates phase congruency map from image gradients to compute FSIM.
# It captures perceptually relevant structural information for similarity comparison.
# Author: Antonio Scardace

def phase_congruency(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    dx, dy = np.gradient(img)
    magnitude = np.sqrt(dx**2 + dy**2)
    return magnitude / (np.max(magnitude) + 1e-8)

# This function computes gradient magnitude using Sobel filters to compute FSIM.
# It emphasizes edge strength and orientation changes in the image.
# Author: Antonio Scardace

def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    gy = gx.T
    grad_x = cv2.filter2D(img, -1, gx)
    grad_y = cv2.filter2D(img, -1, gy)
    return np.sqrt(grad_x**2 + grad_y**2)

# This function extracts FSIM features (phase congruency and gradient magnitude).
# These features are later used to compute the similarity score between images.
# Author: Antonio Scardace

def compute_fsim_features(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pc = phase_congruency(img)
    gm = gradient_magnitude(img)
    return pc, gm

# This function computes the FSIM score from pre-computed feature maps.
# It combines phase congruency and gradient magnitude similarities, weighted by PCm.
# Author: Antonio Scardace

def compute_fsim_from_features(pc_ref, gm_ref, pc_dist, gm_dist, T1=0.85, T2=160) -> float:
    S_g = (2 * gm_ref * gm_dist + T2) / (gm_ref**2 + gm_dist**2 + T2)
    S_pc = (2 * pc_ref * pc_dist + T1) / (pc_ref**2 + pc_dist**2 + T1)
    SL = S_g * S_pc
    PCm = np.maximum(pc_ref, pc_dist)
    return np.sum(SL * PCm) / (np.sum(PCm) + 1e-8)

# This function computes the FSIM score between two images given their file paths.
# The score ranges from 0 to 1.
# Author: Antonio Scardace

def compute_fsim(img_1_path: str, img_2_path: str) -> float:
    img_ref = cv2.imread(img_1_path, cv2.IMREAD_GRAYSCALE)
    img_dist = cv2.imread(img_2_path, cv2.IMREAD_GRAYSCALE)
    pc_ref, gm_ref = compute_fsim_features(img_ref)
    pc_dist, gm_dist = compute_fsim_features(img_dist)
    return compute_fsim_from_features(pc_ref, gm_ref, pc_dist, gm_dist)