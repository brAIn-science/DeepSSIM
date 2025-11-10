import cv2
import torch
import psutil
import argparse
import numpy as np

from tqdm import tqdm

from src.utils.utils import get_image_path
from src.utils.fsim import compute_fsim_features, compute_fsim_from_features

# This script generates an NxM FSIM score matrix for (real_image, synth_image) pairs.
# It uses multiprocessing to efficiently process one row per batch in parallel.
# A fixed random seed is set to ensure reproducibility and avoid variability in FSIM computation.
# Author: Antonio Scardace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers',        type=int, default=psutil.cpu_count(logical=False))
    parser.add_argument('--dataset_images_dir', type=str, required=True)
    parser.add_argument('--output_matrix_path', type=str, required=True)
    parser.add_argument('--real_indices_path',  type=str, required=True)
    parser.add_argument('--synth_indices_path', type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    # Loads index lists mapping real and synthetic images.  
    # Initializes the FSIM matrix (rows = real images, columns = synthetic images). 

    real_indices = np.load(args.real_indices_path)['data'].tolist()
    synth_indices = np.load(args.synth_indices_path)['data'].tolist()
    n, m = len(real_indices), len(synth_indices)
    fsim_matrix = np.zeros((n, m), dtype=np.float16)

    # Pre-computes FSIM features (phase congruency and gradient magnitude)
    # for all synthetic images to avoid redundant computation in the main loop.

    synth_features = []
    for skey in tqdm(synth_indices, 'Loading synthetic images', len(synth_indices)):
        synth_img = cv2.imread(get_image_path(skey, args.dataset_images_dir), cv2.IMREAD_GRAYSCALE)
        pc_s, gm_s = compute_fsim_features(synth_img)
        synth_features.append((pc_s, gm_s))

    # For each real image, computes its FSIM features.  
    # Compares them against all pre-computed synthetic features.  
    # Stores the FSIM scores in the corresponding matrix entries. 
    # Saves the final FSIM score matrix to disk in compressed NumPy format.  

    for i, rkey in enumerate(tqdm(real_indices, 'Making GMSD matrix', len(real_indices))):
        real_img = cv2.imread(get_image_path(rkey, args.dataset_images_dir), cv2.IMREAD_GRAYSCALE)
        pc_r, gm_r = compute_fsim_features(real_img)
        for j, (pc_s, gm_s) in enumerate(synth_features):
            fsim_matrix[i, j] = compute_fsim_from_features(pc_r, gm_r, pc_s, gm_s)

    np.savez(args.output_matrix_path, data=fsim_matrix)