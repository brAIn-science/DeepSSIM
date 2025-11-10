import os
import torch
import psutil
import shutil
import argparse
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from collections import deque

from src.utils.ssim import process_batch

# This script generates an NxM SSIM score matrix for (real_image, synth_image) pairs.
# It uses multiprocessing to efficiently process one row per batch in parallel.
# A fixed random seed is set to ensure reproducibility and avoid variability in SSIM computation.
# Author: Antonio Scardace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers',        type=int, default=psutil.cpu_count(logical=False))
    parser.add_argument('--ssim_tmp_dir',       type=str, required=True)
    parser.add_argument('--dataset_images_dir', type=str, required=True)
    parser.add_argument('--output_matrix_path', type=str, required=True)
    parser.add_argument('--real_indices_path',  type=str, required=True)
    parser.add_argument('--synth_indices_path', type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs(args.ssim_tmp_dir, exist_ok=True)

    # Loads index lists mapping real and synthetic images.  
    # Initializes the SSIM matrix (rows = real images, columns = synthetic images). 

    real_indices = np.load(args.real_indices_path)['data'].tolist()
    synth_indices = np.load(args.synth_indices_path)['data'].tolist()
    n, m = len(real_indices), len(synth_indices)
    ssim_matrix = np.zeros((n, m), dtype=np.float16)

    # The generator function prevents loading all data into memory at once.
    # Each batch processes a single row of SSIM scores in parallel.

    def batch_gen():
        for ridx, real_key in enumerate(real_indices):
            yield ridx, real_key, synth_indices, args.dataset_images_dir, args.ssim_tmp_dir

    num_workers = min(args.num_workers, mp.cpu_count())
    num_workers = max(args.num_workers, 1)
    with mp.Pool(num_workers) as pool:
        tasks = pool.imap_unordered(process_batch, batch_gen())
        deque(tqdm(tasks, 'Computing SSIM rows', n), maxlen=0)

    # Aggregates the computed SSIM values into the final matrix.
    # Saves the matrix and removes the temporary directory containing per-row results.

    for ridx in tqdm(range(n), 'Making SSIM matrix', n):
        row_path = os.path.join(args.ssim_tmp_dir, str(ridx) + '.npz')
        row_data = np.load(row_path)['data']
        ssim_matrix[ridx, :] = row_data

    np.savez(args.output_matrix_path, data=ssim_matrix)
    shutil.rmtree(args.ssim_tmp_dir, ignore_errors=True)