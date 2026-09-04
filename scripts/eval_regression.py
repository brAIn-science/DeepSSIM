import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

# This script verifies the correctness of the precomputed SSIM similarity matrix.
# It compares the stored SSIM values against the ground-truth scores from the evaluation set.
# It enables a quantitative assessment of matrix integrity and indexing consistency.
# Author: Antonio Scardace

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_images_dir', type=str, required=True, help='Directory containing the dataset images.')
    parser.add_argument('--matrix_path',        type=str, required=True, help='Path to the precomputed score matrix file.')
    parser.add_argument('--real_indices_path',  type=str, required=True, help='Path to the real image index metadata file.')
    parser.add_argument('--synth_indices_path', type=str, required=True, help='Path to the synthetic image index metadata file.')
    parser.add_argument('--testset_csv',        type=str, required=True, help='Path to the test set CSV file containing ground truth SSIM scores.')
    args = parser.parse_args()

    # Loads the pre-computed score matrix and the index lists.
    # The index lists map real and synthetic image keys to their respective positions in the score matrix.

    real_indices = np.load(args.real_indices_path)['data'].tolist()
    synth_indices = np.load(args.synth_indices_path)['data'].tolist()
    score_matrix = np.load(args.matrix_path)['data']
    testset = pd.read_csv(args.testset_csv)
    
    # Extracts the predicted SSIM from the matrix and pairs it with the ground truth SSIM from the dataset.
    # Computes error metrics (MAE and RMSE) between stored and actual values to quantitatively validate the matrix.
    
    matrix_ssim_list = []
    actual_ssim_list = []

    for _, row in tqdm(testset.iterrows(), 'Computing predicted scores', len(testset)):
        ridx = real_indices.index(row['real_key'])
        sidx = synth_indices.index(row['synth_key'])
        matrix_ssim_list.append(score_matrix[ridx, sidx])
        actual_ssim_list.append(row['ssim'])        

    diff = np.array(matrix_ssim_list) - np.array(actual_ssim_list)
    mae, mae_std = np.mean(np.abs(diff)), np.std(np.abs(diff))
    rmse, rmse_std = np.sqrt(np.mean(diff**2)), np.std(diff**2) 
    print('MAE =', mae, '±', mae_std)
    print('RMSE =', rmse, '±', rmse_std)