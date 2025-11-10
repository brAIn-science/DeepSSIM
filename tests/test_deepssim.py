import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

from src.utils.utils import get_image_path
from src.factories.registry import MetricFactoryRegistry

# This script verifies the correctness of the DeepSSIM similarity matrix.
# It compares stored DeepSSIM values with freshly computed DeepSSIM scores between image pairs.
# It is useful to ensure the integrity and consistency of the precomputed matrix.
# Author: Antonio Scardace

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_images_dir', type=str, required=True)
    parser.add_argument('--matrix_path',        type=str, required=True)
    parser.add_argument('--real_indices_path',  type=str, required=True)
    parser.add_argument('--synth_indices_path', type=str, required=True)
    parser.add_argument('--model_path',         type=str, required=True)
    args = parser.parse_args()

    # Loads the score matrix and the index lists.
    # The index lists map real and synthetic image keys to their respective positions in the score matrix.

    real_indices = np.load(args.real_indices_path)['data'].tolist()
    synth_indices = np.load(args.synth_indices_path)['data'].tolist()
    score_matrix = np.load(args.matrix_path)['data']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric_factory = MetricFactoryRegistry.get_metric('deepssim', False)
    feature_extractor = metric_factory.create_feature_extractor(args.model_path, device)
    
    # Iterates over all real image keys and, for each, randomly selects a synthetic key.
    # Compares the corresponding DeepSSIM value from the precomputed matrix with the freshly computed DeepSSIM.
    # Sets fixed seeds to ensure reproducibility. This removes randomness and ensures consistent results across runs.
    
    matrix_deepssim_list = []
    actual_deepssim_list = []

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    for real_key in tqdm(real_indices, 'Verifying DeepSSIM Matrix', len(real_indices)):
        synth_key = random.choice(synth_indices)
        sidx, ridx = synth_indices.index(synth_key), real_indices.index(real_key)
        matrix_deepssim_list.append(score_matrix[ridx, sidx])

        rpath = get_image_path(real_key, args.dataset_images_dir)
        spath = get_image_path(synth_key, args.dataset_images_dir)
        r_emb, s_emb = feature_extractor.get_image_embedding(rpath), feature_extractor.get_image_embedding(spath)
        actual_deepssim_list.append(F.cosine_similarity(r_emb, s_emb, dim=0).item())      

    # Computes the mean absolute error and standard deviation between stored and computed SSIM values.
    # Useful for theoretical analysis and for the matrix validation.
        
    diff = np.abs(np.array(matrix_deepssim_list) - np.array(actual_deepssim_list))
    mae, std_dev = np.mean(diff), np.std(diff)
    print('Mean Absolute Error =', mae)
    print('Standard Deviation =', std_dev)