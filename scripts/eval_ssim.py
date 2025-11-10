import torch
import random
import argparse
import monai.utils
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report

from src.utils.ssim import calculate_ssim
from src.utils.utils import get_image_path
from src.utils.ssim import calculate_ssim_augmented
from src.scorers.deepssim import DeepSsimEmbeddingScorer

# This script computes classification metrics to evaluate the scoring method.
# It builds y_true (ground truth labels) from the dataset and y_pred (predicted labels) computing the SSIM.
# When the --augment flag is enabled, random data augmentations are applied to each image before scoring.
# Author: Antonio Scardace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--augment',            action='store_true')
    parser.add_argument('--dataset_images_dir', type=str, required=True)
    parser.add_argument('--testset_csv',        type=str, required=True)
    args = parser.parse_args()

    # Computes true labels (y_true) and predicted labels (y_pred).
    # For each row in the dataset, matches the real and synthetic image pairs to their score.

    dataset = pd.read_csv(args.testset_csv)
    scorer = DeepSsimEmbeddingScorer()
    calc_ssim = calculate_ssim_augmented if args.augment else calculate_ssim
 
    y_true = []
    y_pred = []
    scores = []

    monai.utils.misc.set_determinism(42)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    for _, row in tqdm(dataset.iterrows(), 'Pairing y_true and y_pred', len(dataset)):
        image_1_path = get_image_path(row['real_key'], args.dataset_images_dir)
        image_2_path = get_image_path(row['synth_key'], args.dataset_images_dir)
        score = calc_ssim(image_1_path, image_2_path, True)
        y_pred.append(scorer.classify(score))
        y_true.append(row['label'])
        scores.append([score])

    # Computes classification metrics and display results.
    # Metrics include precision and recall for each class, along with the weighted and the macro F1-score.

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    silhouette = silhouette_score(np.array(scores), np.array(y_true))

    print('Different [Precision, Recall] =', report['0']['precision'], report['0']['recall'])
    print('Similar [Precision, Recall] =', report['2']['precision'], report['2']['recall'])
    print('Duplicate [Precision, Recall] =', report['1']['precision'], report['1']['recall'])
    print('Weighted F1-Score =', report['weighted avg']['f1-score'])
    print('Macro F1-Score =', report['macro avg']['f1-score'])
    print('Silhouette Score =', silhouette)