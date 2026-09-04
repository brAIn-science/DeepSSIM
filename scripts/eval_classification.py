import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import roc_curve
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report

from src.factories.registry import MetricFactoryRegistry

# This script computes classification metrics to evaluate the performance of the scoring method.
# It aligns the ground-truth labels with predictions derived from the precomputed score matrix.
# Author: Antonio Scardace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric_name',        type=str, required=True, choices=['dar', 'ums_sscd', 'ums_l2', 'deepssim'])
    parser.add_argument('--matrix_path',        type=str, required=True, help='Path to the precomputed score matrix file.')
    parser.add_argument('--real_indices_path',  type=str, required=True, help='Path to the real image index metadata file.')
    parser.add_argument('--synth_indices_path', type=str, required=True, help='Path to the synthetic image index metadata file.')
    parser.add_argument('--testset_csv',        type=str, required=True, help='Path to the test set CSV file containing ground truth labels.')
    args = parser.parse_args()

    # Loads the evaluation dataset, the precomputed score matrix, and the index lists.  
    # These contain mappings of real and synthetic image pairs with scores.

    real_indices = np.load(args.real_indices_path)['data'].tolist()
    synth_indices = np.load(args.synth_indices_path)['data'].tolist()
    score_matrix = np.load(args.matrix_path)['data']
    dataset = pd.read_csv(args.testset_csv)

    # Extracts the specific score for each image pair and classifies it using the designated metric logic.
    # Builds the target and prediction arrays required for the scikit-learn metrics.

    scores = []
    y_true = []
    y_pred = []
    binary_y_true = []

    metric_factory = MetricFactoryRegistry.get_metric(args.metric_name)
    scorer = metric_factory.create_embedding_scorer()

    for i, row in tqdm(dataset.iterrows(), 'Matching labels and scores', len(dataset)):
        ridx, sidx = real_indices.index(row['real_key']), synth_indices.index(row['synth_key'])
        score = score_matrix[ridx, sidx]
        scores.append([score])
        y_pred.append(scorer.classify(score))
        y_true.append(row['final_label'])
        binary_y_true.append(int(row['final_label'] == 1))

    # Computes multi-class classification metrics (Precision, Recall, Macro F1).
    # Evaluates the clustering separation (Silhouette) and the TPR@5%FPR.

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    silhouette = silhouette_score(np.array(scores), np.array(y_true))

    score_values = np.array(scores).flatten()
    fpr, tpr, _ = roc_curve(binary_y_true, score_values)
    valid_idx_5 = np.where(fpr <= 0.05)[0]
    tpr_at_5_fpr = tpr[valid_idx_5[-1]] if len(valid_idx_5) > 0 else 0.0

    print('Different [Precision, Recall] =', report['0']['precision'], report['0']['recall'])
    print('Similar [Precision, Recall] =', report['2']['precision'], report['2']['recall'])
    print('Duplicate [Precision, Recall] =', report['1']['precision'], report['1']['recall'])
    print('Macro F1-Score =', report['macro avg']['f1-score'])
    print('TPR@5%FPR =', tpr_at_5_fpr)
    print('Silhouette Score =', silhouette)