import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report

from src.factories.registry import MetricFactoryRegistry

# This script computes classification metrics to evaluate the scoring method.
# It builds y_true (ground truth labels) from the dataset and y_pred (predicted labels) from the score matrix.
# Author: Antonio Scardace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric_name',        type=str, required=True, choices=['dar', 'chen', 'semdedup', 'deepssim'])
    parser.add_argument('--matrix_path',        type=str, required=True)
    parser.add_argument('--real_indices_path',  type=str, required=True)
    parser.add_argument('--synth_indices_path', type=str, required=True)
    parser.add_argument('--testset_csv',        type=str, required=True)
    args = parser.parse_args()

    # Loads the test set, the score matrix, and the index lists.  
    # These contain mappings of real and synthetic image pairs with scores.

    real_indices = np.load(args.real_indices_path)['data'].tolist()
    synth_indices = np.load(args.synth_indices_path)['data'].tolist()
    score_matrix = np.load(args.matrix_path)['data']
    dataset = pd.read_csv(args.testset_csv)

    # Computes true labels (y_true) and predicted labels (y_pred).
    # For each row in the dataset, matches the real and synthetic image pairs to their score.
 
    y_true = []
    y_pred = []
    scores = []

    metric_factory = MetricFactoryRegistry.get_metric(args.metric_name)
    scorer = metric_factory.create_embedding_scorer()

    for i, row in tqdm(dataset.iterrows(), 'Pairing y_true and y_pred', len(dataset)):
        ridx, sidx = real_indices.index(row['real_key']), synth_indices.index(row['synth_key'])
        score = score_matrix[ridx, sidx]
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