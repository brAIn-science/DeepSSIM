import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

from src.utils.plot import PlotHistogram

# This script generates a histogram to visualize the performance of the scoring metric.
# The histogram colors the bins based on the true labels.
# Threshold lines are also drawn to indicate the selected cut-off points.
# It helps assess the accuracy of the scoring method and identify cases where it may be less reliable.
# Author: Antonio Scardace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--low_threshold',      type=float, required=True)
    parser.add_argument('--upper_threshold',    type=float, required=True)
    parser.add_argument('--exp_title',          type=str, required=True)
    parser.add_argument('--output_path',        type=str, required=True)
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

    # Computes label-score pairs by matching dataset entries to indices.  
    # This loop extracts scores for each real-synthetic image pair of the dataset.

    results = []

    for i, row in tqdm(dataset.iterrows(), 'Matching labels and scores', len(dataset)):
        ridx, sidx = real_indices.index(row['real_key']), synth_indices.index(row['synth_key'])
        results.append([row['label'], score_matrix[ridx, sidx]])

    # Reorders labels to better distinguish between different, duplicate, and similar images.
    # Generates and saves the histogram with the specified thresholds in the given path.

    thresholds = {args.low_threshold, args.upper_threshold}
    custom_order = [0, 1, 2]
    df = pd.DataFrame(results, columns=['label', 'score'])
    plotter = PlotHistogram(df, args.exp_title, thresholds, args.output_path)
    plotter.save_hist(custom_order)