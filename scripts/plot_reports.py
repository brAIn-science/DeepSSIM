import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

from src.utils.plot import PlotHistogram

# This script generates a stratified histogram to visualize the performance of the scoring metric.
# The distribution is grouped by ground-truth labels, with vertical lines marking the cut-off thresholds.
# It enables a visual assessment of class separation and overall metric reliability.
# Author: Antonio Scardace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_csv',        type=str,   required=True, help='Path to the test set CSV file containing annotated image pairs.')
    parser.add_argument('--real_indices_path',  type=str,   required=True, help='Path to the real image index metadata file.')
    parser.add_argument('--synth_indices_path', type=str,   required=True, help='Path to the synthetic image index metadata file.')
    parser.add_argument('--matrix_path',        type=str,   required=True, help='Path to the pre-computed score matrix file.')
    parser.add_argument('--output_path',        type=str,   required=True, help='Path to save the generated histogram image.')
    parser.add_argument('--exp_title',          type=str,   required=True, help='Title of the experiment to display on the plot.')
    parser.add_argument('--low_threshold',      type=float, required=True, help='Lower score threshold for classification.')
    parser.add_argument('--upper_threshold',    type=float, required=True, help='Upper score threshold for classification.')
    args = parser.parse_args()

    # Loads the evaluation set, the pre-computed score matrix, and the index lists.  
    # These contain mappings of real and synthetic image pairs with scores.  

    real_indices = np.load(args.real_indices_path)['data'].tolist()
    synth_indices = np.load(args.synth_indices_path)['data'].tolist()
    score_matrix = np.load(args.matrix_path)['data']
    dataset = pd.read_csv(args.testset_csv)

    # This loop extracts scores for each real-synthetic image pair of the dataset.
    # Applies a custom categorical order to logically separate different, similar, and duplicate classes.
    # Renders and saves the final thresholded histogram.

    results = []
    for i, row in tqdm(dataset.iterrows(), 'Matching labels and scores', len(dataset)):
        ridx, sidx = real_indices.index(row['real_key']), synth_indices.index(row['synth_key'])
        results.append([row['label'], score_matrix[ridx, sidx]])

    thresholds = {args.low_threshold, args.upper_threshold}
    custom_order = [0, 2, 1]
    df = pd.DataFrame(results, columns=['label', 'score'])
    plotter = PlotHistogram(df, args.exp_title, thresholds, args.output_path)
    plotter.save_hist(custom_order)