import argparse
import pandas as pd

from statsmodels.stats.inter_rater import (
    aggregate_raters,
    fleiss_kappa
)

# This script computes Fleiss' Kappa to evaluate inter-rater reliability on the annotated test set.
# It aggregates the labels provided by multiple annotators to measure their categorical agreement.
# Author: Antonio Scardace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the CSV annotated dataset.')
    args = parser.parse_args()

    # Validates the presence of the required rater columns in the dataset.
    # Aggregates the categorical ratings and computes the overall Fleiss' Kappa score.

    df = pd.read_csv(args.dataset_path)
    rater_columns = ['r1_label', 'r2_label', 'r3_label']

    missing_columns = [col for col in rater_columns if col not in df.columns]
    if missing_columns:
        raise ValueError('The provided dataset does not contain the expected columns.')
    
    ratings = df[rater_columns]
    agg_ratings, categories = aggregate_raters(ratings)
    kappa_score = fleiss_kappa(agg_ratings)
    print("Fleiss' Kappa computed on", len(df), "samples is", kappa_score)