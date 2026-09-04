import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# This class generates and saves a labeled histogram based on score data grouped by category.
# It is designed to visualize the score distribution and highlight decision thresholds.
# Author: Antonio Scardace

class PlotHistogram:

    def __init__(self, df: pd.DataFrame, title: str, thresholds: set[float], output_path: str) -> None:
        self.df = df
        self.title = title
        self.thresholds = thresholds
        self.output_path = output_path

    # Plots histogram bars for each label using a distinct color and label.
    # Skips empty groups to avoid plotting errors.

    def __plot_histograms(self, palette: dict, labels: dict, order: list[int]) -> None:
        for label in order:
            color = palette[label]
            subset = self.df[self.df['final_label'] == label]
            bins = np.histogram_bin_edges(self.df['score'], bins=65)
            plt.hist(subset['score'], bins=bins, color=color, edgecolor='black', linewidth=0.25, alpha=0.9, label=labels[label])

    # Adds vertical dashed lines at each threshold value.
    # These lines help visually separate different scoring ranges.

    def __add_threshold_lines(self) -> None:
        for t in self.thresholds:
            plt.axvline(t, color='black', ls='--', lw=1)

    # Finalizes the plot by setting the axis labels, legend, and layout.
    # It then saves the figure to disk and closes it to free resources.

    def __make_save_plot(self) -> None:
        plt.xlabel(self.title, fontsize=24)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(self.output_path, bbox_inches='tight', pad_inches=0, facecolor='white')
        plt.close()

    # Main method for creating and saving the histogram.
    # It sets up the colors and labels, draws the plot, and saves it.

    def save_hist(self, custom_order: list[int]) -> None:
        palette = {0: 'green', 1: 'red', 2: 'orange'}
        labels = {0: 'Different', 1: 'Duplicate', 2: 'Similar'}
        
        sns.set_theme(context='paper')
        plt.figure(figsize=(6, 5))
        self.__plot_histograms(palette, labels, custom_order)
        self.__add_threshold_lines()
        self.__make_save_plot()