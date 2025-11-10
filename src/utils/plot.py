import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# This class generates and saves a labeled histogram based on score data grouped by category labels.
# It is designed to visualize the score distribution and highlight decision thresholds.
# Author: Antonio Scardace

class PlotHistogram:

    def __init__(self, df: pd.DataFrame, title: str, thresholds: set[float], output_path: str) -> None:
        self.df = df
        self.title = title
        self.thresholds = thresholds
        self.output_path = output_path

    # Plots histogram bars for each label using a distinct color and label.
    # Ignores empty groups to avoid plotting errors.

    def __plot_histograms(self, palette: dict, labels: dict, order: list[int]) -> None:
        for label in order:
            color = palette[label]
            subset = self.df[self.df['label'] == label]
            bins = np.histogram_bin_edges(self.df['score'], bins=55)
            plt.hist(subset['score'], bins=bins, color=color, edgecolor='black', alpha=0.9, label=labels[label])

    # Adds vertical dashed lines at each threshold value.
    # These lines help to visually separate scoring ranges.

    def __add_threshold_lines(self) -> None:
        for t in self.thresholds:
            plt.axvline(t, color='black', ls='--', lw=1)

    # Finalizes the plot by setting labels, legend, and layout.
    # Then saves the figure to disk and closes it to free resources.

    def __make_save_plot(self) -> None:
        plt.xlabel(self.title, fontsize=19)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.legend(loc='upper left', fontsize=17, title='Legend', title_fontsize=17)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(self.output_path, bbox_inches='tight', pad_inches=0, facecolor='white')
        plt.close()

    # Main method to create and save the histogram.
    # Sets up colors and labels, draws the plot, and saves it.

    def save_hist(self, custom_order: list[int]) -> None:
        palette = {0: 'green', 1: 'red', 2: 'orange'}
        labels = {0: 'Different', 1: 'Duplicate', 2: 'Similar'}
        
        sns.set_theme(context='paper')
        plt.figure(figsize=(6, 5))
        self.__plot_histograms(palette, labels, custom_order)
        self.__add_threshold_lines()
        self.__make_save_plot()