import numpy as np

# This class computes average metrics over multiple samples.
# It provides attributes and methods to track loss and performance metrics over time.
# Author: Antonio Scardace

class AverageMetricsMeter:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.loss_values = []
        self.performance_sum = 0.0
        self.num_samples = 0

    def add(self, loss_value: float, performance_value: float, num: int) -> None:
        self.loss_values.extend([loss_value] * num)
        self.performance_sum += performance_value * num
        self.num_samples += num

    def loss_mean(self) -> float:
        return np.mean(self.loss_values) if self.loss_values else 0.0

    def loss_std(self) -> float:
        return np.std(self.loss_values, ddof=1) if len(self.loss_values) > 1 else 0.0

    def performance_mean(self) -> float:
        return self.performance_sum / self.num_samples if self.num_samples > 0 else 0.0