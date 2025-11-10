from src.factories.dar import DarFactory
from src.factories.chen import ChenFactory
from src.factories.semdedup import SemDeDupFactory
from src.factories.deepssim import DeepSsimFactory
from src.factories.base import AbstractMetricFactory

# This class provides a registry-based factory for selecting metric-specific Concrete Factory.
# This design enhances modularity and makes it easy to extend with new metric-specific factories.
# Author: Antonio Scardace

class MetricFactoryRegistry:

    # Returns the appropriate handler class based on the metric name.
    # Raises an error if the requested metric is not available.

    @staticmethod
    def get_metric(name: str, augment: bool = False) -> AbstractMetricFactory:
        if name == 'dar': return DarFactory(augment)
        elif name == 'chen': return ChenFactory(augment)
        elif name == 'semdedup': return SemDeDupFactory(augment)
        elif name == 'deepssim': return DeepSsimFactory(augment)
        else: raise ValueError('Metric not available.')