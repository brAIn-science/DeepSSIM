from src.factories.dar import DarFactory
from src.factories.ums_l2 import UmsL2Factory
from src.factories.ums_sscd import UmsSscdFactory
from src.factories.deepssim import DeepSsimFactory
from src.factories.base import AbstractMetricFactory

# This class provides a registry-based factory for selecting metric-specific Concrete Factories.
# This design enhances modularity and makes it easy to extend the framework with new metric-specific factories.
# Author: Antonio Scardace

class MetricFactoryRegistry:

    # Returns the appropriate handler class based on the metric name.
    # Raises an error if the requested metric is not available.

    @staticmethod
    def get_metric(name: str, augment: bool = False) -> AbstractMetricFactory:
        if name == 'dar': return DarFactory(augment)
        elif name == 'ums_sscd': return UmsSscdFactory(augment)
        elif name == 'ums_l2': return UmsL2Factory(augment)
        elif name == 'deepssim': return DeepSsimFactory(augment)
        else: raise ValueError('Metric not available.')