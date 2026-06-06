"""Resource estimation for fault-tolerant quantum computing.

Provides T-gate counting, surface code distance selection, magic state
distillation cost modeling, and end-to-end physical resource estimation.
"""

from quantumium.resource_estimation.analyzer import CircuitAnalyzer
from quantumium.resource_estimation.resources import LogicalResources, PhysicalEstimate
from quantumium.resource_estimation.estimator import ResourceEstimator
from quantumium.resource_estimation.factory import MagicStateFactory

__all__ = [
    "CircuitAnalyzer",
    "LogicalResources",
    "PhysicalEstimate",
    "ResourceEstimator",
    "MagicStateFactory",
]
