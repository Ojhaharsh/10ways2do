"""
ML Philosophy Benchmark

A comprehensive benchmark comparing 10 fundamentally different ML approaches
across 4 real-world domains.

Domains:
- A: Information Extraction
- B: Anomaly Detection
- C: Recommendation
- D: Time Series Forecasting

Each domain implements 10 approaches representing different ML philosophies.
"""

__version__ = "1.0.0"
__author__ = "ML Benchmark Team"

from . import core
from . import analysis
from . import domain_a_information_extraction
from . import domain_b_anomaly_detection
from . import domain_c_recommendation
from . import domain_d_time_series

__all__ = [
    "core",
    "analysis",
    "domain_a_information_extraction",
    "domain_b_anomaly_detection",
    "domain_c_recommendation",
    "domain_d_time_series",
]