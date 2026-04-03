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

from importlib import import_module


_MODULES = {
    "core": ".core",
    "analysis": ".analysis",
    "domain_a_information_extraction": ".domain_a_information_extraction",
    "domain_b_anomaly_detection": ".domain_b_anomaly_detection",
    "domain_c_recommendation": ".domain_c_recommendation",
    "domain_d_time_series": ".domain_d_time_series",
    "domain_e_tabular_decisioning": ".domain_e_tabular_decisioning",
}


def __getattr__(name):
    """Lazily import top-level subpackages on first access."""
    if name in _MODULES:
        module = import_module(_MODULES[name], __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "core",
    "analysis",
    "domain_a_information_extraction",
    "domain_b_anomaly_detection",
    "domain_c_recommendation",
    "domain_d_time_series",
    "domain_e_tabular_decisioning",
]