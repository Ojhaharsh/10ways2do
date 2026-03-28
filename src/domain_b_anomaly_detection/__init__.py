"""
Domain B: Anomaly Detection

Approaches:
1. Statistical (Z-score, IQR)
2. Distance-based (KNN, LOF)
3. Tree-based (Isolation Forest)
4. Autoencoder (Reconstruction error)
5. RNN/LSTM (Sequence anomaly)
6. Transformer (Attention-based)
7. Graph-based (For network data)
8. Ensemble (Combining methods)
9. Hybrid (Multi-stage detection)
10. Systems (Production monitoring)
"""

from .data_generator import create_anomaly_dataset
from .run_all import run_all_approaches