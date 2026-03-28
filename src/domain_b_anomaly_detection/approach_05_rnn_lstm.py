"""
Approach 5: RNN/LSTM for Sequence Anomaly Detection

Philosophy: Model temporal patterns; anomalies break patterns.
- Predict next value; high error = anomaly
- Capture temporal dependencies
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

from ..core.base_model import BaseApproach


class SequenceDataset(Dataset):
    """Dataset for sequence prediction."""
    
    def __init__(self, X: np.ndarray, seq_len: int):
        self.X = torch.FloatTensor(X)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.X) - self.seq_len
    
    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.seq_len],
            self.X[idx + self.seq_len]
        )


class LSTMPredictor(nn.Module):
    """LSTM for next-step prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class LSTMAnomalyDetector(BaseApproach):
    """LSTM-based anomaly detector using prediction error."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("RNN/LSTM", config)
        
        self.seq_len = config.get('seq_len', 10) if config else 10
        self.hidden_dim = config.get('hidden_dim', 64) if config else 64
        self.num_layers = config.get('num_layers', 2) if config else 2
        self.epochs = config.get('epochs', 30) if config else 30
        self.batch_size = config.get('batch_size', 32) if config else 32
        self.threshold_percentile = config.get('threshold_percentile', 95) if config else 95
        
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.metrics.interpretability_score = 0.3
        self.metrics.maintenance_complexity = 0.6
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        
        X_scaled = self.scaler.fit_transform(X_train)
        input_dim = X_scaled.shape[1]
        
        self.model = LSTMPredictor(
            input_dim, self.hidden_dim, self.num_layers
        ).to(self.device)
        
        dataset = SequenceDataset(X_scaled, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for seq, target in loader:
                seq, target = seq.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(seq)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
        
        # Compute threshold
        errors = self._compute_errors(X_scaled)
        self.threshold = np.percentile(errors, self.threshold_percentile)
        
        self.is_trained = True
    
    def _compute_errors(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute prediction errors for all points."""
        self.model.eval()
        errors = np.zeros(len(X_scaled))
        
        with torch.no_grad():
            for i in range(self.seq_len, len(X_scaled)):
                seq = torch.FloatTensor(X_scaled[i-self.seq_len:i]).unsqueeze(0).to(self.device)
                pred = self.model(seq).cpu().numpy()[0]
                actual = X_scaled[i]
                errors[i] = np.mean((pred - actual) ** 2)
        
        return errors
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        return (scores > 0.5).astype(int)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        errors = self._compute_errors(X_scaled)
        scores = errors / (self.threshold * 2)
        return np.clip(scores, 0, 1)
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Learn to predict next step; anomalies are unpredictable',
            'inductive_bias': 'Time series has learnable temporal patterns',
            'strengths': 'Captures temporal dependencies, context-aware',
            'weaknesses': 'Needs sufficient history, slow inference, training complexity',
            'best_for': 'Time series with strong temporal patterns'
        }
    
    def get_model_size(self) -> float:
        if self.model is None:
            return 0.0
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        wrong = y_test != y_pred
        scores = self.score(X_test)
        
        for idx in np.where(wrong)[0][:n_cases]:
            failures.append({
                'index': int(idx),
                'type': 'false_positive' if y_pred[idx] == 1 else 'false_negative',
                'prediction_error': float(scores[idx] * self.threshold * 2)
            })
        return failures