"""
Approach 6: Transformer for Anomaly Detection

Philosophy: Use attention to model complex dependencies.
- Self-attention captures global patterns
- Can model non-local anomalies
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from ..core.base_model import BaseApproach


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerAnomaly(nn.Module):
    """Transformer for sequence anomaly detection."""
    
    def __init__(self, input_dim: int, d_model: int, nhead: int, 
                 num_layers: int, seq_len: int):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_proj = nn.Linear(d_model, input_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.output_proj(x)


class TransformerAnomalyDetector(BaseApproach):
    """Transformer-based anomaly detector."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Transformer", config)
        
        self.seq_len = config.get('seq_len', 20) if config else 20
        self.d_model = config.get('d_model', 64) if config else 64
        self.nhead = config.get('nhead', 4) if config else 4
        self.num_layers = config.get('num_layers', 2) if config else 2
        self.epochs = config.get('epochs', 30) if config else 30
        self.batch_size = config.get('batch_size', 32) if config else 32
        
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.metrics.interpretability_score = 0.4
        self.metrics.maintenance_complexity = 0.7
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create overlapping sequences."""
        sequences = []
        for i in range(len(X) - self.seq_len + 1):
            sequences.append(X[i:i + self.seq_len])
        return np.array(sequences)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        
        if y_train is not None:
            X_normal = X_train[y_train == 0]
        else:
            X_normal = X_train
        
        X_scaled = self.scaler.fit_transform(X_normal)
        sequences = self._create_sequences(X_scaled)
        
        input_dim = X_scaled.shape[1]
        self.model = TransformerAnomaly(
            input_dim, self.d_model, self.nhead, self.num_layers, self.seq_len
        ).to(self.device)
        
        dataset = TensorDataset(torch.FloatTensor(sequences))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch in loader:
                x = batch[0].to(self.device)
                
                # Mask last element and predict
                input_seq = x.clone()
                input_seq[:, -1, :] = 0
                
                optimizer.zero_grad()
                output = self.model(input_seq)
                loss = criterion(output[:, -1, :], x[:, -1, :])
                loss.backward()
                optimizer.step()
        
        # Compute threshold
        self.model.eval()
        errors = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                input_seq = x.clone()
                input_seq[:, -1, :] = 0
                output = self.model(input_seq)
                err = torch.mean((output[:, -1, :] - x[:, -1, :]) ** 2, dim=1)
                errors.extend(err.cpu().numpy())
        
        self.threshold = np.percentile(errors, 95)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        return (scores > 0.5).astype(int)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        scores = np.zeros(len(X))
        
        self.model.eval()
        with torch.no_grad():
            for i in range(self.seq_len - 1, len(X)):
                seq = torch.FloatTensor(X_scaled[i-self.seq_len+1:i+1]).unsqueeze(0).to(self.device)
                input_seq = seq.clone()
                input_seq[:, -1, :] = 0
                
                output = self.model(input_seq)
                err = torch.mean((output[:, -1, :] - seq[:, -1, :]) ** 2).item()
                scores[i] = err
        
        return np.clip(scores / (self.threshold * 2), 0, 1)
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Use self-attention to model complex temporal dependencies',
            'inductive_bias': 'All positions can attend to all others; long-range patterns matter',
            'strengths': 'Captures long-range dependencies, parallel training',
            'weaknesses': 'Quadratic complexity, needs more data',
            'best_for': 'Long sequences with complex patterns'
        }
    
    def get_model_size(self) -> float:
        if self.model is None:
            return 0.0
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        wrong = y_test != y_pred
        for idx in np.where(wrong)[0][:n_cases]:
            failures.append({
                'index': int(idx),
                'type': 'false_positive' if y_pred[idx] == 1 else 'false_negative'
            })
        return failures