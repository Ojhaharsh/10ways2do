"""
Approach 6: Transformer for Time Series

Philosophy: Use attention for global dependencies.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset

from ..core.base_model import BaseApproach


class TimeSeriesTransformer(nn.Module):
    """Transformer for time series forecasting."""
    
    def __init__(self, input_dim: int, d_model: int, nhead: int,
                 num_layers: int, forecast_horizon: int, output_dim: int):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        max_len = 500
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, 
                                                    dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Linear(d_model, output_dim * forecast_horizon)
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
    
    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pe[:, :x.size(1)]
        
        x = self.transformer(x)
        
        out = self.fc(x[:, -1, :])
        return out.view(-1, self.forecast_horizon, self.output_dim)


class TransformerForecaster(BaseApproach):
    """Transformer-based time series forecaster."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Transformer", config)
        
        self.d_model = config.get('d_model', 64) if config else 64
        self.nhead = config.get('nhead', 4) if config else 4
        self.num_layers = config.get('num_layers', 2) if config else 2
        self.epochs = config.get('epochs', 50) if config else 50
        self.batch_size = config.get('batch_size', 32) if config else 32
        
        self.model = None
        self.forecast_horizon = 24
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.metrics.interpretability_score = 0.4
        self.metrics.maintenance_complexity = 0.7
    
    def train(self, X_train: np.ndarray = None, y_train: np.ndarray = None,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              train_series: np.ndarray = None, forecast_horizon: int = 24) -> None:
        
        self.forecast_horizon = forecast_horizon
        
        input_dim = X_train.shape[2] if X_train.ndim == 3 else 1
        output_dim = y_train.shape[2] if y_train.ndim == 3 else 1
        
        self.model = TimeSeriesTransformer(
            input_dim, self.d_model, self.nhead,
            self.num_layers, forecast_horizon, output_dim
        ).to(self.device)
        
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Use self-attention to model dependencies at any distance',
            'inductive_bias': 'All positions can attend to all others equally',
            'strengths': 'Captures long-range dependencies, parallel training',
            'weaknesses': 'Quadratic complexity, needs more data, may overfit',
            'best_for': 'Long sequences with complex dependencies'
        }
    
    def get_model_size(self) -> float:
        if self.model is None:
            return 0.0
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        errors = np.mean((y_test - y_pred) ** 2, axis=(1, 2))
        worst_indices = np.argsort(-errors)[:n_cases]
        for idx in worst_indices:
            failures.append({'index': int(idx), 'mse': float(errors[idx])})
        return failures