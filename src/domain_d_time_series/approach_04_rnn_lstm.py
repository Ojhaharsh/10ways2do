"""
Approach 4: RNN/LSTM for Time Series

Philosophy: Learn temporal dependencies through recurrence.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..core.base_model import BaseApproach


class LSTMModel(nn.Module):
    """LSTM for sequence-to-sequence forecasting."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 forecast_horizon: int, num_layers: int = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim * forecast_horizon)
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.view(-1, self.forecast_horizon, self.output_dim)


class LSTMForecaster(BaseApproach):
    """LSTM-based time series forecaster."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("RNN/LSTM", config)
        
        self.hidden_dim = config.get('hidden_dim', 64) if config else 64
        self.num_layers = config.get('num_layers', 2) if config else 2
        self.epochs = config.get('epochs', 50) if config else 50
        self.batch_size = config.get('batch_size', 32) if config else 32
        self.lr = config.get('lr', 1e-3) if config else 1e-3
        
        self.model = None
        self.forecast_horizon = 24
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.metrics.interpretability_score = 0.3
        self.metrics.maintenance_complexity = 0.6
    
    def train(self, X_train: np.ndarray = None, y_train: np.ndarray = None,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              train_series: np.ndarray = None, forecast_horizon: int = 24) -> None:
        
        self.forecast_horizon = forecast_horizon
        
        if X_train is None:
            raise ValueError("Need X_train")
        
        input_dim = X_train.shape[2] if X_train.ndim == 3 else 1
        output_dim = y_train.shape[2] if y_train.ndim == 3 else 1
        
        self.model = LSTMModel(
            input_dim, self.hidden_dim, output_dim,
            forecast_horizon, self.num_layers
        ).to(self.device)
        
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Learn temporal patterns through recurrent hidden states',
            'inductive_bias': 'Sequential processing, hidden state as memory',
            'strengths': 'Captures temporal dependencies, handles variable length',
            'weaknesses': 'Vanishing gradients, slow training, hard to parallelize',
            'best_for': 'Clear sequential patterns, moderate sequence lengths'
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