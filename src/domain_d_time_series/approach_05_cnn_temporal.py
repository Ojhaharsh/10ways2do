"""
Approach 5: Temporal CNN for Time Series

Philosophy: Use convolutions to extract local patterns.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..core.base_model import BaseApproach


class TCN(nn.Module):
    """Temporal Convolutional Network."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 forecast_horizon: int, kernel_size: int = 3, num_layers: int = 4):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size,
                                   padding=padding, dilation=dilation))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim * forecast_horizon)
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
    
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        conv_out = self.conv_layers(x)
        
        # Take the last time step
        out = self.fc(conv_out[:, :, -1])
        return out.view(-1, self.forecast_horizon, self.output_dim)


class TCNForecaster(BaseApproach):
    """Temporal CNN forecaster."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Temporal CNN", config)
        
        self.hidden_dim = config.get('hidden_dim', 64) if config else 64
        self.num_layers = config.get('num_layers', 4) if config else 4
        self.epochs = config.get('epochs', 50) if config else 50
        self.batch_size = config.get('batch_size', 32) if config else 32
        
        self.model = None
        self.forecast_horizon = 24
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.metrics.interpretability_score = 0.4
        self.metrics.maintenance_complexity = 0.5
    
    def train(self, X_train: np.ndarray = None, y_train: np.ndarray = None,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              train_series: np.ndarray = None, forecast_horizon: int = 24) -> None:
        
        self.forecast_horizon = forecast_horizon
        
        input_dim = X_train.shape[2] if X_train.ndim == 3 else 1
        output_dim = y_train.shape[2] if y_train.ndim == 3 else 1
        
        self.model = TCN(
            input_dim, self.hidden_dim, output_dim,
            forecast_horizon, num_layers=self.num_layers
        ).to(self.device)
        
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
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
            'mental_model': 'Extract local patterns via convolutions, expand receptive field with dilation',
            'inductive_bias': 'Local patterns are building blocks, dilations capture multi-scale',
            'strengths': 'Parallel training, stable gradients, handles long sequences',
            'weaknesses': 'Fixed receptive field, less flexible than attention',
            'best_for': 'Long sequences, when local patterns are important'
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