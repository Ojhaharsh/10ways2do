"""
Approach 4: Autoencoder for Anomaly Detection

Philosophy: Anomalies have high reconstruction error.
- Learn to compress and reconstruct normal data
- Anomalies don't fit the learned manifold
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from ..core.base_model import BaseApproach


class Autoencoder(nn.Module):
    """Simple autoencoder architecture."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed
    
    def encode(self, x):
        return self.encoder(x)


class AutoencoderDetector(BaseApproach):
    """Autoencoder-based anomaly detector."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Autoencoder", config)
        
        self.hidden_dims = config.get('hidden_dims', [32, 16]) if config else [32, 16]
        self.latent_dim = config.get('latent_dim', 8) if config else 8
        self.epochs = config.get('epochs', 50) if config else 50
        self.batch_size = config.get('batch_size', 64) if config else 64
        self.lr = config.get('lr', 1e-3) if config else 1e-3
        self.threshold_percentile = config.get('threshold_percentile', 95) if config else 95
        
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.metrics.interpretability_score = 0.4
        self.metrics.maintenance_complexity = 0.6
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        
        # Use normal samples only
        if y_train is not None:
            X_normal = X_train[y_train == 0]
        else:
            X_normal = X_train
        
        X_scaled = self.scaler.fit_transform(X_normal)
        
        # Create model
        input_dim = X_scaled.shape[1]
        self.model = Autoencoder(input_dim, self.hidden_dims, self.latent_dim).to(self.device)
        
        # DataLoader
        dataset = TensorDataset(torch.FloatTensor(X_scaled))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.model(x)
                loss = criterion(reconstructed, x)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # Compute threshold
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        
        self.threshold = np.percentile(errors, self.threshold_percentile)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        return (scores > 0.5).astype(int)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        
        scores = errors / (self.threshold * 2)
        return np.clip(scores, 0, 1)
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Learn compressed representation of normal data; anomalies reconstruct poorly',
            'inductive_bias': 'Normal data lies on a lower-dimensional manifold',
            'strengths': 'Learns complex patterns, unsupervised, flexible architecture',
            'weaknesses': 'Requires tuning, may reconstruct anomalies if similar to training',
            'best_for': 'Complex multivariate data, when anomalies differ structurally'
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
                'reconstruction_error': float(scores[idx] * self.threshold * 2),
                'threshold': float(self.threshold)
            })
        return failures


class VariationalAutoencoder(nn.Module):
    """VAE for anomaly detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class VAEDetector(BaseApproach):
    """Variational Autoencoder detector."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("VAE Detector", config)
        
        self.hidden_dim = config.get('hidden_dim', 32) if config else 32
        self.latent_dim = config.get('latent_dim', 8) if config else 8
        self.epochs = config.get('epochs', 50) if config else 50
        self.batch_size = config.get('batch_size', 64) if config else 64
        
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.metrics.interpretability_score = 0.3
        self.metrics.maintenance_complexity = 0.7
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        
        if y_train is not None:
            X_normal = X_train[y_train == 0]
        else:
            X_normal = X_train
        
        X_scaled = self.scaler.fit_transform(X_normal)
        input_dim = X_scaled.shape[1]
        
        self.model = VariationalAutoencoder(input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        
        dataset = TensorDataset(torch.FloatTensor(X_scaled))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch in loader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                recon, mu, log_var = self.model(x)
                
                recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kl_loss
                
                loss.backward()
                optimizer.step()
        
        # Compute threshold
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            recon, mu, log_var = self.model(X_tensor)
            errors = torch.mean((X_tensor - recon) ** 2, dim=1).cpu().numpy()
        
        self.threshold = np.percentile(errors, 95)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        return (scores > 0.5).astype(int)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            recon, _, _ = self.model(X_tensor)
            errors = torch.mean((X_tensor - recon) ** 2, dim=1).cpu().numpy()
        
        return np.clip(errors / (self.threshold * 2), 0, 1)
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Learn probabilistic latent space; anomalies have low likelihood',
            'inductive_bias': 'Normal data follows a learnable distribution in latent space',
            'strengths': 'Probabilistic interpretation, generates samples, regularized',
            'weaknesses': 'More complex training, may oversmooth',
            'best_for': 'When probabilistic anomaly scores are needed'
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