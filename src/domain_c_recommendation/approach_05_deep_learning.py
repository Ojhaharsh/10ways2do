"""
Approach 5: Deep Learning Recommendations

Philosophy: Learn complex non-linear user-item interactions.
- Neural Collaborative Filtering
- Deep embeddings
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..core.base_model import BaseApproach


class InteractionDataset(Dataset):
    """Dataset for user-item interactions."""
    
    def __init__(self, interactions, n_items, n_negatives=4):
        self.interactions = interactions
        self.n_items = n_items
        self.n_negatives = n_negatives
        
        # Build positive set per user
        self.user_positives = {}
        for u, i, _ in interactions:
            if u not in self.user_positives:
                self.user_positives[u] = set()
            self.user_positives[u].add(i)
    
    def __len__(self):
        return len(self.interactions) * (1 + self.n_negatives)
    
    def __getitem__(self, idx):
        if idx < len(self.interactions):
            u, i, r = self.interactions[idx]
            return torch.tensor(u), torch.tensor(i), torch.tensor(1.0)
        else:
            # Negative sample
            orig_idx = idx % len(self.interactions)
            u, _, _ = self.interactions[orig_idx]
            
            # Sample negative item
            while True:
                neg_item = np.random.randint(self.n_items)
                if neg_item not in self.user_positives.get(u, set()):
                    break
            
            return torch.tensor(u), torch.tensor(neg_item), torch.tensor(0.0)


class NCF(nn.Module):
    """Neural Collaborative Filtering."""
    
    def __init__(self, n_users, n_items, embed_dim=64, hidden_dims=[128, 64]):
        super().__init__()
        
        # GMF path
        self.user_embed_gmf = nn.Embedding(n_users, embed_dim)
        self.item_embed_gmf = nn.Embedding(n_items, embed_dim)
        
        # MLP path
        self.user_embed_mlp = nn.Embedding(n_users, embed_dim)
        self.item_embed_mlp = nn.Embedding(n_items, embed_dim)
        
        mlp_layers = []
        input_dim = embed_dim * 2
        for dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Combine GMF and MLP
        self.output = nn.Linear(embed_dim + hidden_dims[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, user, item):
        # GMF
        user_gmf = self.user_embed_gmf(user)
        item_gmf = self.item_embed_gmf(item)
        gmf = user_gmf * item_gmf
        
        # MLP
        user_mlp = self.user_embed_mlp(user)
        item_mlp = self.item_embed_mlp(item)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp = self.mlp(mlp_input)
        
        # Combine
        combined = torch.cat([gmf, mlp], dim=-1)
        output = torch.sigmoid(self.output(combined))
        
        return output.squeeze()


class NCFRecommender(BaseApproach):
    """Neural Collaborative Filtering recommender."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Deep Learning (NCF)", config)
        
        self.embed_dim = config.get('embed_dim', 64) if config else 64
        self.hidden_dims = config.get('hidden_dims', [128, 64]) if config else [128, 64]
        self.epochs = config.get('epochs', 20) if config else 20
        self.batch_size = config.get('batch_size', 256) if config else 256
        self.lr = config.get('lr', 1e-3) if config else 1e-3
        
        self.model = None
        self.n_users = 0
        self.n_items = 0
        self.train_matrix = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.metrics.interpretability_score = 0.3
        self.metrics.maintenance_complexity = 0.7
    
    def train(self, train_matrix: np.ndarray, train_interactions: List = None,
              X_val: Any = None, y_val: Any = None) -> None:
        
        self.train_matrix = train_matrix
        self.n_users, self.n_items = train_matrix.shape
        
        if train_interactions is None:
            train_interactions = [
                (u, i, train_matrix[u, i])
                for u in range(self.n_users)
                for i in range(self.n_items)
                if train_matrix[u, i] > 0
            ]
        
        self.model = NCF(
            self.n_users, self.n_items,
            self.embed_dim, self.hidden_dims
        ).to(self.device)
        
        dataset = InteractionDataset(train_interactions, self.n_items)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for users, items, labels in loader:
                users = users.to(self.device)
                items = items.to(self.device)
                labels = labels.float().to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(users, items)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        self.is_trained = True
    
    def predict(self, user_ids: List[int], k: int = 10) -> List[List[int]]:
        return [self.recommend(u, k) for u in user_ids]
    
    def recommend(self, user_id: int, k: int = 10,
                  exclude_items: List[int] = None) -> List[int]:
        
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id] * self.n_items).to(self.device)
            item_tensor = torch.arange(self.n_items).to(self.device)
            
            scores = self.model(user_tensor, item_tensor).cpu().numpy()
        
        already_rated = self.train_matrix[user_id] > 0
        scores[already_rated] = -np.inf
        
        if exclude_items:
            for item in exclude_items:
                scores[item] = -np.inf
        
        return np.argsort(-scores)[:k].tolist()
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Learn non-linear user-item interactions with neural networks',
            'inductive_bias': 'Complex patterns require deep representations',
            'strengths': 'Captures non-linear patterns, flexible architecture',
            'weaknesses': 'Data hungry, expensive training, black box',
            'best_for': 'Large datasets with complex interaction patterns'
        }
    
    def get_model_size(self) -> float:
        if self.model is None:
            return 0.0
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: Any, y_test: Any,
                               y_pred: Any, n_cases: int = 10) -> List[Dict]:
        return []