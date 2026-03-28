"""
Approach 7: Graph Neural Networks for Recommendations

Philosophy: Model user-item interactions as a graph.
- Message passing between nodes
- Captures higher-order relationships
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.base_model import BaseApproach


class LightGCNLayer(nn.Module):
    """Single LightGCN layer."""
    
    def forward(self, x, edge_index, edge_weight=None):
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Aggregate
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col] * norm.unsqueeze(-1))
        
        return out


class LightGCN(nn.Module):
    """LightGCN model."""
    
    def __init__(self, n_users, n_items, embed_dim=64, n_layers=3):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        
        self.layers = nn.ModuleList([LightGCNLayer() for _ in range(n_layers)])
        
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
    
    def forward(self, edge_index):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        all_embeddings = [x]
        for layer in self.layers:
            x = layer(x, edge_index)
            all_embeddings.append(x)
        
        # Average all layers
        final_embedding = torch.stack(all_embeddings, dim=0).mean(dim=0)
        
        user_embed = final_embedding[:self.n_users]
        item_embed = final_embedding[self.n_users:]
        
        return user_embed, item_embed


class GraphNeuralRecommender(BaseApproach):
    """Graph Neural Network recommender using LightGCN."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Graph Neural Network (LightGCN)", config)
        
        self.embed_dim = config.get('embed_dim', 64) if config else 64
        self.n_layers = config.get('n_layers', 3) if config else 3
        self.epochs = config.get('epochs', 50) if config else 50
        self.batch_size = config.get('batch_size', 1024) if config else 1024
        self.lr = config.get('lr', 1e-3) if config else 1e-3
        
        self.model = None
        self.edge_index = None
        self.n_users = 0
        self.n_items = 0
        self.train_matrix = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.metrics.interpretability_score = 0.4
        self.metrics.maintenance_complexity = 0.7
    
    def _build_graph(self, train_matrix):
        """Build bipartite graph from interaction matrix."""
        users, items = np.where(train_matrix > 0)
        
        # User->Item edges
        src = users
        dst = items + self.n_users  # Offset item indices
        
        # Item->User edges (bidirectional)
        src_rev = items + self.n_users
        dst_rev = users
        
        edge_index = torch.tensor(
            np.stack([
                np.concatenate([src, src_rev]),
                np.concatenate([dst, dst_rev])
            ]),
            dtype=torch.long
        )
        
        return edge_index
    
    def train(self, train_matrix: np.ndarray, train_interactions: List = None,
              X_val: Any = None, y_val: Any = None) -> None:
        
        self.train_matrix = train_matrix
        self.n_users, self.n_items = train_matrix.shape
        
        self.edge_index = self._build_graph(train_matrix).to(self.device)
        
        self.model = LightGCN(
            self.n_users, self.n_items,
            self.embed_dim, self.n_layers
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Get positive pairs
        pos_users, pos_items = np.where(train_matrix > 0)
        
        self.model.train()
        for epoch in range(self.epochs):
            # Sample batch
            idx = np.random.permutation(len(pos_users))[:self.batch_size]
            batch_users = pos_users[idx]
            batch_pos_items = pos_items[idx]
            
            # Sample negatives
            batch_neg_items = np.random.randint(0, self.n_items, len(batch_users))
            
            users_t = torch.tensor(batch_users).to(self.device)
            pos_items_t = torch.tensor(batch_pos_items).to(self.device)
            neg_items_t = torch.tensor(batch_neg_items).to(self.device)
            
            optimizer.zero_grad()
            
            user_embed, item_embed = self.model(self.edge_index)
            
            user_e = user_embed[users_t]
            pos_item_e = item_embed[pos_items_t]
            neg_item_e = item_embed[neg_items_t]
            
            pos_scores = (user_e * pos_item_e).sum(dim=-1)
            neg_scores = (user_e * neg_item_e).sum(dim=-1)
            
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            
            loss.backward()
            optimizer.step()
        
        self.is_trained = True
    
    def predict(self, user_ids: List[int], k: int = 10) -> List[List[int]]:
        return [self.recommend(u, k) for u in user_ids]
    
    def recommend(self, user_id: int, k: int = 10,
                  exclude_items: List[int] = None) -> List[int]:
        
        self.model.eval()
        with torch.no_grad():
            user_embed, item_embed = self.model(self.edge_index)
            user_e = user_embed[user_id]
            scores = (user_e * item_embed).sum(dim=-1).cpu().numpy()
        
        already_rated = self.train_matrix[user_id] > 0
        scores[already_rated] = -np.inf
        
        if exclude_items:
            for item in exclude_items:
                scores[item] = -np.inf
        
        return np.argsort(-scores)[:k].tolist()
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Propagate information through user-item interaction graph',
            'inductive_bias': 'Higher-order connections carry useful signal',
            'strengths': 'Captures graph structure, handles sparse data well',
            'weaknesses': 'Computationally expensive, memory intensive',
            'best_for': 'When interaction graph structure is informative'
        }
    
    def get_model_size(self) -> float:
        if self.model is None:
            return 0.0
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: Any, y_test: Any,
                               y_pred: Any, n_cases: int = 10) -> List[Dict]:
        return []