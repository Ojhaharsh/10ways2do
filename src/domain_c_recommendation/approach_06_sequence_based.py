"""
Approach 6: Sequence-Based Recommendations

Philosophy: Model user behavior as sequences.
- Order matters
- Session-based recommendations
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from ..core.base_model import BaseApproach


class SequenceDataset(Dataset):
    """Dataset for sequential recommendations."""
    
    def __init__(self, sequences, n_items, max_len=50):
        self.sequences = sequences
        self.n_items = n_items
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx][-self.max_len:]
        
        if len(seq) < 2:
            seq = [0] * 2  # Padding
        
        input_seq = seq[:-1]
        target = seq[-1]
        
        return torch.tensor(input_seq), torch.tensor(target)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return inputs, targets


class GRU4Rec(nn.Module):
    """GRU-based sequential recommender."""
    
    def __init__(self, n_items, embed_dim=64, hidden_dim=128, n_layers=1):
        super().__init__()
        
        self.embedding = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, n_items)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        logits = self.output(output[:, -1, :])
        return logits


class SequentialRecommender(BaseApproach):
    """GRU-based sequential recommender."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Sequence-Based (GRU4Rec)", config)
        
        self.embed_dim = config.get('embed_dim', 64) if config else 64
        self.hidden_dim = config.get('hidden_dim', 128) if config else 128
        self.epochs = config.get('epochs', 20) if config else 20
        self.batch_size = config.get('batch_size', 64) if config else 64
        
        self.model = None
        self.n_items = 0
        self.user_sequences = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.metrics.interpretability_score = 0.3
        self.metrics.maintenance_complexity = 0.6
    
    def train(self, train_matrix: np.ndarray, train_interactions: List = None,
              X_val: Any = None, y_val: Any = None) -> None:
        
        self.n_items = train_matrix.shape[1]
        
        # Build sequences from interactions
        if train_interactions:
            for u, i, r in sorted(train_interactions, key=lambda x: x[0]):
                if u not in self.user_sequences:
                    self.user_sequences[u] = []
                self.user_sequences[u].append(i)
        else:
            for u in range(train_matrix.shape[0]):
                items = np.where(train_matrix[u] > 0)[0].tolist()
                if items:
                    self.user_sequences[u] = items
        
        sequences = [seq for seq in self.user_sequences.values() if len(seq) >= 2]
        
        if not sequences:
            self.is_trained = True
            return
        
        self.model = GRU4Rec(self.n_items, self.embed_dim, self.hidden_dim).to(self.device)
        
        dataset = SequenceDataset(sequences, self.n_items)
        loader = DataLoader(dataset, batch_size=self.batch_size, 
                           shuffle=True, collate_fn=collate_fn)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
        
        self.is_trained = True
    
    def predict(self, user_ids: List[int], k: int = 10) -> List[List[int]]:
        return [self.recommend(u, k) for u in user_ids]
    
    def recommend(self, user_id: int, k: int = 10,
                  exclude_items: List[int] = None) -> List[int]:
        
        if self.model is None or user_id not in self.user_sequences:
            return list(range(k))
        
        seq = self.user_sequences[user_id][-50:]
        
        self.model.eval()
        with torch.no_grad():
            input_seq = torch.tensor([seq]).to(self.device)
            logits = self.model(input_seq)[0].cpu().numpy()
        
        # Exclude seen items
        for item in seq:
            logits[item] = -np.inf
        
        if exclude_items:
            for item in exclude_items:
                logits[item] = -np.inf
        
        return np.argsort(-logits)[:k].tolist()
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'User preferences evolve; recent actions predict next',
            'inductive_bias': 'Sequential patterns and temporal dynamics matter',
            'strengths': 'Captures dynamics, session-aware, handles evolving preferences',
            'weaknesses': 'Needs sequences, cold start, may overfit to recent',
            'best_for': 'Session-based recommendations, music/video streaming'
        }
    
    def get_model_size(self) -> float:
        if self.model is None:
            return 0.0
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: Any, y_test: Any,
                               y_pred: Any, n_cases: int = 10) -> List[Dict]:
        return []