"""
Approach 8: Transformer-Based Recommendations

Philosophy: Use attention for sequential recommendations.
- Self-attention over interaction history
- Captures complex patterns
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import math

from ..core.base_model import BaseApproach


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation."""
    
    def __init__(self, n_items, embed_dim=64, n_heads=2, n_layers=2, max_len=50):
        super().__init__()
        
        self.item_embedding = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, n_heads, embed_dim * 4, 
            dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.output = nn.Linear(embed_dim, n_items)
        self.max_len = max_len
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # Embeddings
        item_emb = self.item_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        
        x = item_emb + pos_emb
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Padding mask
        padding_mask = (x.sum(dim=-1) == 0)
        
        x = self.transformer(x, mask=mask, src_key_padding_mask=padding_mask)
        
        return self.output(x[:, -1, :])


class TransformerRecommender(BaseApproach):
    """Transformer-based sequential recommender (SASRec)."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Transformer (SASRec)", config)
        
        self.embed_dim = config.get('embed_dim', 64) if config else 64
        self.n_heads = config.get('n_heads', 2) if config else 2
        self.n_layers = config.get('n_layers', 2) if config else 2
        self.max_len = config.get('max_len', 50) if config else 50
        self.epochs = config.get('epochs', 30) if config else 30
        self.batch_size = config.get('batch_size', 64) if config else 64
        
        self.model = None
        self.n_items = 0
        self.user_sequences = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.metrics.interpretability_score = 0.4
        self.metrics.maintenance_complexity = 0.7
    
    def train(self, train_matrix: np.ndarray, train_interactions: List = None,
              X_val: Any = None, y_val: Any = None) -> None:
        
        self.n_items = train_matrix.shape[1]
        
        # Build sequences
        if train_interactions:
            for u, i, _ in train_interactions:
                if u not in self.user_sequences:
                    self.user_sequences[u] = []
                self.user_sequences[u].append(i + 1)  # +1 for padding
        else:
            for u in range(train_matrix.shape[0]):
                items = np.where(train_matrix[u] > 0)[0]
                if len(items) > 0:
                    self.user_sequences[u] = (items + 1).tolist()
        
        sequences = [seq for seq in self.user_sequences.values() if len(seq) >= 2]
        
        if not sequences:
            self.is_trained = True
            return
        
        self.model = SASRec(
            self.n_items, self.embed_dim, self.n_heads, 
            self.n_layers, self.max_len
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            np.random.shuffle(sequences)
            
            for i in range(0, len(sequences), self.batch_size):
                batch_seqs = sequences[i:i + self.batch_size]
                
                # Pad sequences
                max_len = min(self.max_len, max(len(s) for s in batch_seqs))
                inputs = torch.zeros(len(batch_seqs), max_len - 1, dtype=torch.long)
                targets = torch.zeros(len(batch_seqs), dtype=torch.long)
                
                for j, seq in enumerate(batch_seqs):
                    seq = seq[-max_len:]
                    inputs[j, -len(seq)+1:] = torch.tensor(seq[:-1])
                    targets[j] = seq[-1] - 1  # Convert back to 0-indexed
                
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
        
        seq = self.user_sequences[user_id][-self.max_len+1:]
        
        self.model.eval()
        with torch.no_grad():
            input_seq = torch.zeros(1, len(seq), dtype=torch.long).to(self.device)
            input_seq[0] = torch.tensor(seq)
            logits = self.model(input_seq)[0].cpu().numpy()
        
        # Exclude seen
        for item in self.user_sequences[user_id]:
            logits[item - 1] = -np.inf
        
        if exclude_items:
            for item in exclude_items:
                logits[item] = -np.inf
        
        return np.argsort(-logits)[:k].tolist()
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Use self-attention to model complex sequential patterns',
            'inductive_bias': 'Any position can attend to any other; patterns can be non-local',
            'strengths': 'Captures long-range dependencies, parallel training',
            'weaknesses': 'Data hungry, computationally expensive',
            'best_for': 'Complex sequential patterns, sufficient training data'
        }
    
    def get_model_size(self) -> float:
        if self.model is None:
            return 0.0
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: Any, y_test: Any,
                               y_pred: Any, n_cases: int = 10) -> List[Dict]:
        return []