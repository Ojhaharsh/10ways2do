"""
Approach 6: Transformer from Scratch for Information Extraction

Philosophy: Attention-based global context modeling.
- Self-attention captures long-range dependencies
- Parallel computation
- Data hungry
"""

from typing import Dict, List, Any, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import re

from ..core.base_model import BaseApproach
from .approach_04_rnn_lstm import Vocabulary, IEDataset, collate_fn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(context)


class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class TransformerTagger(nn.Module):
    """Transformer encoder for sequence labeling"""
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, d_ff: int, num_labels: int,
                 max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, num_labels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, mask=None):
        # Embedding + positional encoding
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Create attention mask from padding
        if mask is None:
            mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Output projection
        logits = self.fc(self.dropout(x))
        
        return logits


class TransformerScratchIE(BaseApproach):
    """Transformer from scratch for IE"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Transformer (Scratch)", config)
        self.vocab = Vocabulary()
        self.label2idx = {}
        self.idx2label = {}
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Config
        self.d_model = config.get('d_model', 256) if config else 256
        self.num_heads = config.get('num_heads', 8) if config else 8
        self.num_layers = config.get('num_layers', 4) if config else 4
        self.d_ff = config.get('d_ff', 512) if config else 512
        self.batch_size = config.get('batch_size', 32) if config else 32
        self.epochs = config.get('epochs', 10) if config else 10
        
        self.metrics.interpretability_score = 0.5  # Attention weights somewhat interpretable
        self.metrics.maintenance_complexity = 0.6
    
    def _build_label_vocab(self, y_train: List[Dict]) -> None:
        fields = ['name', 'email', 'phone', 'location', 'job_title',
                  'company', 'education', 'degree', 'years_experience']
        labels = ['O']
        for field in fields:
            labels.extend([f'B-{field}', f'I-{field}'])
        self.label2idx = {l: i for i, l in enumerate(labels)}
        self.idx2label = {i: l for l, i in self.label2idx.items()}
    
    def train(self, X_train: List[str], y_train: List[Dict],
              X_val: Optional[List[str]] = None,
              y_val: Optional[List[Dict]] = None) -> None:
        """Train transformer"""
        
        self.vocab.build(X_train)
        self._build_label_vocab(y_train)
        
        train_dataset = IEDataset(X_train, y_train, self.vocab, self.label2idx)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, collate_fn=collate_fn)
        
        self.model = TransformerTagger(
            vocab_size=len(self.vocab),
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            num_labels=len(self.label2idx)
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(input_ids)
                
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
        
        self.is_trained = True
    
    def predict(self, X: List[str]) -> List[Dict[str, Any]]:
        """Predict using transformer"""
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for text in X:
                tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())[:512]
                token_ids = torch.tensor(
                    [[self.vocab.word2idx.get(t, 1) for t in tokens]],
                    dtype=torch.long
                ).to(self.device)
                
                logits = self.model(token_ids)
                predictions = logits.argmax(dim=-1)[0].cpu().numpy()
                
                extracted = self._extract_entities(tokens, predictions)
                results.append(extracted)
        
        return results
    
    def _extract_entities(self, tokens: List[str], predictions: np.ndarray) -> Dict:
        """Extract entities from predictions"""
        entities = {}
        current_entity = None
        current_tokens = []
        
        for token, pred_idx in zip(tokens, predictions):
            label = self.idx2label.get(pred_idx, 'O')
            
            if label.startswith('B-'):
                if current_entity and current_tokens:
                    entities[current_entity] = ' '.join(current_tokens)
                current_entity = label[2:]
                current_tokens = [token]
            elif label.startswith('I-') and current_entity == label[2:]:
                current_tokens.append(token)
            else:
                if current_entity and current_tokens:
                    entities[current_entity] = ' '.join(current_tokens)
                current_entity = None
                current_tokens = []
        
        if current_entity and current_tokens:
            entities[current_entity] = ' '.join(current_tokens)
        
        if 'years_experience' in entities:
            try:
                entities['years_experience'] = int(re.search(r'\d+', entities['years_experience']).group())
            except:
                entities['years_experience'] = None
        
        return entities
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Learn global context through self-attention over all positions',
            'inductive_bias': 'All positions can attend to all others, position encoded separately',
            'strengths': 'Captures long-range dependencies, parallel training, flexible',
            'weaknesses': 'O(n²) complexity, data hungry, requires careful tuning',
            'best_for': 'Sufficient training data, when global context matters'
        }
    
    def get_model_size(self) -> float:
        if self.model is None:
            return 0.0
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: List[str], y_test: List[Dict],
                               y_pred: List[Dict], n_cases: int = 10) -> List[Dict]:
        failures = []
        for i, (text, true, pred) in enumerate(zip(X_test, y_test, y_pred)):
            mismatches = {f: {'expected': true.get(f), 'predicted': pred.get(f)}
                         for f in true if f != 'skills' and
                         str(true.get(f, '')).lower() != str(pred.get(f, '')).lower()}
            if mismatches and len(failures) < n_cases:
                failures.append({'index': i, 'text_preview': text[:200], 'mismatches': mismatches})
        return failures