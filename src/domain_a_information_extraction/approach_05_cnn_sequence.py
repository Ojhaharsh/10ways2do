"""
Approach 5: CNN for Sequence Information Extraction

Philosophy: Extract local patterns through convolutions.
- Fast parallel computation
- Good for local patterns
- Limited global context
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import re

from ..core.base_model import BaseApproach
from .approach_04_rnn_lstm import Vocabulary, IEDataset, collate_fn


class CNNTagger(nn.Module):
    """CNN for sequence labeling"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 num_filters: int, filter_sizes: List[int],
                 num_labels: int, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple conv layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs, padding=fs//2)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_labels)
    
    def forward(self, input_ids, lengths=None):
        # (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # (batch, embed_dim, seq_len) for conv1d
        embedded = embedded.transpose(1, 2)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            conv_outputs.append(conv_out)
        
        # Concatenate along channel dimension
        # (batch, num_filters * len(filter_sizes), seq_len)
        combined = torch.cat(conv_outputs, dim=1)
        
        # (batch, seq_len, num_filters * len(filter_sizes))
        combined = combined.transpose(1, 2)
        combined = self.dropout(combined)
        
        # (batch, seq_len, num_labels)
        logits = self.fc(combined)
        
        return logits


class CNNIE(BaseApproach):
    """CNN-based Information Extraction"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("CNN Sequence", config)
        self.vocab = Vocabulary()
        self.label2idx = {}
        self.idx2label = {}
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Config
        self.embedding_dim = config.get('embedding_dim', 128) if config else 128
        self.num_filters = config.get('num_filters', 128) if config else 128
        self.filter_sizes = config.get('filter_sizes', [3, 5, 7]) if config else [3, 5, 7]
        self.batch_size = config.get('batch_size', 32) if config else 32
        self.epochs = config.get('epochs', 10) if config else 10
        
        self.metrics.interpretability_score = 0.4  # Filter patterns somewhat interpretable
        self.metrics.maintenance_complexity = 0.5
    
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
        """Train CNN model"""
        
        self.vocab.build(X_train)
        self._build_label_vocab(y_train)
        
        train_dataset = IEDataset(X_train, y_train, self.vocab, self.label2idx)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, collate_fn=collate_fn)
        
        self.model = CNNTagger(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            num_filters=self.num_filters,
            filter_sizes=self.filter_sizes,
            num_labels=len(self.label2idx)
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
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
                optimizer.step()
        
        self.is_trained = True
    
    def predict(self, X: List[str]) -> List[Dict[str, Any]]:
        """Predict using CNN"""
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
        
        # Post-process years_experience
        if 'years_experience' in entities:
            try:
                entities['years_experience'] = int(re.search(r'\d+', entities['years_experience']).group())
            except:
                entities['years_experience'] = None
        
        return entities
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Extract local n-gram patterns through parallel convolutions',
            'inductive_bias': 'Local context is sufficient, position-invariant patterns',
            'strengths': 'Fast training and inference, good for local patterns, parallelizable',
            'weaknesses': 'Limited long-range context, fixed receptive field',
            'best_for': 'Short to medium sequences, when speed is critical'
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