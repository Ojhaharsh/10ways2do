"""
Approach 4: RNN/LSTM for Information Extraction

Philosophy: Learn sequential patterns through recurrence.
- Captures order information
- Vanishing gradient issues
- Slower training than transformers
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import re
from collections import Counter
import pickle

from ..core.base_model import BaseApproach


class Vocabulary:
    """Simple vocabulary for tokenization"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_counts = Counter()
    
    def build(self, texts: List[str]) -> None:
        """Build vocabulary from texts"""
        for text in texts:
            tokens = self._tokenize(text)
            self.word_counts.update(tokens)
        
        # Add most common words
        for word, _ in self.word_counts.most_common(self.max_size - len(self.word2idx)):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    
    def encode(self, text: str) -> List[int]:
        tokens = self._tokenize(text)
        return [self.word2idx.get(t, 1) for t in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2word.get(i, '<UNK>') for i in indices]
    
    def __len__(self):
        return len(self.word2idx)


class IEDataset(Dataset):
    """Dataset for IE sequence labeling"""
    
    def __init__(self, texts: List[str], labels: List[Dict], 
                 vocab: Vocabulary, label2idx: Dict, max_len: int = 512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label2idx = label2idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_dict = self.labels[idx]
        
        # Encode text
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())[:self.max_len]
        token_ids = [self.vocab.word2idx.get(t, 1) for t in tokens]
        
        # Create BIO labels
        bio_labels = ['O'] * len(tokens)
        
        for field, value in label_dict.items():
            if field == 'skills' or value is None:
                continue
            
            value_tokens = re.findall(r'\b\w+\b|[^\w\s]', str(value).lower())
            
            # Find value in tokens
            for i in range(len(tokens) - len(value_tokens) + 1):
                if tokens[i:i+len(value_tokens)] == value_tokens:
                    bio_labels[i] = f'B-{field}'
                    for j in range(1, len(value_tokens)):
                        bio_labels[i+j] = f'I-{field}'
                    break
        
        label_ids = [self.label2idx.get(l, 0) for l in bio_labels]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'length': len(token_ids)
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'lengths': torch.tensor(lengths)
    }


class BiLSTMTagger(nn.Module):
    """Bidirectional LSTM for sequence labeling"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_labels: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
    
    def forward(self, input_ids, lengths=None):
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        if lengths is not None:
            packed = pack_padded_sequence(embedded, lengths.cpu(), 
                                          batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)
        
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        
        return logits


class RNNLSTMIE(BaseApproach):
    """RNN/LSTM approach for Information Extraction"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("RNN/LSTM", config)
        self.vocab = Vocabulary()
        self.label2idx = {}
        self.idx2label = {}
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Config
        self.embedding_dim = config.get('embedding_dim', 128) if config else 128
        self.hidden_dim = config.get('hidden_dim', 256) if config else 256
        self.num_layers = config.get('num_layers', 2) if config else 2
        self.batch_size = config.get('batch_size', 32) if config else 32
        self.epochs = config.get('epochs', 10) if config else 10
        self.lr = config.get('lr', 1e-3) if config else 1e-3
        
        self.metrics.interpretability_score = 0.3  # Hidden states are opaque
        self.metrics.maintenance_complexity = 0.6
    
    def _build_label_vocab(self, y_train: List[Dict]) -> None:
        """Build label vocabulary"""
        fields = ['name', 'email', 'phone', 'location', 'job_title',
                  'company', 'education', 'degree', 'years_experience']
        
        labels = ['O']
        for field in fields:
            labels.append(f'B-{field}')
            labels.append(f'I-{field}')
        
        self.label2idx = {l: i for i, l in enumerate(labels)}
        self.idx2label = {i: l for l, i in self.label2idx.items()}
    
    def train(self, X_train: List[str], y_train: List[Dict],
              X_val: Optional[List[str]] = None,
              y_val: Optional[List[Dict]] = None) -> None:
        """Train LSTM model"""
        
        # Build vocabularies
        self.vocab.build(X_train)
        self._build_label_vocab(y_train)
        
        # Create datasets
        train_dataset = IEDataset(X_train, y_train, self.vocab, self.label2idx)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, collate_fn=collate_fn)
        
        # Initialize model
        self.model = BiLSTMTagger(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_labels=len(self.label2idx),
            num_layers=self.num_layers
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths']
                
                optimizer.zero_grad()
                logits = self.model(input_ids, lengths)
                
                # Reshape for loss
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
        
        self.is_trained = True
    
    def predict(self, X: List[str]) -> List[Dict[str, Any]]:
        """Predict using trained LSTM"""
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for text in X:
                # Tokenize
                tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())[:512]
                token_ids = torch.tensor(
                    [[self.vocab.word2idx.get(t, 1) for t in tokens]], 
                    dtype=torch.long
                ).to(self.device)
                
                # Predict
                logits = self.model(token_ids)
                predictions = logits.argmax(dim=-1)[0].cpu().numpy()
                
                # Convert to entities
                extracted = self._extract_entities(tokens, predictions)
                results.append(extracted)
        
        return results
    
    def _extract_entities(self, tokens: List[str], predictions: np.ndarray) -> Dict:
        """Extract entities from BIO predictions"""
        entities = {}
        current_entity = None
        current_tokens = []
        
        for token, pred_idx in zip(tokens, predictions):
            label = self.idx2label.get(pred_idx, 'O')
            
            if label.startswith('B-'):
                # Save previous entity
                if current_entity and current_tokens:
                    value = ' '.join(current_tokens)
                    if current_entity == 'years_experience':
                        try:
                            value = int(re.search(r'\d+', value).group())
                        except:
                            value = None
                    entities[current_entity] = value
                
                # Start new entity
                current_entity = label[2:]
                current_tokens = [token]
            
            elif label.startswith('I-') and current_entity == label[2:]:
                current_tokens.append(token)
            
            else:
                # Save previous entity
                if current_entity and current_tokens:
                    value = ' '.join(current_tokens)
                    if current_entity == 'years_experience':
                        try:
                            value = int(re.search(r'\d+', value).group())
                        except:
                            value = None
                    entities[current_entity] = value
                
                current_entity = None
                current_tokens = []
        
        # Don't forget last entity
        if current_entity and current_tokens:
            value = ' '.join(current_tokens)
            entities[current_entity] = value
        
        return entities
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Learn sequential dependencies through recurrent hidden states',
            'inductive_bias': 'Recent context matters more (despite bidirectional), sequential processing',
            'strengths': 'Captures order, handles variable length, moderate memory',
            'weaknesses': 'Vanishing gradients, slow sequential training, limited long-range',
            'best_for': 'Moderate sequence lengths, when order matters, limited compute'
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