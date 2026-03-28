"""
Approach 7: Pretrained Transformer (BERT) for Information Extraction

Philosophy: Transfer learning from massive pretraining.
- Leverages pretrained knowledge
- Strong out-of-the-box performance
- Fine-tuning vs feature extraction trade-offs
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re

try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from ..core.base_model import BaseApproach


class BERTIEDataset(Dataset):
    """Dataset for BERT-based IE"""
    
    def __init__(self, texts: List[str], labels: List[Dict],
                 tokenizer, label2idx: Dict, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_dict = self.labels[idx]
        
        # Tokenize with BERT tokenizer
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        offset_mapping = encoding['offset_mapping'].squeeze()
        
        # Create labels aligned with tokenization
        labels = torch.zeros(self.max_len, dtype=torch.long)
        
        for field, value in label_dict.items():
            if field == 'skills' or value is None:
                continue
            
            value_str = str(value)
            start = text.find(value_str)
            if start == -1:
                continue
            end = start + len(value_str)
            
            # Find token positions
            for i, (token_start, token_end) in enumerate(offset_mapping.tolist()):
                if token_start == token_end:  # Special token
                    continue
                if token_start >= start and token_end <= end:
                    if token_start == start:
                        labels[i] = self.label2idx.get(f'B-{field}', 0)
                    else:
                        labels[i] = self.label2idx.get(f'I-{field}', 0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class PretrainedTransformerIE(BaseApproach):
    """BERT-based Information Extraction"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Pretrained Transformer (BERT)", config)
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required")
        
        self.model_name = config.get('model_name', 'bert-base-uncased') if config else 'bert-base-uncased'
        self.tokenizer = None
        self.model = None
        self.label2idx = {}
        self.idx2label = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.batch_size = config.get('batch_size', 16) if config else 16
        self.epochs = config.get('epochs', 3) if config else 3
        self.lr = config.get('lr', 2e-5) if config else 2e-5
        
        self.metrics.interpretability_score = 0.4
        self.metrics.maintenance_complexity = 0.5
    
    def _build_label_vocab(self) -> None:
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
        """Fine-tune BERT for IE"""
        
        self._build_label_vocab()
        
        # Load pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2idx)
        ).to(self.device)
        
        # Create dataset
        train_dataset = BERTIEDataset(X_train, y_train, self.tokenizer, self.label2idx)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        # Training
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
        
        self.is_trained = True
    
    def predict(self, X: List[str]) -> List[Dict[str, Any]]:
        """Predict using fine-tuned BERT"""
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for text in X:
                encoding = self.tokenizer(
                    text,
                    max_length=256,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    return_offsets_mapping=True
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                offset_mapping = encoding['offset_mapping'].squeeze().tolist()
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = outputs.logits.argmax(dim=-1)[0].cpu().numpy()
                
                # Extract entities based on offset mapping
                extracted = self._extract_entities(text, predictions, offset_mapping)
                results.append(extracted)
        
        return results
    
    def _extract_entities(self, text: str, predictions: np.ndarray,
                          offset_mapping: List) -> Dict:
        """Extract entities using offset mapping"""
        entities = {}
        current_entity = None
        current_start = None
        current_end = None
        
        for i, (pred_idx, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == end:  # Special token
                continue
            
            label = self.idx2label.get(pred_idx, 'O')
            
            if label.startswith('B-'):
                if current_entity and current_start is not None:
                    entities[current_entity] = text[current_start:current_end]
                current_entity = label[2:]
                current_start = start
                current_end = end
            elif label.startswith('I-') and current_entity == label[2:]:
                current_end = end
            else:
                if current_entity and current_start is not None:
                    entities[current_entity] = text[current_start:current_end]
                current_entity = None
                current_start = None
                current_end = None
        
        if current_entity and current_start is not None:
            entities[current_entity] = text[current_start:current_end]
        
        # Post-process
        if 'years_experience' in entities:
            try:
                entities['years_experience'] = int(re.search(r'\d+', entities['years_experience']).group())
            except:
                entities['years_experience'] = None
        
        return entities
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Transfer knowledge from massive pretraining to specific task',
            'inductive_bias': 'Language understanding from pretraining generalizes to extraction',
            'strengths': 'Strong performance with little data, captures semantics, state-of-the-art',
            'weaknesses': 'Large model size, slow inference, may overfit on small data',
            'best_for': 'When accuracy is paramount, sufficient compute available'
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