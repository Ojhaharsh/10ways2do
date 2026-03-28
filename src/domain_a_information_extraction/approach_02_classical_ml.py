"""
Approach 2: Classical ML for Information Extraction

Philosophy: Learn patterns from features, not raw data.
- Feature engineering is key
- Limited context window
- Interpretable feature importance
"""

import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

from ..core.base_model import BaseApproach, ModelMetrics


class ClassicalMLIE(BaseApproach):
    """
    Classical ML approach to IE using feature-based sequence labeling.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Classical ML (CRF-style)", config)
        self.models = {}  # One model per field
        self.vectorizers = {}
        self.label_encoders = {}
        
        self.metrics.interpretability_score = 0.8
        self.metrics.maintenance_complexity = 0.4
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b|[^\w\s]', text)
    
    def _extract_token_features(self, tokens: List[str], idx: int) -> Dict[str, Any]:
        """Extract features for a single token"""
        token = tokens[idx]
        
        features = {
            'token': token.lower(),
            'is_upper': token.isupper(),
            'is_title': token.istitle(),
            'is_digit': token.isdigit(),
            'is_alpha': token.isalpha(),
            'length': len(token),
            'has_at': '@' in token,
            'has_dot': '.' in token,
            'has_hyphen': '-' in token,
            'prefix_2': token[:2].lower() if len(token) >= 2 else token.lower(),
            'suffix_2': token[-2:].lower() if len(token) >= 2 else token.lower(),
            'prefix_3': token[:3].lower() if len(token) >= 3 else token.lower(),
            'suffix_3': token[-3:].lower() if len(token) >= 3 else token.lower(),
        }
        
        # Context features
        if idx > 0:
            features['prev_token'] = tokens[idx-1].lower()
            features['prev_is_title'] = tokens[idx-1].istitle()
        else:
            features['prev_token'] = '<START>'
            features['prev_is_title'] = False
            
        if idx < len(tokens) - 1:
            features['next_token'] = tokens[idx+1].lower()
            features['next_is_title'] = tokens[idx+1].istitle()
        else:
            features['next_token'] = '<END>'
            features['next_is_title'] = False
        
        # Position features
        features['position'] = idx / len(tokens)
        features['is_first'] = idx == 0
        features['is_last'] = idx == len(tokens) - 1
        
        return features
    
    def _features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dict to vector"""
        vector = []
        
        # Numeric features
        vector.append(1 if features['is_upper'] else 0)
        vector.append(1 if features['is_title'] else 0)
        vector.append(1 if features['is_digit'] else 0)
        vector.append(1 if features['is_alpha'] else 0)
        vector.append(features['length'] / 20)  # Normalize
        vector.append(1 if features['has_at'] else 0)
        vector.append(1 if features['has_dot'] else 0)
        vector.append(1 if features['has_hyphen'] else 0)
        vector.append(features['position'])
        vector.append(1 if features['is_first'] else 0)
        vector.append(1 if features['is_last'] else 0)
        vector.append(1 if features['prev_is_title'] else 0)
        vector.append(1 if features['next_is_title'] else 0)
        
        return np.array(vector)
    
    def _prepare_training_data(self, X_train: List[str], y_train: List[Dict], 
                                field: str) -> Tuple[List[str], List[int]]:
        """Prepare training data for a specific field"""
        contexts = []
        labels = []
        
        for text, entities in zip(X_train, y_train):
            value = entities.get(field)
            if value is None:
                continue
            
            value_str = str(value).lower()
            text_lower = text.lower()
            
            # Find value in text
            pos = text_lower.find(value_str)
            if pos != -1:
                # Extract context window
                start = max(0, pos - 50)
                end = min(len(text), pos + len(value_str) + 50)
                context = text[start:end]
                contexts.append(context)
                labels.append(1)
                
                # Add negative examples
                neg_start = (pos + len(value_str) + 10) % len(text)
                neg_end = min(len(text), neg_start + 100)
                if neg_end > neg_start:
                    neg_context = text[neg_start:neg_end]
                    contexts.append(neg_context)
                    labels.append(0)
        
        return contexts, labels
    
    def train(self, X_train: List[str], y_train: List[Dict],
              X_val: Optional[List[str]] = None,
              y_val: Optional[List[Dict]] = None) -> None:
        """Train separate models for each field"""
        
        fields = ['name', 'email', 'phone', 'location', 'job_title', 
                  'company', 'education', 'degree']
        
        for field in fields:
            contexts, labels = self._prepare_training_data(X_train, y_train, field)
            
            if len(contexts) < 10:
                continue
            
            # Create pipeline
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                analyzer='char_wb'
            )
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            
            # Train
            X_vec = vectorizer.fit_transform(contexts)
            model.fit(X_vec, labels)
            
            self.vectorizers[field] = vectorizer
            self.models[field] = model
        
        self.is_trained = True
    
    def _extract_candidates(self, text: str, field: str) -> List[Tuple[str, float]]:
        """Extract candidate values for a field"""
        candidates = []
        
        # Field-specific candidate extraction
        if field == 'email':
            pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                candidates.append(match.group())
        
        elif field == 'phone':
            patterns = [
                r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
                r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}'
            ]
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    candidates.append(match.group())
        
        elif field == 'years_experience':
            pattern = r'(\d+)\+?\s*years?'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                candidates.append(match.group(1))
        
        else:
            # General: extract noun phrases (simplified)
            words = text.split()
            for i in range(len(words)):
                for j in range(i+1, min(i+5, len(words)+1)):
                    candidate = ' '.join(words[i:j])
                    if len(candidate) > 2 and candidate[0].isupper():
                        candidates.append(candidate)
        
        return list(set(candidates))[:20]  # Limit candidates
    
    def predict(self, X: List[str]) -> List[Dict[str, Any]]:
        """Predict by ranking candidates"""
        results = []
        
        for text in X:
            extracted = {}
            
            for field in self.models.keys():
                candidates = self._extract_candidates(text, field)
                
                if not candidates:
                    extracted[field] = None
                    continue
                
                # Score each candidate
                best_score = -1
                best_candidate = None
                
                for candidate in candidates:
                    # Get context around candidate
                    pos = text.lower().find(candidate.lower())
                    if pos == -1:
                        continue
                    
                    start = max(0, pos - 50)
                    end = min(len(text), pos + len(candidate) + 50)
                    context = text[start:end]
                    
                    # Score
                    vec = self.vectorizers[field].transform([context])
                    proba = self.models[field].predict_proba(vec)[0][1]
                    
                    if proba > best_score:
                        best_score = proba
                        best_candidate = candidate
                
                extracted[field] = best_candidate if best_score > 0.5 else None
            
            # Post-process years_experience
            if extracted.get('years_experience'):
                try:
                    extracted['years_experience'] = int(extracted['years_experience'])
                except:
                    extracted['years_experience'] = None
            
            results.append(extracted)
        
        return results
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Learn to recognize field values through engineered features and context patterns',
            'inductive_bias': 'Assumes fields have consistent contextual patterns that can be captured by n-grams',
            'strengths': 'Works with moderate data, interpretable features, fast inference',
            'weaknesses': 'Requires feature engineering, limited context understanding, struggles with novel formats',
            'best_for': 'Structured documents with consistent formats, when interpretability matters'
        }
    
    def collect_failure_cases(self, X_test: List[str], y_test: List[Dict],
                               y_pred: List[Dict], n_cases: int = 10) -> List[Dict]:
        failures = []
        for i, (text, true, pred) in enumerate(zip(X_test, y_test, y_pred)):
            mismatches = {}
            for field in true.keys():
                if field == 'skills':
                    continue
                true_val = str(true.get(field, '')).lower()
                pred_val = str(pred.get(field, '')).lower()
                if true_val != pred_val:
                    mismatches[field] = {'expected': true.get(field), 'predicted': pred.get(field)}
            
            if mismatches and len(failures) < n_cases:
                failures.append({
                    'index': i,
                    'text_preview': text[:200],
                    'mismatches': mismatches
                })
        return failures
    
    def get_model_size(self) -> float:
        import pickle
        total = 0
        for field in self.models:
            total += len(pickle.dumps(self.models[field]))
            total += len(pickle.dumps(self.vectorizers[field]))
        return total / (1024 * 1024)