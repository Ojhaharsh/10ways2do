"""
Approach 3: Tree-Based Models for Information Extraction

Philosophy: Decision trees for structured feature spaces.
- Strong on tabular/hybrid features
- Handles non-linear relationships
- Poor sequence understanding
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle

from ..core.base_model import BaseApproach


class TreeBasedIE(BaseApproach):
    """
    Tree-based approach using XGBoost for field extraction.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Tree-Based (XGBoost)", config)
        self.models = {}
        self.vectorizers = {}
        self.feature_extractors = {}
        
        self.metrics.interpretability_score = 0.7  # Feature importance available
        self.metrics.maintenance_complexity = 0.4
    
    def _extract_structural_features(self, text: str) -> np.ndarray:
        """Extract structural features from text"""
        features = []
        
        # Length features
        features.append(len(text))
        features.append(len(text.split()))
        features.append(len(text.split('\n')))
        
        # Character type ratios
        alpha = sum(c.isalpha() for c in text)
        digit = sum(c.isdigit() for c in text)
        space = sum(c.isspace() for c in text)
        total = len(text) + 1
        
        features.append(alpha / total)
        features.append(digit / total)
        features.append(space / total)
        
        # Special character counts
        features.append(text.count('@'))
        features.append(text.count('.'))
        features.append(text.count('-'))
        features.append(text.count('('))
        features.append(text.count(':'))
        
        # Keyword presence
        keywords = ['experience', 'years', 'education', 'skills', 
                   'email', 'phone', 'name', 'university', 'degree']
        for kw in keywords:
            features.append(1 if kw in text.lower() else 0)
        
        # Capitalization patterns
        words = text.split()
        if words:
            features.append(sum(w[0].isupper() for w in words if w) / len(words))
            features.append(sum(w.isupper() for w in words) / len(words))
        else:
            features.extend([0, 0])
        
        return np.array(features)
    
    def _prepare_window_data(self, X_train: List[str], y_train: List[Dict],
                              field: str, window_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare windowed training data"""
        X_windows = []
        y_labels = []
        
        for text, entities in zip(X_train, y_train):
            value = entities.get(field)
            if value is None:
                continue
            
            value_str = str(value).lower()
            text_lower = text.lower()
            
            # Sliding window approach
            for i in range(0, len(text) - window_size, window_size // 2):
                window = text[i:i+window_size]
                
                # Check if value is in window
                label = 1 if value_str in window.lower() else 0
                
                # Extract features
                struct_feats = self._extract_structural_features(window)
                X_windows.append(struct_feats)
                y_labels.append(label)
        
        return np.array(X_windows), np.array(y_labels)
    
    def train(self, X_train: List[str], y_train: List[Dict],
              X_val: Optional[List[str]] = None,
              y_val: Optional[List[Dict]] = None) -> None:
        """Train XGBoost models for each field"""
        
        fields = ['name', 'email', 'phone', 'location', 'job_title',
                  'company', 'education', 'degree']
        
        for field in fields:
            X_windows, y_labels = self._prepare_window_data(X_train, y_train, field)
            
            if len(X_windows) < 20 or y_labels.sum() < 5:
                continue
            
            # Train XGBoost
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            model.fit(X_windows, y_labels)
            self.models[field] = model
            
            # Also train a text vectorizer for candidate ranking
            vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            texts = [str(y.get(field, '')) for y in y_train if y.get(field)]
            if texts:
                vectorizer.fit(texts)
                self.vectorizers[field] = vectorizer
        
        self.is_trained = True
    
    def _extract_candidates(self, text: str, field: str) -> List[str]:
        """Extract candidate values"""
        candidates = []
        
        if field == 'email':
            pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            candidates = re.findall(pattern, text, re.IGNORECASE)
        
        elif field == 'phone':
            patterns = [r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
                       r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}']
            for p in patterns:
                candidates.extend(re.findall(p, text))
        
        elif field == 'years_experience':
            matches = re.findall(r'(\d+)\+?\s*years?', text, re.IGNORECASE)
            candidates = matches
        
        else:
            # N-gram candidates
            words = text.split()
            for i in range(len(words)):
                for n in range(1, 5):
                    if i + n <= len(words):
                        candidate = ' '.join(words[i:i+n])
                        if len(candidate) > 2:
                            candidates.append(candidate)
        
        return list(set(candidates))[:30]
    
    def predict(self, X: List[str]) -> List[Dict[str, Any]]:
        """Predict using tree models"""
        results = []
        
        for text in X:
            extracted = {}
            
            for field in self.models.keys():
                candidates = self._extract_candidates(text, field)
                
                if not candidates:
                    extracted[field] = None
                    continue
                
                # Score candidates using window features
                best_score = -1
                best_candidate = None
                
                for candidate in candidates:
                    pos = text.lower().find(candidate.lower())
                    if pos == -1:
                        continue
                    
                    # Get window around candidate
                    start = max(0, pos - 50)
                    end = min(len(text), pos + len(candidate) + 50)
                    window = text[start:end]
                    
                    # Extract features and predict
                    features = self._extract_structural_features(window).reshape(1, -1)
                    proba = self.models[field].predict_proba(features)[0][1]
                    
                    if proba > best_score:
                        best_score = proba
                        best_candidate = candidate
                
                extracted[field] = best_candidate if best_score > 0.3 else None
            
            # Post-process
            if extracted.get('years_experience'):
                try:
                    extracted['years_experience'] = int(extracted['years_experience'])
                except:
                    extracted['years_experience'] = None
            
            results.append(extracted)
        
        return results
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Use decision trees to learn non-linear feature combinations',
            'inductive_bias': 'Assumes axis-aligned decision boundaries are sufficient',
            'strengths': 'Handles mixed features well, feature importance, robust to outliers',
            'weaknesses': 'Poor at sequence modeling, requires good feature engineering',
            'best_for': 'Hybrid structured/unstructured data, when features are well-defined'
        }
    
    def get_model_size(self) -> float:
        total = 0
        for model in self.models.values():
            total += len(pickle.dumps(model))
        return total / (1024 * 1024)
    
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