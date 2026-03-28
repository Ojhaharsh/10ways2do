"""
Approach 9: Hybrid Systems for Information Extraction

Philosophy: Combine multiple approaches for best results.
- Rules for high-precision fields
- ML for complex patterns
- LLM for fallback/validation
"""

from typing import Dict, List, Any, Optional, Tuple
import re

from ..core.base_model import BaseApproach
from .approach_01_rule_based import RuleBasedIE
from .approach_02_classical_ml import ClassicalMLIE
from .approach_03_tree_based import TreeBasedIE


class HybridIE(BaseApproach):
    """
    Hybrid IE combining rules, ML, and optional LLM.
    
    Strategy:
    1. Rules for high-precision fields (email, phone)
    2. ML model for learned patterns
    3. Confidence-based fusion
    4. Optional LLM for low-confidence cases
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Hybrid (Rules + ML)", config)
        
        # Component models
        self.rule_model = RuleBasedIE()
        self.ml_model = ClassicalMLIE()
        
        # Field routing configuration
        self.rule_fields = {'email', 'phone', 'years_experience'}  # High precision with rules
        self.ml_fields = {'name', 'location', 'job_title', 'company', 'education', 'degree'}
        
        # Confidence thresholds
        self.confidence_threshold = config.get('confidence_threshold', 0.7) if config else 0.7
        
        self.metrics.interpretability_score = 0.6
        self.metrics.maintenance_complexity = 0.7  # Multiple components
    
    def train(self, X_train: List[str], y_train: List[Dict],
              X_val: Optional[List[str]] = None,
              y_val: Optional[List[Dict]] = None) -> None:
        """Train component models"""
        
        # Rules don't need training
        self.rule_model.train(X_train, y_train)
        
        # Train ML model
        self.ml_model.train(X_train, y_train, X_val, y_val)
        
        self.is_trained = True
    
    def predict(self, X: List[str]) -> List[Dict[str, Any]]:
        """Predict using hybrid strategy"""
        
        # Get predictions from both models
        rule_preds = self.rule_model.predict(X)
        ml_preds = self.ml_model.predict(X)
        
        # Fuse predictions
        results = []
        for rule_pred, ml_pred, text in zip(rule_preds, ml_preds, X):
            fused = self._fuse_predictions(rule_pred, ml_pred, text)
            results.append(fused)
        
        return results
    
    def _fuse_predictions(self, rule_pred: Dict, ml_pred: Dict, text: str) -> Dict:
        """Fuse predictions from different models"""
        fused = {}
        
        all_fields = self.rule_fields | self.ml_fields
        
        for field in all_fields:
            rule_val = rule_pred.get(field)
            ml_val = ml_pred.get(field)
            
            if field in self.rule_fields:
                # Prefer rules for high-precision fields
                if rule_val is not None:
                    fused[field] = rule_val
                elif ml_val is not None:
                    fused[field] = ml_val
                else:
                    fused[field] = None
            else:
                # For other fields, prefer ML but validate
                if ml_val is not None:
                    # Validate ML prediction exists in text
                    if str(ml_val).lower() in text.lower():
                        fused[field] = ml_val
                    elif rule_val is not None:
                        fused[field] = rule_val
                    else:
                        fused[field] = ml_val  # Use anyway
                elif rule_val is not None:
                    fused[field] = rule_val
                else:
                    fused[field] = None
        
        return fused
    
    def predict_with_confidence(self, X: List[str]) -> List[Tuple[Dict, Dict]]:
        """Predict with confidence scores"""
        rule_preds = self.rule_model.predict(X)
        ml_preds = self.ml_model.predict(X)
        
        results = []
        for rule_pred, ml_pred, text in zip(rule_preds, ml_preds, X):
            fused = {}
            confidence = {}
            
            for field in self.rule_fields | self.ml_fields:
                rule_val = rule_pred.get(field)
                ml_val = ml_pred.get(field)
                
                if rule_val is not None and ml_val is not None:
                    if str(rule_val).lower() == str(ml_val).lower():
                        fused[field] = rule_val
                        confidence[field] = 1.0  # High confidence - both agree
                    else:
                        # Disagreement - use field-specific preference
                        if field in self.rule_fields:
                            fused[field] = rule_val
                            confidence[field] = 0.7
                        else:
                            fused[field] = ml_val
                            confidence[field] = 0.6
                elif rule_val is not None:
                    fused[field] = rule_val
                    confidence[field] = 0.8 if field in self.rule_fields else 0.5
                elif ml_val is not None:
                    fused[field] = ml_val
                    confidence[field] = 0.6 if field in self.ml_fields else 0.4
                else:
                    fused[field] = None
                    confidence[field] = 0.0
            
            results.append((fused, confidence))
        
        return results
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Combine multiple approaches, routing by field characteristics',
            'inductive_bias': 'Different fields benefit from different extraction strategies',
            'strengths': 'Best of both worlds, graceful degradation, configurable',
            'weaknesses': 'More complex to maintain, potential inconsistencies, slower',
            'best_for': 'Production systems requiring high reliability across diverse fields'
        }
    
    def get_model_size(self) -> float:
        return self.rule_model.get_model_size() + self.ml_model.get_model_size()
    
    def collect_failure_cases(self, X_test: List[str], y_test: List[Dict],
                               y_pred: List[Dict], n_cases: int = 10) -> List[Dict]:
        failures = []
        
        # Also get component predictions for analysis
        rule_preds = self.rule_model.predict(X_test)
        ml_preds = self.ml_model.predict(X_test)
        
        for i, (text, true, pred, rule_p, ml_p) in enumerate(
            zip(X_test, y_test, y_pred, rule_preds, ml_preds)):
            
            mismatches = {}
            for f in true:
                if f != 'skills' and str(true.get(f, '')).lower() != str(pred.get(f, '')).lower():
                    mismatches[f] = {
                        'expected': true.get(f),
                        'predicted': pred.get(f),
                        'rule_said': rule_p.get(f),
                        'ml_said': ml_p.get(f)
                    }
            
            if mismatches and len(failures) < n_cases:
                failures.append({
                    'index': i,
                    'text_preview': text[:200],
                    'mismatches': mismatches,
                    'analysis': self._analyze_hybrid_failure(mismatches)
                })
        
        return failures
    
    def _analyze_hybrid_failure(self, mismatches: Dict) -> str:
        """Analyze why hybrid approach failed"""
        analyses = []
        for field, mismatch in mismatches.items():
            if mismatch['rule_said'] == mismatch['expected']:
                analyses.append(f"{field}: Rule was correct but ML overrode it")
            elif mismatch['ml_said'] == mismatch['expected']:
                analyses.append(f"{field}: ML was correct but fusion chose wrong")
            else:
                analyses.append(f"{field}: Both components failed")
        return "; ".join(analyses)


class EnsembleIE(BaseApproach):
    """
    Ensemble IE using voting across multiple models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Ensemble IE", config)
        
        self.models = []
        self.weights = []
        
        self.metrics.interpretability_score = 0.5
        self.metrics.maintenance_complexity = 0.8
    
    def add_model(self, model: BaseApproach, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.weights.append(weight)
    
    def train(self, X_train: List[str], y_train: List[Dict],
              X_val: Optional[List[str]] = None,
              y_val: Optional[List[Dict]] = None) -> None:
        """Train all models"""
        for model in self.models:
            model.train(X_train, y_train, X_val, y_val)
        self.is_trained = True
    
    def predict(self, X: List[str]) -> List[Dict[str, Any]]:
        """Predict using weighted voting"""
        all_preds = [model.predict(X) for model in self.models]
        
        results = []
        for i in range(len(X)):
            fused = {}
            sample_preds = [preds[i] for preds in all_preds]
            
            # Get all fields
            all_fields = set()
            for pred in sample_preds:
                all_fields.update(pred.keys())
            
            for field in all_fields:
                # Weighted voting
                votes = {}
                for pred, weight in zip(sample_preds, self.weights):
                    val = pred.get(field)
                    if val is not None:
                        val_str = str(val).lower()
                        votes[val_str] = votes.get(val_str, 0) + weight
                
                if votes:
                    best_val = max(votes, key=votes.get)
                    # Find original case
                    for pred in sample_preds:
                        if str(pred.get(field, '')).lower() == best_val:
                            fused[field] = pred[field]
                            break
                else:
                    fused[field] = None
            
            results.append(fused)
        
        return results
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Aggregate predictions from multiple diverse models',
            'inductive_bias': 'Ensemble reduces individual model biases through averaging',
            'strengths': 'More robust, reduces variance, handles edge cases better',
            'weaknesses': 'Slower inference, higher resource usage, complex debugging',
            'best_for': 'When reliability is critical and resources are available'
        }
    
    def get_model_size(self) -> float:
        return sum(m.get_model_size() for m in self.models)
    
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