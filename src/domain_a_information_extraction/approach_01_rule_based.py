"""
Approach 1: Rule-Based / Heuristics for Information Extraction

Philosophy: Encode human knowledge directly through patterns.
- No training data needed
- High precision when rules match
- Low recall for edge cases
"""

import re
from typing import Dict, List, Any, Optional
from ..core.base_model import BaseApproach, ModelMetrics


class RuleBasedIE(BaseApproach):
    """
    Rule-based information extraction using regex and heuristics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Rule-Based IE", config)
        self.patterns = self._compile_patterns()
        
        # Interpretability & maintenance scores
        self.metrics.interpretability_score = 1.0  # Fully interpretable
        self.metrics.maintenance_complexity = 0.3  # Moderate - rules need updates
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile extraction patterns"""
        patterns = {
            'email': [
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE),
            ],
            'phone': [
                re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),
                re.compile(r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}'),
                re.compile(r'\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'),
                re.compile(r'\b\d{3}\.\d{3}\.\d{4}\b'),
            ],
            'name': [
                # Name typically at the start, capitalized
                re.compile(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', re.MULTILINE),
                re.compile(r'Name:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', re.IGNORECASE),
                re.compile(r'👤\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'),
            ],
            'location': [
                re.compile(r'(?:Location|Located|Based):\s*(.+?)(?:\n|$)', re.IGNORECASE),
                re.compile(r'📍\s*(.+?)(?:\n|$)'),
                re.compile(r'\b([A-Z][a-z]+,\s*[A-Z]{2})\b'),  # City, ST format
            ],
            'job_title': [
                re.compile(r'(?:Position|Title|Role):\s*(.+?)(?:\n|$)', re.IGNORECASE),
                re.compile(r'(?:Currently|Working as|I\'m a)\s+(?:a\s+)?([A-Za-z\s]+(?:Engineer|Developer|Manager|Scientist|Architect|Lead))', re.IGNORECASE),
                re.compile(r'^((?:Senior\s+|Staff\s+|Principal\s+)?(?:Software|Machine Learning|Data|Backend|Frontend|Full Stack|DevOps|Cloud)\s*(?:Engineer|Developer|Scientist|Architect))', re.IGNORECASE | re.MULTILINE),
            ],
            'company': [
                re.compile(r'(?:Company|at|@)\s+([A-Z][A-Za-z\s]+?)(?:\n|,|\.|$)', re.IGNORECASE),
                re.compile(r'working at\s+([A-Z][A-Za-z]+)', re.IGNORECASE),
            ],
            'years_experience': [
                re.compile(r'(\d+)\+?\s*years?\s*(?:of\s+)?(?:experience|exp)', re.IGNORECASE),
                re.compile(r'(?:experience|exp):\s*(\d+)', re.IGNORECASE),
            ],
            'education': [
                re.compile(r'(?:University|Institute|College):\s*(.+?)(?:\n|$)', re.IGNORECASE),
                re.compile(r'(?:from|at)\s+((?:[A-Z][a-z]+\s+)*(?:University|Institute|College|Tech))', re.IGNORECASE),
                re.compile(r'((?:MIT|Stanford|Berkeley|Carnegie Mellon|Georgia Tech|UCLA|Cornell|Columbia)[^,\n]*)'),
            ],
            'degree': [
                re.compile(r'(?:Degree|Graduated):\s*(.+?)(?:\n|$)', re.IGNORECASE),
                re.compile(r'((?:Bachelor|Master|PhD|Doctor)[^,\n]+(?:Science|Engineering|Arts)[^,\n]*)', re.IGNORECASE),
                re.compile(r'(B\.?S\.?|M\.?S\.?|Ph\.?D\.?)\s+(?:in\s+)?([A-Za-z\s]+)', re.IGNORECASE),
            ],
        }
        return patterns
    
    def train(self, X_train: Any, y_train: Any,
              X_val: Optional[Any] = None, y_val: Optional[Any] = None) -> None:
        """
        Rule-based approach doesn't need training.
        We could potentially tune patterns based on training data,
        but that would make it a hybrid approach.
        """
        # No-op for pure rule-based
        self.is_trained = True
    
    def predict(self, X: List[str]) -> List[Dict[str, Any]]:
        """Extract information using rules"""
        results = []
        
        for text in X:
            extracted = {}
            
            for field, patterns in self.patterns.items():
                extracted[field] = self._extract_field(text, patterns)
            
            # Post-processing
            extracted = self._post_process(extracted)
            results.append(extracted)
        
        return results
    
    def _extract_field(self, text: str, patterns: List[re.Pattern]) -> Optional[str]:
        """Extract a single field using patterns"""
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                # Return first capturing group if exists, else full match
                if match.groups():
                    return match.group(1).strip()
                return match.group(0).strip()
        return None
    
    def _post_process(self, extracted: Dict) -> Dict:
        """Post-process extracted data"""
        # Clean years_experience
        if extracted.get('years_experience'):
            try:
                extracted['years_experience'] = int(re.search(r'\d+', str(extracted['years_experience'])).group())
            except:
                extracted['years_experience'] = None
        
        # Normalize email
        if extracted.get('email'):
            extracted['email'] = extracted['email'].lower()
        
        return extracted
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Encode human pattern recognition as explicit rules',
            'inductive_bias': 'Assumes patterns are consistent and enumerable',
            'strengths': 'High precision, no training data needed, fully interpretable, zero latency addition',
            'weaknesses': 'Low recall on edge cases, requires manual maintenance, brittle to format changes',
            'best_for': 'Stable, well-defined formats; high-precision requirements; no ML infrastructure'
        }
    
    def collect_failure_cases(self, X_test: List[str], y_test: List[Dict],
                               y_pred: List[Dict], n_cases: int = 10) -> List[Dict]:
        """Collect failure cases for analysis"""
        failures = []
        
        for i, (text, true, pred) in enumerate(zip(X_test, y_test, y_pred)):
            # Find fields that didn't match
            mismatches = {}
            for field in true.keys():
                if field == 'skills':  # Skip list fields
                    continue
                if str(true.get(field, '')).lower() != str(pred.get(field, '')).lower():
                    mismatches[field] = {
                        'expected': true.get(field),
                        'predicted': pred.get(field)
                    }
            
            if mismatches and len(failures) < n_cases:
                failures.append({
                    'index': i,
                    'text_preview': text[:200] + '...',
                    'mismatches': mismatches,
                    'reason': self._analyze_failure(text, mismatches)
                })
        
        return failures
    
    def _analyze_failure(self, text: str, mismatches: Dict) -> str:
        """Analyze why extraction failed"""
        reasons = []
        
        for field, mismatch in mismatches.items():
            expected = mismatch['expected']
            predicted = mismatch['predicted']
            
            if predicted is None:
                reasons.append(f"{field}: No pattern matched (expected '{expected}')")
            else:
                reasons.append(f"{field}: Wrong match - got '{predicted}' instead of '{expected}'")
        
        return "; ".join(reasons)
    
    def get_model_size(self) -> float:
        """Model size in MB (just the patterns)"""
        import sys
        return sys.getsizeof(self.patterns) / (1024 * 1024)


class EnhancedRuleBasedIE(RuleBasedIE):
    """
    Enhanced rule-based IE with dictionary lookups and validation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.name = "Enhanced Rule-Based IE"
        
        # Load dictionaries for validation
        self.known_companies = {
            'google', 'facebook', 'meta', 'amazon', 'microsoft', 'apple',
            'netflix', 'uber', 'airbnb', 'stripe', 'dropbox', 'twitter',
            'linkedin', 'salesforce', 'adobe', 'oracle', 'ibm', 'intel',
            'nvidia', 'tesla', 'spacex'
        }
        
        self.known_universities = {
            'mit', 'stanford', 'berkeley', 'carnegie mellon', 'georgia tech',
            'ucla', 'cornell', 'columbia', 'harvard', 'yale', 'princeton'
        }
        
        self.job_keywords = {
            'engineer', 'developer', 'scientist', 'manager', 'architect',
            'lead', 'director', 'analyst', 'designer'
        }
    
    def _post_process(self, extracted: Dict) -> Dict:
        """Enhanced post-processing with validation"""
        extracted = super()._post_process(extracted)
        
        # Validate company against known list
        if extracted.get('company'):
            company_lower = extracted['company'].lower()
            for known in self.known_companies:
                if known in company_lower:
                    # Use canonical name
                    extracted['company'] = known.title()
                    break
        
        # Validate education against known universities
        if extracted.get('education'):
            edu_lower = extracted['education'].lower()
            for known in self.known_universities:
                if known in edu_lower:
                    extracted['education'] = known.title()
                    break
        
        return extracted


if __name__ == "__main__":
    # Test the rule-based approach
    from .data_generator import create_ie_dataset, get_ie_fields
    from ..core.metrics import compute_ie_metrics
    
    # Generate test data
    dataset = create_ie_dataset(n_train=100, n_val=20, n_test=50)
    
    # Test basic rule-based
    model = RuleBasedIE()
    model.train(dataset['train']['X'], dataset['train']['y'])
    
    predictions = model.predict(dataset['test']['X'])
    
    metrics = compute_ie_metrics(
        dataset['test']['y'],
        predictions,
        get_ie_fields()
    )
    
    print("Rule-Based IE Results:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print("\nPhilosophy:")
    for key, value in model.get_philosophy().items():
        print(f"  {key}: {value}")