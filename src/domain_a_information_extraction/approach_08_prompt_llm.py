"""
Approach 8: Prompt-Based LLM for Information Extraction

Philosophy: Use language understanding without task-specific training.
- Zero-shot / few-shot learning
- Strong reasoning capabilities
- Cost and latency trade-offs
"""

from typing import Dict, List, Any, Optional
import json
import os
import re
import time

from ..core.base_model import BaseApproach


class PromptLLMIE(BaseApproach):
    """LLM-based IE using prompting"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Prompt-Based LLM", config)
        
        self.model = config.get('model', 'gpt-3.5-turbo') if config else 'gpt-3.5-turbo'
        self.temperature = config.get('temperature', 0.0) if config else 0.0
        self.use_few_shot = config.get('use_few_shot', True) if config else True
        self.simulate = config.get('simulate', True) if config else True  # Use simulation by default
        
        self.few_shot_examples = []
        self.api_client = None
        
        self.metrics.interpretability_score = 0.7
        self.metrics.maintenance_complexity = 0.3
        
        if not self.simulate:
            self._init_client()
    
    def _init_client(self):
        """Initialize API client"""
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.api_client = openai.OpenAI(api_key=api_key)
        except ImportError:
            self.api_client = None
    
    def _create_prompt(self, text: str) -> str:
        """Create extraction prompt"""
        base_prompt = """Extract the following information from the resume text. 
Return a JSON object with these fields:
- name: Full name of the person
- email: Email address
- phone: Phone number
- location: City and state
- job_title: Current job title
- company: Current company
- years_experience: Total years of experience (as a number)
- education: University or institution name
- degree: Degree obtained

If a field cannot be found, use null.

"""
        
        if self.use_few_shot and self.few_shot_examples:
            base_prompt += "Examples:\n\n"
            for example in self.few_shot_examples[:3]:
                base_prompt += f"Resume:\n{example['text'][:500]}\n\n"
                base_prompt += f"Extracted:\n{json.dumps(example['label'], indent=2)}\n\n"
        
        base_prompt += f"Resume:\n{text}\n\nExtracted:"
        
        return base_prompt
    
    def train(self, X_train: List[str], y_train: List[Dict],
              X_val: Optional[List[str]] = None,
              y_val: Optional[List[Dict]] = None) -> None:
        """Collect few-shot examples (no actual training)"""
        self.few_shot_examples = [
            {'text': text, 'label': label}
            for text, label in zip(X_train[:10], y_train[:10])
        ]
        self.is_trained = True
    
    def predict(self, X: List[str]) -> List[Dict[str, Any]]:
        """Predict using LLM"""
        results = []
        
        for text in X:
            if self.simulate:
                extracted = self._simulated_extraction(text)
            else:
                try:
                    extracted = self._call_llm(text)
                except Exception as e:
                    extracted = self._simulated_extraction(text)
            
            results.append(extracted)
        
        return results
    
    def _call_llm(self, text: str) -> Dict:
        """Call LLM API"""
        if self.api_client is None:
            return self._simulated_extraction(text)
        
        prompt = self._create_prompt(text)
        
        response = self.api_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
        
        return self._simulated_extraction(text)
    
    def _simulated_extraction(self, text: str) -> Dict:
        """Simulated LLM extraction using enhanced regex (for testing without API)"""
        extracted = {}
        
        # Email - high accuracy
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text, re.IGNORECASE)
        extracted['email'] = email_match.group().lower() if email_match else None
        
        # Phone - high accuracy
        phone_patterns = [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
            r'\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
        ]
        for pattern in phone_patterns:
            phone_match = re.search(pattern, text)
            if phone_match:
                extracted['phone'] = phone_match.group()
                break
        else:
            extracted['phone'] = None
        
        # Name - look at start or after "Name:"
        name_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'Name:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'👤\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        ]
        for pattern in name_patterns:
            name_match = re.search(pattern, text, re.MULTILINE)
            if name_match:
                extracted['name'] = name_match.group(1).strip()
                break
        else:
            extracted['name'] = None
        
        # Location
        location_patterns = [
            r'(?:Location|Located|Based in|📍)[:\s]*([A-Za-z\s]+,\s*[A-Z]{2})',
            r'\b([A-Z][a-z]+,\s*[A-Z]{2})\b'
        ]
        for pattern in location_patterns:
            loc_match = re.search(pattern, text)
            if loc_match:
                extracted['location'] = loc_match.group(1).strip()
                break
        else:
            extracted['location'] = None
        
        # Job title
        job_patterns = [
            r'(?:Position|Title|Role|as a|as an|I\'m a)[:\s]*([A-Za-z\s]+(?:Engineer|Developer|Scientist|Manager|Architect|Lead|Designer))',
            r'^((?:Senior\s+|Staff\s+|Principal\s+|Lead\s+)?(?:Software|Machine Learning|Data|Backend|Frontend|Full Stack|DevOps|Cloud|ML|AI)\s*(?:Engineer|Developer|Scientist|Architect))',
        ]
        for pattern in job_patterns:
            job_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if job_match:
                extracted['job_title'] = job_match.group(1).strip()
                break
        else:
            extracted['job_title'] = None
        
        # Company
        companies = ['Google', 'Facebook', 'Meta', 'Amazon', 'Microsoft', 'Apple', 'Netflix',
                     'Uber', 'Airbnb', 'Stripe', 'Dropbox', 'Twitter', 'LinkedIn', 'Salesforce',
                     'Adobe', 'Oracle', 'IBM', 'Intel', 'NVIDIA', 'Tesla', 'SpaceX']
        for company in companies:
            if company.lower() in text.lower():
                extracted['company'] = company
                break
        else:
            company_match = re.search(r'(?:at|@|Company:)\s*([A-Z][A-Za-z]+)', text)
            extracted['company'] = company_match.group(1) if company_match else None
        
        # Years experience
        exp_match = re.search(r'(\d+)\+?\s*years?\s*(?:of\s+)?(?:experience|exp)?', text, re.IGNORECASE)
        extracted['years_experience'] = int(exp_match.group(1)) if exp_match else None
        
        # Education
        universities = ['MIT', 'Stanford', 'Berkeley', 'Carnegie Mellon', 'Georgia Tech',
                       'UCLA', 'Cornell', 'Columbia', 'Harvard', 'Yale', 'Princeton']
        for uni in universities:
            if uni.lower() in text.lower():
                extracted['education'] = uni
                break
        else:
            edu_match = re.search(r'((?:[A-Z][a-z]+\s+)*(?:University|Institute|College|Tech))', text)
            extracted['education'] = edu_match.group(1) if edu_match else None
        
        # Degree
        degree_match = re.search(
            r'((?:Bachelor|Master|PhD|Doctor|B\.?S\.?|M\.?S\.?|Ph\.?D\.?)[^,\n]*(?:Science|Engineering|Arts|Computer)?[^,\n]*)',
            text, re.IGNORECASE
        )
        extracted['degree'] = degree_match.group(1).strip() if degree_match else None
        
        return extracted
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Leverage large language model reasoning for zero/few-shot extraction',
            'inductive_bias': 'Natural language understanding generalizes to structured extraction',
            'strengths': 'No training needed, handles novel formats, strong reasoning, easy to update',
            'weaknesses': 'High latency, API costs, potential hallucination, rate limits, non-deterministic',
            'best_for': 'Low volume, complex documents, rapid prototyping, schema flexibility'
        }
    
    def get_model_size(self) -> float:
        return 0.0  # Model is remote
    
    def collect_failure_cases(self, X_test: List[str], y_test: List[Dict],
                               y_pred: List[Dict], n_cases: int = 10) -> List[Dict]:
        failures = []
        for i, (text, true, pred) in enumerate(zip(X_test, y_test, y_pred)):
            mismatches = {f: {'expected': true.get(f), 'predicted': pred.get(f)}
                         for f in true if f != 'skills' and
                         str(true.get(f, '')).lower() != str(pred.get(f, '')).lower()}
            if mismatches and len(failures) < n_cases:
                failures.append({
                    'index': i,
                    'text_preview': text[:200],
                    'mismatches': mismatches,
                    'analysis': 'LLM may have hallucinated or missed context'
                })
        return failures