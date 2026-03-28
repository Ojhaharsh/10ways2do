"""
Synthetic data generator for Information Extraction domain.
Generates resume-like documents with structured fields to extract.
"""

import random
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from faker import Faker

fake = Faker()
Faker.seed(42)


@dataclass
class ResumeData:
    """Structured resume data"""
    name: str
    email: str
    phone: str
    location: str
    job_title: str
    company: str
    years_experience: int
    skills: List[str]
    education: str
    degree: str
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'location': self.location,
            'job_title': self.job_title,
            'company': self.company,
            'years_experience': self.years_experience,
            'skills': self.skills,
            'education': self.education,
            'degree': self.degree
        }


# Skill categories
TECH_SKILLS = [
    'Python', 'Java', 'JavaScript', 'C++', 'Go', 'Rust', 'SQL', 'R',
    'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy',
    'AWS', 'GCP', 'Azure', 'Docker', 'Kubernetes',
    'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
    'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask',
    'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch',
    'Git', 'CI/CD', 'Agile', 'Scrum'
]

JOB_TITLES = [
    'Software Engineer', 'Senior Software Engineer', 'Staff Engineer',
    'Machine Learning Engineer', 'Data Scientist', 'Data Engineer',
    'Backend Developer', 'Frontend Developer', 'Full Stack Developer',
    'DevOps Engineer', 'SRE', 'Cloud Architect',
    'Product Manager', 'Engineering Manager', 'Tech Lead'
]

DEGREES = [
    'Bachelor of Science in Computer Science',
    'Master of Science in Computer Science',
    'Bachelor of Engineering',
    'Master of Engineering',
    'PhD in Computer Science',
    'Bachelor of Science in Mathematics',
    'Master of Science in Data Science',
    'Bachelor of Science in Information Technology'
]

UNIVERSITIES = [
    'MIT', 'Stanford University', 'UC Berkeley', 'Carnegie Mellon',
    'Georgia Tech', 'University of Michigan', 'University of Washington',
    'Cornell University', 'Columbia University', 'UCLA',
    'University of Texas at Austin', 'University of Illinois',
    'Purdue University', 'University of Wisconsin'
]

COMPANIES = [
    'Google', 'Facebook', 'Amazon', 'Microsoft', 'Apple',
    'Netflix', 'Uber', 'Airbnb', 'Stripe', 'Dropbox',
    'Twitter', 'LinkedIn', 'Salesforce', 'Adobe', 'Oracle',
    'IBM', 'Intel', 'NVIDIA', 'Tesla', 'SpaceX'
]


class ResumeGenerator:
    """Generate synthetic resume data"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        Faker.seed(seed)
    
    def generate_resume(self) -> Tuple[str, ResumeData]:
        """Generate a single resume text and its structured data"""
        
        # Generate structured data
        data = ResumeData(
            name=fake.name(),
            email=fake.email(),
            phone=fake.phone_number(),
            location=f"{fake.city()}, {fake.state_abbr()}",
            job_title=random.choice(JOB_TITLES),
            company=random.choice(COMPANIES),
            years_experience=random.randint(1, 20),
            skills=random.sample(TECH_SKILLS, random.randint(3, 8)),
            education=random.choice(UNIVERSITIES),
            degree=random.choice(DEGREES)
        )
        
        # Generate text (multiple formats)
        template = random.choice([
            self._template_formal,
            self._template_compact,
            self._template_detailed,
            self._template_creative
        ])
        
        text = template(data)
        return text, data
    
    def _template_formal(self, data: ResumeData) -> str:
        """Formal resume template"""
        skills_str = ", ".join(data.skills)
        return f"""
{data.name}
{data.email} | {data.phone}
{data.location}

PROFESSIONAL EXPERIENCE
{data.job_title} at {data.company}
{data.years_experience} years of experience in software development

SKILLS
{skills_str}

EDUCATION
{data.degree}
{data.education}
""".strip()
    
    def _template_compact(self, data: ResumeData) -> str:
        """Compact resume template"""
        skills_str = " • ".join(data.skills)
        return f"""
{data.name} | {data.email} | {data.phone} | {data.location}

Currently working as {data.job_title} at {data.company} with {data.years_experience}+ years experience.
Skills: {skills_str}
Education: {data.degree} from {data.education}
""".strip()
    
    def _template_detailed(self, data: ResumeData) -> str:
        """Detailed resume template"""
        skills_list = "\n".join([f"  - {skill}" for skill in data.skills])
        return f"""
Resume

Personal Information:
  Name: {data.name}
  Email Address: {data.email}
  Phone Number: {data.phone}
  Current Location: {data.location}

Work Experience:
  Current Position: {data.job_title}
  Company: {data.company}
  Total Experience: {data.years_experience} years

Technical Skills:
{skills_list}

Educational Background:
  Degree: {data.degree}
  Institution: {data.education}
""".strip()
    
    def _template_creative(self, data: ResumeData) -> str:
        """Creative/informal resume template"""
        skills_str = " | ".join(data.skills[:5])
        return f"""
👤 {data.name}
📧 {data.email}
📱 {data.phone}
📍 {data.location}

I'm a {data.job_title} currently working at {data.company}. 
With {data.years_experience} years in the industry, I've developed expertise in {skills_str}.

🎓 {data.degree} - {data.education}
""".strip()
    
    def generate_dataset(self, n_samples: int) -> Tuple[List[str], List[Dict]]:
        """Generate a dataset of resumes"""
        texts = []
        labels = []
        
        for _ in range(n_samples):
            text, data = self.generate_resume()
            texts.append(text)
            labels.append(data.to_dict())
        
        return texts, labels
    
    def add_noise_to_text(self, text: str, noise_level: float) -> str:
        """Add noise to text for robustness testing"""
        if noise_level == 0:
            return text
        
        chars = list(text)
        n_changes = int(len(chars) * noise_level)
        
        for _ in range(n_changes):
            idx = random.randint(0, len(chars) - 1)
            change_type = random.choice(['swap', 'delete', 'insert', 'typo'])
            
            if change_type == 'swap' and idx < len(chars) - 1:
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            elif change_type == 'delete':
                chars[idx] = ''
            elif change_type == 'insert':
                chars[idx] = chars[idx] + random.choice('abcdefghijklmnopqrstuvwxyz ')
            elif change_type == 'typo' and chars[idx].isalpha():
                # Common typo: nearby key
                chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
        
        return ''.join(chars)


class InvoiceGenerator:
    """Generate synthetic invoice data"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
    
    def generate_invoice(self) -> Tuple[str, Dict]:
        """Generate invoice text and structured data"""
        
        invoice_number = f"INV-{random.randint(10000, 99999)}"
        date = fake.date_this_year().strftime("%Y-%m-%d")
        due_date = fake.date_between(start_date='today', end_date='+30d').strftime("%Y-%m-%d")
        
        vendor_name = fake.company()
        vendor_address = fake.address().replace('\n', ', ')
        
        customer_name = fake.company()
        customer_address = fake.address().replace('\n', ', ')
        
        n_items = random.randint(1, 5)
        items = []
        subtotal = 0
        
        for _ in range(n_items):
            description = fake.bs().title()
            quantity = random.randint(1, 10)
            unit_price = round(random.uniform(10, 500), 2)
            total = round(quantity * unit_price, 2)
            subtotal += total
            
            items.append({
                'description': description,
                'quantity': quantity,
                'unit_price': unit_price,
                'total': total
            })
        
        tax_rate = random.choice([0.05, 0.07, 0.08, 0.10])
        tax = round(subtotal * tax_rate, 2)
        grand_total = round(subtotal + tax, 2)
        
        data = {
            'invoice_number': invoice_number,
            'date': date,
            'due_date': due_date,
            'vendor_name': vendor_name,
            'vendor_address': vendor_address,
            'customer_name': customer_name,
            'customer_address': customer_address,
            'items': items,
            'subtotal': subtotal,
            'tax': tax,
            'tax_rate': tax_rate,
            'total': grand_total
        }
        
        # Generate text
        items_text = "\n".join([
            f"  {item['description']}: {item['quantity']} x ${item['unit_price']:.2f} = ${item['total']:.2f}"
            for item in items
        ])
        
        text = f"""
INVOICE

Invoice Number: {invoice_number}
Date: {date}
Due Date: {due_date}

From:
{vendor_name}
{vendor_address}

Bill To:
{customer_name}
{customer_address}

Items:
{items_text}

Subtotal: ${subtotal:.2f}
Tax ({tax_rate*100:.0f}%): ${tax:.2f}
Total: ${grand_total:.2f}

Payment Terms: Net 30
""".strip()
        
        return text, data
    
    def generate_dataset(self, n_samples: int) -> Tuple[List[str], List[Dict]]:
        """Generate dataset of invoices"""
        texts = []
        labels = []
        
        for _ in range(n_samples):
            text, data = self.generate_invoice()
            texts.append(text)
            labels.append(data)
        
        return texts, labels


def get_ie_fields() -> List[str]:
    """Get fields for IE evaluation"""
    return ['name', 'email', 'phone', 'location', 'job_title', 
            'company', 'years_experience', 'education', 'degree']


def create_ie_dataset(n_train: int = 5000, n_val: int = 1000, 
                       n_test: int = 1000, seed: int = 42) -> Dict:
    """Create complete IE dataset"""
    generator = ResumeGenerator(seed)
    
    X_train, y_train = generator.generate_dataset(n_train)
    X_val, y_val = generator.generate_dataset(n_val)
    X_test, y_test = generator.generate_dataset(n_test)
    
    return {
        'train': {'X': X_train, 'y': y_train},
        'val': {'X': X_val, 'y': y_val},
        'test': {'X': X_test, 'y': y_test},
        'fields': get_ie_fields(),
        'generator': generator
    }


if __name__ == "__main__":
    # Test generation
    generator = ResumeGenerator()
    text, data = generator.generate_resume()
    print("Generated Resume:")
    print(text)
    print("\nStructured Data:")
    print(json.dumps(data.to_dict(), indent=2))