# 10ways2do — Multi-Dimensional AI Benchmark Platform

10ways2do is a contamination-resistant, multi-dimensional evaluation engine and visualization platform for Large Language Models (LLMs). 

Unlike static multiple-choice benchmarks that models can memorize, 10ways2do generates **unique, programmatic challenges at runtime** and evaluates models across 8 distinct performance axes, providing a realistic view of how a model will perform in production.

## Key Features

- **Dynamic Challenge Generation:** Avoids data contamination by procedurally generating tests with different seeds, difficulties, and parameters at runtime.
- **8-Axis Radar Scoring:** Moves beyond single-number leaderboards. Evaluates Accuracy, Speed, Cost, Robustness, Fairness, Consistency, Generalization, and Efficiency.
- **Provider Agnostic:** Built-in support for OpenAI, Anthropic, Google Gemini, Perplexity, local mock testing, and more.
- **Interactive Dashboard:** Beautiful, dynamic UI featuring interactive Scatter Plots (Performance vs Cost) and Head-to-Head Radar Comparisons.

## The 10 Evaluation Domains

The benchmark tests models across 10 distinct cognitive and operational domains:
1. Anomaly Detection
2. Classification
3. Code Generation
4. Hallucination Resistance
5. Information Extraction
6. Instruction Following
7. Logical Deduction
8. Reasoning
9. Summarization
10. Tool Use Planning

## Setup & Installation

### 1) Create and activate a virtual environment

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Set your API Keys (Optional)
Depending on which models you want to test, set your environment variables or pass them directly via the CLI:
```bash
# Example for Perplexity
set PERPLEXITY_API_KEY=your_key_here
```

## Running Evaluations

Use the CLI tool to trigger evaluations. Results are automatically saved to `results/` and registered in the `challenge_registry.json`.

**Run a baseline mock test:**
```bash
python main.py --evaluate-model --provider mock --model mock-v1
```

**Run a real model evaluation (e.g., Perplexity Sonar):**
```bash
python main.py --evaluate-model --provider perplexity --model sonar --api-key YOUR_API_KEY
```

By default, this runs the `balanced` scoring profile across all 10 domains.

## Viewing the Dashboard

To view the interactive leaderboards, radar charts, and scatter plots, you need to serve the project directory over HTTP (to allow the UI to fetch the local JSON results).

1. Start a local server from the root directory:
```bash
python -m http.server 8765
```

2. Open your browser and navigate to:
[http://localhost:8765/website/index.html](http://localhost:8765/website/index.html)

## License
MIT (see LICENSE)
