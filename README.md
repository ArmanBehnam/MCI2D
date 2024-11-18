# MCI Progression Analysis by GNN Explanation

## Overview

This repository contains code and analysis for studying Mild Cognitive Impairment (MCI) progression using Graph Neural Networks (GNNs) explanation methods, with a focus on causal relationships and explainability.

We analyze MCI progression and reversion using temporal medical data through a novel approach combining GNNs with causal explanation methods. Our research focuses on identifying key factors influencing MCI transitions and understanding their causal relationships.

## Key Components

### 1. Data Preprocessing
- Temporal ordering of patient visits
- Missing data imputation using MICE and KNN
- Sample size: 369 patients with multiple classes

### 2. GNN Training
- Models: T-GCN, GAT, GraphSage
- Classification task: Predicting disease progression stages
- Training parameters:
  - K-fold cross-validation
  - Early stopping
  - Hyperparameter tuning via Optuna
  - AdamW optimizer
  - 200 epochs

### 3. GNN Explanation Methods
- CXGNN (Novel Causal Explainer)
- Guided Backpropagation (Guidedbp)
- GNNExplainer
- Integrated Gradients (IG)
- RCExplainer

### 4. MCI Progress and Reversion Analysis
- Causal subgraph analysis
- Biomarker tracking
- Sex-specific influence analysis

## Results

### GNN Model Performance

| Method    | Accuracy | F-1 Score | AUROC | Precision |
|-----------|----------|-----------|--------|-----------|
| T-GCN     | **0.724**    | **0.700**     | 0.639  | 0.694     |
| GAT       | 0.346    | 0.372     | **0.723**  | **0.786**    |
| GraphSage | 0.356    | 0.399     | 0.564  | 0.775     |

### Explanation Method Evaluation

| Method        | GES  | GEA  | GEF   |
|---------------|------|------|--------|
| CXGNN        | **1.00** | **0.78** | **0.001**  |
| Guidedbp     | 0.64 | 0.22 | 8.600  |
| GNNExplainer | 0.21 | 0.22 | 5.350  |
| IG           | 0.29 | 0.10 | 11.330 |
| RCExplainer  | 0.22 | 0.22 | 7.100  |

### Key Findings

#### Causal Factors
Important variables identified through expressivity analysis:
- Hypertension (primary target node)
- Arrhythmia
- Congestive heart failure
- Coronary artery disease
- Stroke
- Lipid-related issues

#### Biomarker Expressivity
- Neurofilament Light Chain: 0.2901
- Glial Fibrillary Acidic Protein: 0.2868
- ptau181: 0.2104
- APOE ε4 allele: 0.0807

#### Sex-Specific Analysis
Direct connections with sex (edge weights):
- Osteoporosis (0.3485)
- Coronary artery disease (0.2275)
- Glial fibrillary acidic protein biomarker (0.1422)

## Evaluation Metrics

- **Graph Explanation Stability (GES)**: Measures consistency of explanations across different graph structures
- **Graph Explanation Accuracy (GEA)**: Evaluates accuracy of the explanation method
- **Graph Explanation Faithfulness (GEF)**: Measures how faithful the explanation is to the model's decision process

## Limitations

- Sample size limitations (n=369)
- Generalizability concerns due to Midwestern population focus (MCSA cohort)
- Potential unmeasured confounders
- Need for external validation

## Installation

```bash
# Clone the repository
git clone https://github.com/username/mci-progression-analysis.git

# Navigate to the project directory
cd mci-progression-analysis

# Install required packages
pip install -r requirements.txt
```

## Usage

```python
# Example code for running the analysis
from mci_analysis import GNNAnalyzer

# Initialize the analyzer
analyzer = GNNAnalyzer()

# Run the analysis
results = analyzer.run()
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

This study was supported by:
- NIA R01 AG068007
- Eric & Wendy Schmidt Fund for AI Research & Innovation
- Mayo Clinic Study of Aging (NIH Grants U01 AG006786, P50 AG016574, R01AG057708)
- GHR Foundation
- Mayo Foundation for Medical Education and Research
- Rochester Epidemiology Project (R01 AG034676)

## Contact

Arman Behnam - abehnam@hawk.iit.edu
