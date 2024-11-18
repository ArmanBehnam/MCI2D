# MCI Progression Analysis by GNN Explanation

## Overview

This repository contains code and analysis for studying Mild Cognitive Impairment (MCI) progression using Graph Neural Networks (GNNs) explanation methods, with a focus on causal relationships and explainability.

We analyze MCI progression and reversion using temporal medical data through a novel approach combining GNNs with causal explanation methods. Our research focuses on identifying key factors influencing MCI transitions and understanding their causal relationships. [https://www.youtube.com/watch?v=TGNxpDf3Hyk](**Video**)    paper 


## Installation

1. Clone the repository:
```bash
git clone https://github.com/ArmanBehnam/MCI2D/tree/main.git
cd mci-progression-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data:
   - File should be in CSV format with columns for features and 'group' labels

2. Run the main analysis:
```python
from mci_analysis.utils.preprocessing import preprocess
from mci_analysis.models.graph_sage import model_SAGE
from mci_analysis.src.causal_analysis import implementation
import pickle

# Process the data
X, y, G, *_ = preprocess('data.csv')

# Save the graph
with open('graph.pkl', 'wb') as f:
    pickle.dump(G, f)

# Train the model
sage_model = model_SAGE(data)

# Run causal analysis
models, *results = implementation(G, y, num_epochs=20)
```

3. Run comprehensive network analysis:
```python
from mci_analysis.src.network_analysis import comprehensive_sexc_analysis

# Load preprocessed graph
with open('graph.pkl', 'rb') as f:
    G = pickle.load(f)

# Run analysis
comprehensive_sexc_analysis(G, target_node='sexc')
```

4. Visualize results:
```python
from mci_analysis.utils.visualization import visualize_results
visualize_results(G, expressivity, y)
```

## Example Scripts

### Basic Analysis
```python
from mci_analysis.utils.preprocessing import preprocess
from mci_analysis.utils.visualization import visualize

# Load and process data
df = pd.read_csv('data.csv')
X, y, G, *_ = preprocess('data.csv')

# Print group statistics
print(f"Group distribution:\n{df['group'].value_counts()}")
print(f"Number of nodes most effective for group 1: {(y == 1).sum()}")
print(f"Number of nodes most effective for group 2: {(y == 2).sum()}")
print(f"Number of nodes most effective for group 3: {(y == 3).sum()}")

# Visualize the graph
visualize(G)
```

### Full Analysis Pipeline
```python
if __name__ == "__main__":
    # Load preprocessed graph
    with open('data/graph.pkl', 'rb') as f:
        G = pickle.load(f)
    
    # Get node labels
    y = pd.Series({node: G.nodes[node]['label'] for node in G.nodes()})
    
    # Run implementation
    results = implementation(G, y, num_epochs=20)
    
    # Analyze and visualize results
    print_results(G, *results)
    visualize_results(G, results[-1], y)
```

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

## Acknowledgments

This study was supported by Eric & Wendy Schmidt Fund for AI Research & Innovation, Mayo Clinic Study of Aging (NIH Grants U01 AG006786, P50 AG016574, R01AG057708).

## Contact

Arman Behnam - abehnam@hawk.iit.edu
