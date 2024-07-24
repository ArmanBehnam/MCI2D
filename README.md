# MCI Progression Analysis by GNN Explanation

This repository contains code and analysis for studying Mild Cognitive Impairment (MCI) progression using Graph Neural Networks (GNNs) explanation methods.

## Overview

We analyze MCI progression and reversion using temporal medical data. Our approach involves data preprocessing, GNN training, and applying various GNN explanation methods to understand the factors influencing MCI transitions.

## Key Components

1. Data Preprocessing
   - Temporal ordering of patient visits
   - Missing data imputation using MICE and KNN

2. GNN Training
   - Models: GCN, T-GCN, GAT, GraphSage
   - Classification task: Predicting disease progression stages

3. GNN Explanation Methods
   - Guided Backpropagation (Guidedbp)
   - GNNExplainer
   - PGMExplainer
   - RCExplainer

4. MCI Progress and Reversion Analysis
   - Study of main nodes and their values over time

## Results

- GNN performance metrics (Accuracy, F1 score, AUROC, Precision)
- Causal subgraph visualization
- Clinical findings on MCI transitions
- Evaluation of GNN explanation methods (GES, GEA, GEF)
- Gender role analysis in MCI transitions

## Evaluation Metrics

We use the following metrics to evaluate our GNN explanation methods:
- Graph Explanation Accuracy (GEA)
- Graph Explanation Faithfulness (GEF)
- Graph Explanation Stability (GES)

## Usage

[Include instructions on how to run the code, any dependencies, and data requirements]

## Contributing

[Include guidelines for contributing to the project]

## License

[Specify the license under which this project is released]

## Contact

[Provide contact information for the project maintainers]
