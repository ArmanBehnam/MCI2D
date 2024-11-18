import argparse
import pickle
import pandas as pd
import torch
from pathlib import Path

from utils.preprocessing import preprocess
from utils.visualization import visualize, visualize_results
from models.graph_sage import model_SAGE
from analysis.causal_analysis import implementation
from analysis.network_analysis import comprehensive_network_analysis, print_analysis_results

def parse_args():
    parser = argparse.ArgumentParser(description='MCI Progression Analysis')
    parser.add_argument('--data-path', type=str, default='data/processed_data.csv',
                       help='Path to the processed data CSV file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save outputs')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs for training')
    parser.add_argument('--target-node', type=str, default='sexc',
                       help='Target node for network analysis')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    return parser.parse_args()

def setup_directories(output_dir):
    """Create necessary directories for outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'models').mkdir(exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)
    (output_path / 'analysis').mkdir(exist_ok=True)
    return output_path

def run_preprocessing(data_path):
    """Run data preprocessing and initial analysis."""
    print("Starting data preprocessing...")
    df = pd.read_csv(data_path)
    print(f"Group distribution:\n{df['group'].value_counts()}")
    
    X, y, G, correlation_threshold, corr_values, num_features, num_classes, hidden_channels, data = preprocess(data_path)
    
    print(f"\nPreprocessing Statistics:")
    print(f"Number of nodes most effective for group 1: {(y == 1).sum()}")
    print(f"Number of nodes most effective for group 2: {(y == 2).sum()}")
    print(f"Number of nodes most effective for group 3: {(y == 3).sum()}")
    print(f"Total number of nodes: {len(y)}")
    
    return X, y, G, data

def run_analysis(G, y, data, epochs, output_path):
    """Run the main analysis pipeline."""
    # Train GraphSAGE model
    print("\nTraining GraphSAGE model...")
    sage_model = model_SAGE(data)
    torch.save(sage_model.state_dict(), output_path / 'models' / 'graphsage_model.pth')

    # Run causal analysis
    print("\nRunning causal analysis...")
    models, best_total_loss, best_model, best_expected_p, best_output, best_new_v, best_node, expressivity = \
        implementation(G, y, num_epochs=epochs)
    
    # Save results
    results = {
        'best_model': best_model,
        'best_node': best_node,
        'expressivity': expressivity,
        'best_new_v': best_new_v
    }
    with open(output_path / 'analysis' / 'causal_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def main():
    args = parse_args()
    output_path = setup_directories(args.output_dir)
    
    # Preprocessing
    X, y, G, data = run_preprocessing(args.data_path)
    
    # Save preprocessed graph
    with open(output_path / 'analysis' / 'graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    
    # Run main analysis
    results = run_analysis(G, y, data, args.epochs, output_path)
    
    # Run network analysis
    print(f"\nRunning network analysis for node {args.target_node}...")
    network_analysis_results = comprehensive_network_analysis(G, args.target_node)
    with open(output_path / 'analysis' / 'network_analysis.pkl', 'wb') as f:
        pickle.dump(network_analysis_results, f)
    
    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize(G)
        visualize_results(G, results['expressivity'], y)
    
    # Print final results
    print("\nAnalysis Results:")
    print(f"Best node: {results['best_node']}")
    print(f"Number of nodes in best subgraph: {len(results['best_new_v'])}")
    print("\nTop 5 nodes by expressivity:")
    for node, exp in sorted(results['expressivity'].items(), 
                          key=lambda x: x[1], reverse=True)[:5]:
        print(f"{node}: {exp:.4f}")

if __name__ == "__main__":
    main()
