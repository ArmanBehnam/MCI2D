import networkx as nx
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from torch_geometric.data import Data
import torch
from sklearn.model_selection import train_test_split

def mutual_info_score(X, y):
    """Calculate mutual information scores for features."""
    kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_discrete = kbd.fit_transform(X)

    mi_scores = []
    for col in range(X_discrete.shape[1]):
        mi = mutual_info_classif(X_discrete[:, col].reshape(-1, 1), y, 
                               discrete_features=True, random_state=42)
        mi_scores.append(mi[0])
    return np.array(mi_scores)

def preprocess(path):
    """Preprocess the data and create graph structure."""
    np.random.seed(42)
    
    # Load and prepare data
    df = pd.read_csv(path)
    X = df.drop(['group', 'REPID', 'year'], axis=1)
    y_original = df['group']

    # Calculate mutual information for each group
    mi_scores = []
    for group in [1, 2, 3]:
        y_binary = (y_original == group).astype(int)
        mi_group = mutual_info_score(X, y_binary)
        mi_scores.append(mi_group)

    # Create graph
    G = nx.Graph()
    y = pd.Series(index=X.columns, dtype=int)
    
    # Add nodes with attributes
    for i, col in enumerate(X.columns):
        most_effective_group = np.argmax([mi_scores[0][i], mi_scores[1][i], mi_scores[2][i]]) + 1
        G.add_node(col,
                  label=most_effective_group,
                  mi_score_1=mi_scores[0][i],
                  mi_score_2=mi_scores[1][i],
                  mi_score_3=mi_scores[2][i],
                  features=X[col].values)
        y[col] = most_effective_group

    # Add edges based on correlation
    corr_matrix = X.corr(method='spearman')
    corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    correlation_threshold = np.mean(corr_values) + 0.5 * np.std(corr_values)

    # Create edges
    for i, col1 in enumerate(sorted(X.columns)):
        for col2 in sorted(X.columns)[i+1:]:
            if abs(corr_matrix.loc[col1, col2]) > correlation_threshold:
                G.add_edge(col1, col2, weight=abs(corr_matrix.loc[col1, col2]))

    # Prepare PyTorch Geometric data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_original, test_size=0.2, random_state=42, stratify=y_original
    )
    
    # Convert to PyTorch tensors
    x_train = torch.FloatTensor(X_train.values)
    x_test = torch.FloatTensor(X_test.values)
    y_train = torch.LongTensor(y_train.values) - 1
    y_test = torch.LongTensor(y_test.values) - 1
    
    # Create edge index
    node_to_index = {node: i for i, node in enumerate(sorted(G.nodes()))}
    edge_index = torch.tensor([
        [node_to_index[u], node_to_index[v]] for u, v in sorted(G.edges())
    ]).t().contiguous()

    # Combine data
    x = torch.cat([x_train, x_test], dim=0)
    y_combined = torch.cat([y_train, y_test], dim=0)
    
    # Create masks
    train_mask = torch.zeros(x.size(0), dtype=torch.bool)
    train_mask[:x_train.size(0)] = True
    test_mask = ~train_mask

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y_combined, 
                train_mask=train_mask, test_mask=test_mask)

    num_features = X.shape[1]
    num_classes = len(y_original.unique())
    hidden_channels = 64

    return X, y, G, correlation_threshold, corr_values, num_features, num_classes, hidden_channels, data
