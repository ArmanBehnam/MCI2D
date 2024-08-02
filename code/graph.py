"""
graph.py
Making the graph from dataset
"""
# ---------------------
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle
from torch_geometric.nn import GINConv
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.utils import add_self_loops, degree
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
def preprocess(path):
    df = pd.read_csv(path)
    df.drop(['age_group','time_point'], axis=1, inplace=True)
    X = df.drop(['group', 'REPID', 'year'], axis=1)
    if df['group'].nunique() < 2:
        raise ValueError("'group' column must have at least two unique values.")
    mi_scores = []
    for year in df['year'].unique():
        df_year = df[df['year'] == year]
        if len(df_year) > 0 and df_year['group'].nunique() > 1:
            X_year = df_year.drop(['group', 'REPID', 'year'], axis=1)
            y_year = df_year['group']
            mi_year = mutual_info_classif(X_year, y_year)
            mi_scores.append(mi_year)
        else:
            print(f"Skipping year {year} due to insufficient data")

    if not mi_scores:
        raise ValueError("No valid data for mutual information calculation")

    avg_mi_scores = np.mean(mi_scores, axis=0)
    mi_threshold = np.mean(avg_mi_scores) - 0.05 * np.std(avg_mi_scores)
    node_labels = (avg_mi_scores > mi_threshold).astype(int)
    print(f"MI Threshold: {mi_threshold:.4f}")
    print(f"Number of columns in X: {len(X.columns)}")

    # Create graph
    G = nx.Graph()
    for i, col in enumerate(X.columns):
        G.add_node(col, mi_score=avg_mi_scores[i], label=node_labels[i])

    corr_matrix = X.corr(method='spearman')
    corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    correlation_threshold = np.mean(corr_values) + 0.5 * np.std(corr_values)  # Adjusted threshold

    # Add edges based on correlation
    for i in range(len(X.columns)):
        for j in range(i+1, len(X.columns)):
            if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                G.add_edge(X.columns[i], X.columns[j], weight=abs(corr_matrix.iloc[i, j]))

    print(f"Correlation Threshold: {correlation_threshold:.4f}")

    # Print node information and top correlations
    for node, data in G.nodes(data=True):
        print(f"Node: {node}, Label: {data['label']}, MI Score: {data['mi_score']:.4f}")

    top_vars = [var for var, score in sorted(zip(X.columns, avg_mi_scores), key=lambda x: x[1], reverse=True)]
    print("\nCorrelations between variables:")
    print(corr_matrix.loc[top_vars, top_vars])

    # Print top features by mutual information
    top_features = sorted(G.nodes(data=True), key=lambda x: x[1]['mi_score'], reverse=True)[:10]
    print("\nTop 10 features by mutual information with target:")
    for feature, data in top_features:
        print(f"{feature}: {data['mi_score']:.4f}")
    y = df['group']
    print("Unique values in target:", y.unique())
    print("Min and max of target:", y.min(), y.max())
    y = y - 1  # Subtract 1 from all labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train = torch.FloatTensor(X_train.values)
    x_test = torch.FloatTensor(X_test.values)
    y_train = torch.LongTensor(y_train.values)
    y_test = torch.LongTensor(y_test.values)
    num_nodes = X.shape[1]
    edge_index = torch.tensor([[i for i in range(num_nodes) for _ in range(num_nodes)],
                               [j for _ in range(num_nodes) for j in range(num_nodes)]],
                              dtype=torch.long)

    train_mask = torch.zeros(x_train.shape[0] + x_test.shape[0], dtype=torch.bool)
    train_mask[:x_train.shape[0]] = True
    test_mask = ~train_mask
    x = torch.cat([x_train, x_test], dim=0)
    y = torch.cat([y_train, y_test], dim=0)
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    num_features = data.num_features
    num_classes = data.y.max().item() + 1
    hidden_channels = 256
    return X, y, G, correlation_threshold,corr_values, num_features, num_classes, hidden_channels, data

def visualize(G):
    seed = 42
    random.seed(seed)
    plt.figure(figsize=(12, 10))
    important_nodes = [node for node, data in G.nodes(data=True) if data['label'] == 1]
    # Separate red and blue nodes
    red_nodes = [node for node, data in G.nodes(data=True) if data['label'] == 1]
    blue_nodes = [node for node, data in G.nodes(data=True) if data['label'] == 0]
    pos = {}
    for i, node in enumerate(blue_nodes):
        pos[node] = (random.uniform(0, 0.5), random.random())
    for i, node in enumerate(red_nodes):
        pos[node] = (random.uniform(0.6, 1), random.random())
    fixed_nodes = list(G.nodes())
    pos = nx.spring_layout(G, pos=pos, fixed=fixed_nodes, k=0.5, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=[G.nodes[n]['label'] for n in G.nodes], node_size=300, cmap=plt.cm.coolwarm)
    # Draw edges with varying thickness based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3)
    # Draw labels above nodes
    label_pos = {k: (v[0], v[1] + 0.02) for k, v in pos.items()}  # Adjust the 0.03 to move labels up/down
    nx.draw_networkx_labels(G, label_pos, font_size=10, verticalalignment='bottom')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2):
        super(GIN, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return x


def model(data):
    in_channels = data.num_features  # number of features
    out_channels = int(data.y.max().item()) + 1  # number of classes
    model = GIN(in_channels, 32, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, None)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    print(f"out shape: {out.shape}")
    print(f"data.y shape: {data.y.shape}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, None)
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask] == data.y[data.test_mask]
        acc = int(correct.sum()) / int(data.test_mask.sum())
        print(f'Test Accuracy: {acc:.4f}')
        print(f'Predictions distribution: {torch.bincount(pred[data.test_mask])}')
        print(f'True labels distribution: {torch.bincount(data.y[data.test_mask])}')

    torch.save(model.state_dict(), 'gnn_model1.pth')



class FeatureImportance(nn.Module):
    def __init__(self, num_features):
        super(FeatureImportance, self).__init__()
        self.importance = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        return x * self.importance


class BalancedConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(BalancedConv, self).__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out + self.bias


class ImprovedGraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(ImprovedGraphSAGE, self).__init__()
        self.feature_importance = FeatureImportance(num_features)
        self.conv1 = BalancedConv(num_features, hidden_channels)
        self.conv2 = BalancedConv(hidden_channels, hidden_channels)
        self.conv3 = BalancedConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.feature_importance(x)
        x = F.gelu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.gelu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.gelu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = F.gelu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x


def custom_loss(outputs, targets, alpha, class_dist):
    ce_loss = F.cross_entropy(outputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha[targets] * (1 - pt) ** 2 * ce_loss

    pred_dist = torch.softmax(outputs, dim=1).mean(dim=0)
    dist_loss = F.kl_div(pred_dist.log(), class_dist, reduction='batchmean')

    return torch.mean(focal_loss) + 0.1 * dist_loss


def evaluate_model(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].max(1)[1]
        y_true = data.y[mask]

        accuracy = pred.eq(y_true).sum().item() / mask.sum().item()
        f1 = f1_score(y_true.cpu(), pred.cpu(), average='weighted')

        y_pred_proba = F.softmax(out[mask], dim=1).cpu().numpy()
        y_true_bin = label_binarize(y_true.cpu(), classes=range(num_classes))
        auroc = roc_auc_score(y_true_bin, y_pred_proba, average='weighted', multi_class='ovr')

    return accuracy, f1, auroc


def train_model(model, data, optimizer, scheduler, alpha, class_dist, num_epochs=500, patience=50):
    best_val_f1 = 0
    best_model = None
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        loss = custom_loss(out[data.train_mask], data.y[data.train_mask], alpha, class_dist)

        if torch.isnan(loss):
            print(f"NaN loss encountered at epoch {epoch}. Stopping training.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        val_accuracy, val_f1, val_auroc = evaluate_model(model, data, data.val_mask)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(
                f'Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val AUROC: {val_auroc:.4f}')

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_model is not None:
        model.load_state_dict(best_model)
    return model

# Prepare the data
def prepare_data(data):
    num_nodes = data.num_nodes
    all_idx = torch.arange(num_nodes)

    train_val_idx, test_idx = train_test_split(all_idx, test_size=0.2, stratify=data.y)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, stratify=data.y[train_val_idx])

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True

    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask[val_idx] = True

    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[test_idx] = True

    return data

def model_SAGE(data):
    # Prepare the data
    data = prepare_data(data)

    # Calculate class weights and distribution
    class_counts = torch.bincount(data.y[data.train_mask])
    class_weights = 1.0 / class_counts.float()
    alpha = class_weights / class_weights.sum()
    class_dist = class_counts.float() / class_counts.sum()
    alpha = alpha.to(data.x.device)
    class_dist = class_dist.to(data.x.device)

    model = ImprovedGraphSAGE(num_features, hidden_channels, num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=1, epochs=500, pct_start=0.3)
    model = train_model(model, data, optimizer, scheduler, alpha, class_dist)

    # Final evaluation
    test_accuracy, test_f1, test_auroc = evaluate_model(model, data, data.test_mask)
    print(f'Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}, Test AUROC: {test_auroc:.4f}')

    # Print prediction distribution
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[data.test_mask].max(1)[1]
        print(f'Predictions distribution: {torch.bincount(pred, minlength=num_classes)}')
        print(f'True labels distribution: {torch.bincount(data.y[data.test_mask], minlength=num_classes)}')

    # Save the best model
    torch.save(model.state_dict(), 'best_improved_sage_model.pth')

if __name__ == "__main__":
    (X, y, G, correlation_threshold, corr_values,
     num_features, num_classes, hidden_channels, data) = preprocess('processed_data.csv')
    visualize(G)
    with open('graph1.pkl', 'wb') as f:
        pickle.dump(G, f)
    # 1
    model(data)

    # 2
    model_SAGE(data)
