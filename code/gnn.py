"""
gnn.py
Graph neural Networks Implementation
"""
# ---------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score
from sklearn.preprocessing import label_binarize
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import optuna
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold

def preprocess_data(df):
    # Convert object columns to float
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop non-numeric columns
    df = df.select_dtypes(include=[np.number])

    # Handle NaN and infinite values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median())

    feature_columns = [col for col in df.columns if col not in ['REPID', 'year', 'group']]

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    x = scaler.fit_transform(df[feature_columns].values)

    # Remove any remaining NaN or infinite values
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    x = torch.tensor(x, dtype=torch.float)
    unique_groups = sorted(df['group'].unique())
    group_to_index = {group: index for index, group in enumerate(unique_groups)}
    y = torch.tensor([group_to_index[group] for group in df['group']], dtype=torch.long)

    edge_index = torch.tensor([[i, i + 1] for i in range(len(df) - 1)], dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)
    return data, feature_columns, group_to_index


# Define GNN models
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 8, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class TGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(TGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lstm = torch.nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x, _ = self.lstm(x.unsqueeze(0))
        x = x.squeeze(0)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)




def evaluate_model(model, data, mask, group_to_index):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].max(1)[1]
        y_true = data.y[mask]

        if group_to_index is not None:
            index_to_group = {v: k for k, v in group_to_index.items()}
            pred_original = torch.tensor([index_to_group[p.item()] for p in pred])
            y_true_original = torch.tensor([index_to_group[y.item()] for y in y_true])
        else:
            pred_original = pred
            y_true_original = y_true

        accuracy = pred_original.eq(y_true_original).sum().item() / mask.sum().item()
        f1 = f1_score(y_true_original.cpu(), pred_original.cpu(), average='weighted')

        y_pred_proba = out[mask].softmax(dim=1).cpu().numpy()
        y_true_bin = label_binarize(y_true_original.cpu(), classes=np.unique(y_true_original.cpu()))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            try:
                auroc = roc_auc_score(y_true_bin, y_pred_proba, average='weighted', multi_class='ovr')
            except ValueError:
                auroc = np.nan

        precision = precision_score(y_true_original.cpu(), pred_original.cpu(), average='weighted', zero_division=0)

    return accuracy, f1, auroc, precision


def train_model(model, data, optimizer, criterion, group_to_index, num_epochs=200, patience=20):
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    model.train()
    best_val_f1 = 0
    best_model = None
    no_improve = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        val_accuracy, val_f1, val_auroc, val_precision = evaluate_model(model, data, data.val_mask, group_to_index)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}')

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model)
    return model


def k_fold_cross_validation(model_class, data, num_features, hidden_channels, num_classes, group_to_index, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(data.x.numpy(), data.y.numpy())):
        print(f"Fold {fold + 1}/{n_splits}")

        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        # Use 20% of train set as validation set
        val_size = int(0.2 * len(train_idx))
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask[train_idx[:val_size]] = True
        train_mask[train_idx[:val_size]] = False

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        model = model_class(num_features, hidden_channels, num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        model = train_model(model, data, optimizer, criterion, group_to_index, num_epochs=300, patience=30)
        accuracy, f1, auroc, precision = evaluate_model(model, data, test_mask, group_to_index)

        fold_results.append({
            'accuracy': accuracy,
            'f1_score': f1,
            'auroc': auroc,
            'precision': precision
        })

    return fold_results


def tune_hyperparameters(model_class, param_grid, data, num_classes):
    def objective(trial):
        params = {
            'hidden_channels': trial.suggest_categorical('hidden_channels', param_grid['hidden_channels']),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'dropout': trial.suggest_float('dropout', 0.1, 0.7),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        }

        model = model_class(data.num_features, params['hidden_channels'], num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'],
                                      weight_decay=params['weight_decay'])
        criterion = nn.CrossEntropyLoss()

        model = train_model(model, data, optimizer, criterion, group_to_index=None, num_epochs=100, patience=20)

        _, val_f1, _, _ = evaluate_model(model, data, data.val_mask, group_to_index=None)
        return val_f1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    return study.best_params


def analyze_results(model, data, feature_columns):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask] == data.y[data.test_mask]
        acc = int(correct.sum()) / int(data.test_mask.sum())

    # Feature importance (approximation)
    feat_importance = torch.abs(data.x[data.test_mask][correct]).mean(dim=0)
    feat_imp = {feat: imp.item() for feat, imp in zip(feature_columns, feat_importance)}

    # Most and least important features
    sorted_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    most_important = sorted_features[:5]
    least_important = sorted_features[-5:]

    return {
        'accuracy': acc,
        'most_important_features': most_important,
        'least_important_features': least_important,
        'top_progressing_diseases': [],  # This would require additional analysis
        'biomarker_disease_correlations': {},  # This would require additional analysis
        'highest_prevalence_age': "Not available"  # This would require additional data
    }


if __name__ == "__main__":
    df = pd.read_csv('processed_data.csv')
    df.drop('age_group', axis=1, inplace=True)

    data, feature_columns, group_to_index = preprocess_data(df)
    unique_groups = sorted(df['group'].unique())
    group_to_index = {group: index for index, group in enumerate(unique_groups)}
    y = torch.tensor([group_to_index[group] for group in df['group']], dtype=torch.long)
    num_features = data.num_features
    num_classes = len(group_to_index)
    print(f'Number of num_classes: {num_classes}')
    hidden_channels = 64

    models = [GCN, GAT, GraphSAGE, TGCN]

    for model_class in models:
        print(f"\nTraining {model_class.__name__}")
        fold_results = k_fold_cross_validation(model_class, data, num_features, hidden_channels, num_classes,
                                               group_to_index)

        avg_results = {metric: np.mean([fold[metric] for fold in fold_results]) for metric in fold_results[0].keys()}

        print(f"\n{model_class.__name__} Average Results:")
        for key, value in avg_results.items():
            print(f"{key} = {value:.4f}")

    with open('gnn_results.pkl', 'wb') as f:
        pickle.dump(avg_results, f)

    print("Analysis complete. Results saved to gnn_results.pkl")

    gcn_param_grid = {
        'hidden_channels': [32, 64, 128, 256],
        'learning_rate': [0.001, 0.01, 0.1],
        'dropout': [0.3, 0.5, 0.7],
        'weight_decay': [1e-4, 1e-3, 1e-2]
    }
    best_params = tune_hyperparameters(GCN, gcn_param_grid, data, num_classes)
    print("Best parameters for GCN:", best_params)
