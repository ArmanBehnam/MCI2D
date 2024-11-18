import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class FeatureImportance(nn.Module):
    def __init__(self, num_features):
        super(FeatureImportance, self).__init__()
        self.importance = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        return x * self.importance

class BalancedConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(BalancedConv, self).__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)
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

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

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
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.feature_importance(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

def train_model(model, data, optimizer, scheduler, alpha, class_dist):
    model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=alpha)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            train_acc = evaluate_model(model, data, data.train_mask)[0]
            val_acc = evaluate_model(model, data, data.val_mask)[0]
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    return model

def evaluate_model(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].max(1)[1]
        correct = pred.eq(data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        return acc, pred
