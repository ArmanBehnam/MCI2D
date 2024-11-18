import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
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

def train_model(data):
    in_channels = data.num_features
    out_channels = 3  # We have 3 classes
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

    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, None)
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask] == data.y[data.test_mask]
        acc = int(correct.sum()) / int(data.test_mask.sum())
        print(f'Test Accuracy: {acc:.4f}')

    torch.save(model.state_dict(), 'gnn_model1.pth')
    return model

def load_pretrained_model(state_dict_path, input_dim, num_classes):
    model = GIN(input_dim, 32, num_classes)
    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model
