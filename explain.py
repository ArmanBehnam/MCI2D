"""
explain.py
by GNN causal explanation
"""
# ---------------------
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GINConv
from torch_geometric.data import Data
import pickle
import pandas as pd
import cxgnn
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return x



def model(state_dict, input_dim):

    hidden_channels = state_dict['conv1.nn.2.weight'].shape[0]
    out_channels = state_dict['lin.weight'].shape[0]
    loaded_model = GIN(input_dim, hidden_channels, out_channels)
    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()

    # Create dummy features to match the input dimension
    for node in G.nodes():
        G.nodes[node]['features'] = [G.nodes[node]['mi_score']] + [0] * (input_dim - 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    node_list = list(G.nodes())
    node_to_index = {node: i for i, node in enumerate(node_list)}
    node_features = torch.tensor([G.nodes[node]['features'] for node in node_list], dtype=torch.float).to(device)
    edge_index = torch.tensor([[node_to_index[u], node_to_index[v]] for u, v in G.edges()],
                              dtype=torch.long).t().contiguous().to(device)
    labels = torch.tensor([G.nodes[node]['label'] for node in node_list], dtype=torch.long).to(device)
    node_features = (node_features - node_features.mean(dim=0)) / (node_features.std(dim=0) + 1e-5)
    data = Data(x=node_features, edge_index=edge_index, y=labels).to(device)

    # Initialize model
    num_classes = len(torch.unique(data.y))
    model = GIN(input_dim, hidden_channels, out_channels).to(device)
    role_id = [G.nodes[node]['label'] for node in G.nodes()]
    print("Graph nodes:", list(G.nodes()))
    print("Node labels:", role_id)

    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

    out = model(data.x, data.edge_index)
    predicted_classes = out.argmax(dim=1)
    return role_id, data, device


def implementation(role_id, data, device, num_epochs=None):
    node_imp = torch.tensor(role_id, dtype=torch.float).to(device)
    N = data.x.size(0)  # Number of nodes
    edge_imp = torch.zeros((N, N), dtype=torch.float).to(device)
    edge_imp[data.edge_index[0], data.edge_index[1]] = 1.

    node_index_to_name = {i: node for i, node in enumerate(G.nodes())}

    # Create data1 DataFrame with node names as columns
    # Create data1 DataFrame with node features
    data1 = pd.DataFrame({node: [G.nodes[node]['features'][0]] for node in G.nodes()})
    print("Data shape:", data1.shape)
    print("Data columns:", data1.columns)
    cg = cxgnn.CausalGraph(V=G.nodes, path=G.edges)
    relative_positives = (node_imp == 1).nonzero(as_tuple=True)[0]
    relative_positives = relative_positives.cpu().tolist()

    models, best_total_loss, best_model, best_expected_p, best_output, best_new_v, best_node = cxgnn.alg_2(
        Graph=cg,
        num_epochs=num_epochs,
        data=data1,
        role_id=role_id,
    )

    print("\nExpressivity for each variable:")
    expressivity = {node: model_info['expected_p'] for node, model_info in models.items()}
    for node, exp in sorted(expressivity.items(), key=lambda x: x[1], reverse=True):
        print(f"{node}: {exp:.4f}")

    return models, best_total_loss, best_model, best_expected_p, best_output, best_new_v, best_node

def visualize(G,best_new_v):
    plt.figure(figsize=(10, 8), dpi=300)
    pos = nx.kamada_kawai_layout(G)
    node_colors = ["#b55822" if node in best_new_v else "#788f8b" for node in G.nodes()]
    edge_colors = ["#C6442A" if u in best_new_v and v in best_new_v else "grey" for u, v in G.edges()]

    nx.draw(G, pos, with_labels=False, node_size=100,
            node_color= node_colors, edge_color=edge_colors, width=2)

    # Add expressivity values as labels above the nodes
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.1)  # Increased offset

    node_labels = {node: f"{node}" for node in G.nodes()}
    label_pos = {k: (v[0], v[1] + 0.02) for k, v in pos.items()}  # Increased offset
    nx.draw_networkx_labels(G, label_pos, font_size=10, font_weight='bold', verticalalignment='bottom')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def results(G, best_new_v, role_id):
    # best_new_v and best_node should already be node names, not indices
    print('The ground truth is: ', [node for node, label in zip(G.nodes(), role_id) if label == 1])
    print('Our method finding is: ', best_new_v)

    # Convert relative_positives and best_new_v to indices for tensor operations
    relative_positives = [node for node, label in zip(G.nodes(), role_id) if label == 1]
    relative_positives_indices = [list(G.nodes()).index(node) for node in relative_positives]
    best_new_v_indices = [list(G.nodes()).index(node) for node in best_new_v]

    my_predictions = torch.zeros(len(G.nodes()))
    my_predictions[best_new_v_indices] = 1
    relative_positives_tensor = torch.zeros(len(G.nodes()))
    relative_positives_tensor[relative_positives_indices] = 1

    my_recall = torch.sum(my_predictions[relative_positives_indices]).item() / len(best_new_v) if len(best_new_v) > 0 else 0
    my_acc = (torch.sum((relative_positives_tensor == 1) & (my_predictions == 1)).item()) / torch.sum(relative_positives_tensor).item() * 100 if torch.sum(relative_positives_tensor).item() > 0 else 0
    my_gt_find = int(set(best_new_v) == set(relative_positives))
    my_validity = int(all(item in relative_positives for item in best_new_v))

    print(f"My Explanation recall is: {my_recall * 100:.2f}%")
    print("My Explanation found the ground truth? ", my_gt_find)
    print("Is our final subgraph valid? ", my_validity)
    print("How well the ground truth is found? {:.2f}".format(my_acc), '\n')


    # Generate random data for demonstration
    expected_p_nodes = {node: random.random() for node in G.nodes()}

    plt.figure(figsize=(12, 8), dpi=300)

    sorted_items = sorted(expected_p_nodes.items(), key=lambda x: x[1], reverse=True)
    sorted_nodes, sorted_values = zip(*sorted_items)

    colors = ['green' if node in best_new_v else 'red' for node in sorted_nodes]

    plt.bar(range(len(sorted_nodes)), sorted_values, color=colors)
    plt.xticks(range(len(sorted_nodes)), sorted_nodes, rotation=90)
    plt.xlabel('Nodes')
    plt.ylabel('Expected Value')
    plt.title('Node Expressivity Distribution')

    legend_elements = [Patch(facecolor='green', edgecolor='green', label='Our Finding'),
                       Patch(facecolor='red', edgecolor='red', label='Not Our Finding')]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Load the graph
    with open('graph1.pkl', 'rb') as f:
        G = pickle.load(f)

    state_dict = torch.load('gnn_model1.pth')
    input_dim = state_dict['conv1.nn.0.weight'].shape[1]
    role_id, data, device = model(state_dict, input_dim)
    models, best_total_loss, best_model, best_expected_p, best_output, best_new_v, best_node = implementation(role_id, data, device, num_epochs=20)
    visualize(G, best_new_v)
    results(G, best_new_v, role_id)
