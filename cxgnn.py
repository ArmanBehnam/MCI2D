from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import product
from torch.distributions import Bernoulli
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
import torch.nn.functional as F
import pandas as pd
class CausalGraph:
    def __init__(self, V, path=[], unobserved_edges=[]):
        self.v = list(V)
        self.set_v = set(V)
        self.labels = {node: Bernoulli(0.5).sample((1,)) for node in self.v}
        self.fn = {v: set() for v in V}  # First neighborhood
        self.sn = {v: set() for v in V}  # Second neighborhood
        self.on = {v: set() for v in V}  # Out of neighborhood
        self.p = set(map(tuple, map(sorted, path)))  # Path to First neighborhood
        self.ue = set(map(tuple, map(sorted, unobserved_edges)))  # Unobserved edges

        for v1, v2 in path:
            self.fn[v1].add(v2)
            self.fn[v2].add(v1)
            self.p.add(tuple(sorted((v1, v2))))

    def __iter__(self):
        return iter(self.v)

    def categorize_neighbors(self,target_node):
        # centrality = {v: len(self.fn[v]) for v in self.v}
        # target_node = max(centrality, key=centrality.get)
        if target_node not in self.set_v:
            return

        one_hop_neighbors = self.fn[target_node]
        two_hop_neighbors = set()

        for neighbor in one_hop_neighbors:
            two_hop_neighbors |= self.fn[neighbor]

        two_hop_neighbors -= one_hop_neighbors
        two_hop_neighbors.discard(target_node)
        out_of_neighborhood = self.set_v - (one_hop_neighbors | two_hop_neighbors | {target_node})

        self.sn[target_node] = two_hop_neighbors
        self.on[target_node] = out_of_neighborhood
        return target_node, one_hop_neighbors, two_hop_neighbors, out_of_neighborhood

    def degrees(self):
        # Calculate degrees of nodes in the graph
        return {node: len(self.fn[node]) for node in self.v}
    def plot(self):
        G = nx.Graph()
        G.add_nodes_from(self.v)
        G.add_edges_from(self.p)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=200, font_size=10, font_weight='bold', node_color="lightblue", edge_color="grey")
        plt.savefig('causal.png')
        plt.show()

    def graph_search(self,cg, v1, v2=None, edge_type="path",target_node = None):
        assert edge_type in ["path", "unobserved"]
        assert v1 in cg.set_v
        assert v2 in cg.set_v or v2 is None

        target, one_hop_neighbors, two_hop_neighbors, out_of_neighborhood = cg.categorize_neighbors(target_node)

        q = deque([v1])
        seen = {v1}
        while len(q) > 0:
            cur = q.popleft()
            cur_fn = cg.fn[cur]
            cur_sn = cg.sn[target_node]
            cur_on = cg.on[target_node]

            cur_neighbors = cur_fn if edge_type == "path" else (cur_sn | cur_on)

            for neighbor in cur_neighbors:
                if neighbor not in seen:
                    if v2 is not None:
                        if (neighbor == v2 and edge_type == "path" and neighbor in one_hop_neighbors) or \
                                (neighbor == v2 and edge_type == "unobserved" and neighbor in (
                                        two_hop_neighbors | out_of_neighborhood)):
                            return True
                    seen.add(neighbor)
                    q.append(neighbor)

        if v2 is None:
            return seen

        return False

    def calculate_probabilities(self, dataset):
        node_counts = {node: 0 for node in self.v}
        total_samples = len(dataset)

        for i in dataset:
            for node, value in i.items():
                if value == 1:
                    node_counts[node] += 1

        node_probabilities = {node: count / total_samples for node, count in node_counts.items()}
        return node_probabilities
    def calculate_joint_probabilities(self, dataset):
        joint_counts = {(node_i, node_j): 0 for node_i in self.v for node_j in self.v if node_i != node_j}
        total_samples = len(dataset)

        for sample in dataset:
            for node_i, node_j in joint_counts.keys():
                if sample[node_i] == 1 and sample[node_j] == 1:
                    joint_counts[(node_i, node_j)] += 1

        joint_probabilities = {}
        min_prob = 1  # initialize the min_prob to 1

        # First, calculate the probabilities for the existing links
        for (node_i, node_j), count in joint_counts.items():
            if (node_i, node_j) in self.p or (node_j, node_i) in self.p:
                prob = count / total_samples
                joint_probabilities[(node_i, node_j)] = prob
                joint_probabilities[(node_j, node_i)] = prob  # update for bidirectional link
                if prob < min_prob:
                    min_prob = prob  # update the minimum probability

        # Now, calculate the probabilities for the non-existing links using the Gumbel distribution
        for (node_i, node_j), count in joint_counts.items():
            if (node_i, node_j) not in self.p and (node_j, node_i) not in self.p:
                # generate a random value from a Gumbel distribution
                gumbel_noise = np.random.gumbel()
                # rescale the gumbel noise to be in [0, min_prob)
                # scaled_gumbel_noise = min_prob * (gumbel_noise - np.min(gumbel_noise)) / (np.max(gumbel_noise) - np.min(gumbel_noise))
                joint_probabilities[(node_i, node_j)] = gumbel_noise
                joint_probabilities[(node_j, node_i)] = gumbel_noise  # update for bidirectional link

        return joint_probabilities
class NNModel(nn.Module):
    def __init__(self, input_size, output_size, h_size, h_layers):
        super(NNModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.h_size = h_size
        self.h_layers = h_layers
        layers = [nn.Linear(self.input_size, self.h_size), nn.ReLU()]
        for _ in range(h_layers - 1):
            layers += [nn.Linear(self.h_size, self.h_size), nn.ReLU()]
        layers.append(nn.Linear(self.h_size, self.output_size))
        self.nn = nn.Sequential(*layers)
        self.nn.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
    def forward(self, u):
        return torch.sigmoid(self.nn(u))


class NCM:
    def __init__(self, graph, target_node, learning_rate, h_size, h_layers, data, role_id):
        self.graph = graph
        self.h_size = h_size
        self.h_layers = h_layers
        self.learning_rate = learning_rate
        self.target_node = target_node
        self.states = {graph.target_node: torch.tensor([graph.labels[target_node]], dtype=torch.float32)}

        self.u_i = {}
        self.u_ij = {}
        for v in graph.one_hop_neighbors | graph.two_hop_neighbors:
            try:
                # Access data directly using the node name as the column name
                self.u_i[v] = torch.tensor([data[v].iloc[0]], dtype=torch.float32)
                self.u_ij[v] = torch.tensor([data[v].iloc[0]], dtype=torch.float32)
            except KeyError:
                v_index = graph.v.index(v)
                # print(f"Warning: Node {v} not found in data. Using role_id value.")
                self.u_i[v] = torch.tensor([float(role_id[v_index])], dtype=torch.float32)
                self.u_ij[v] = torch.tensor([float(role_id[v_index])], dtype=torch.float32)

        self.u = torch.cat(list(self.states.values()) + list(self.u_i.values()) + list(self.u_ij.values()), dim=0)
        self.model = NNModel(input_size=len(self.u), output_size=1, h_size=h_size, h_layers=h_layers)
        self.ratio = len(self.graph.p) / len(self.graph.v)
    def add_gaussian_noise(self, tensor, mean=0.0, std=0.01):
        noise = torch.randn(tensor.size()) * std + mean
        return torch.clamp(tensor + noise, 0, 1)

    def ncm_forward(self, add_noise=False):
        if add_noise:
            for k in self.u_i:
                self.u_i[k] = self.add_gaussian_noise(self.u_i[k])
            for k in self.u_ij:
                self.u_ij[k] = self.add_gaussian_noise(self.u_ij[k])
            self.u = torch.cat(list(self.states.values()) + list(self.u_i.values()) + list(self.u_ij.values()), dim=0)
        f = self.model.forward(self.u)  # Pass self.u here
        return torch.sigmoid(f)
def calculate_prob(graph, f, target_node):
    nodes_n1_n2 = graph.fn[target_node] | graph.sn[target_node]
    if not nodes_n1_n2:
        return 0.0  # Return 0 if no neighbors are present
    sum_prob = 0.0
    for v_j in nodes_n1_n2:
        # product = 1.0
        if (target_node, v_j) in graph.p or (v_j, target_node) in graph.p:
            # product *= f
            sum_prob += f.item()
    probability = sum_prob / len(nodes_n1_n2) if nodes_n1_n2 else 0.0
    # print(f"Debug: Calculated Probability for Node {target_node}: {probability}")
    return probability
def calculate_expected_prob(cg, P_do,label_probs):
    expected_value = 0.0
    for y, y_prob in label_probs.items():
        inner_sum = 0.0
        for v_i in cg.one_hop_neighbors:
            inner_sum += P_do
        expected_value += y_prob * inner_sum
        return expected_value / len(cg.one_hop_neighbors) if cg.one_hop_neighbors else 0.0


def compute_probability_of_node_label(cg, target_node, role_id):
    unique_labels = np.unique(role_id)
    num_nodes = len(role_id)
    all_combinations = product(unique_labels, repeat=num_nodes)
    label_probabilities = {label: 0 for label in unique_labels}

    target_index = cg.v.index(target_node)

    for combination in all_combinations:
        current_label = combination[target_index]
        label_probabilities[current_label] += 1

    total_combinations = len(unique_labels) ** num_nodes
    for label in label_probabilities:
        label_probabilities[label] /= total_combinations

    return label_probabilities


def train(cg, learning_rate, h_size, h_layers, num_epochs, data, role_id, target_node):
    cg.target_node, cg.one_hop_neighbors, cg.two_hop_neighbors, cg.out_of_neighborhood = cg.categorize_neighbors(
        target_node=target_node)
    ncm = NCM(cg, target_node, learning_rate=learning_rate, h_size=h_size, h_layers=h_layers, data=data,
              role_id=role_id)
    print(f" Target node: {target_node}")
    print(f"One-hop neighbors: {cg.one_hop_neighbors}")
    print(f"Two-hop neighbors: {cg.two_hop_neighbors}")
    optimizer = optim.Adam(ncm.model.parameters(), lr=ncm.learning_rate)
    new_v = {cg.target_node}.union(cg.one_hop_neighbors)
    loss_history = []

    target_index = cg.v.index(target_node)

    for i in range(num_epochs):
        f = ncm.ncm_forward(add_noise=True)
        P_do = calculate_prob(cg, f, cg.target_node)
        label_probs = compute_probability_of_node_label(cg, cg.target_node, role_id)
        expected_p = calculate_expected_prob(cg, P_do, label_probs)
        expected_p_tensor = torch.tensor([expected_p], dtype=torch.float32) if isinstance(expected_p,
                                                                                          float) else expected_p
        output = (expected_p_tensor.clone().detach() >= 0.05).float()

        loss = torch.nn.functional.binary_cross_entropy(f.view(1),
                                                        torch.tensor([role_id[target_index]], dtype=torch.float).view(
                                                            1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())


    return loss_history, loss, ncm.model.state_dict(), expected_p, output, new_v

def print_expected_p_for_each_node(models):
    for node, model_info in models.items():
        expected_p = model_info['expected_p']
        print(f"Node: {node}, Expected Probability: {expected_p}")
def alg_2(Graph, num_epochs, data, role_id):

    if num_epochs is None:
        num_epochs = 100  # Default value, change as necessary
    models = {}
    for node in Graph.v:
        Graph.target_node = node

        loss_history, total_loss, model, expected_p, output, new_v = train(
            Graph, 0.005, 32, 2, num_epochs, data, role_id, node
        )

        models[node] = {
            'model': model,
            'expected_p': expected_p,
            'total_loss': total_loss,
            'output': output,
            'new_v': new_v,
            'loss_history': loss_history
        }
        # print(f"Node: {models[node]}")
    print_expected_p_for_each_node(models)
    best_node = max(models.keys(), key=lambda k: models[k]['expected_p'])
    best_model = models[best_node]['model']
    best_total_loss = models[best_node]['total_loss']
    best_expected_p = models[best_node]['expected_p']
    best_output = models[best_node]['output']
    best_new_v = models[best_node]['new_v']

    print(best_new_v, best_node)
    expressivity = {}
    for node, model_info in models.items():
        expressivity[node] = model_info['expected_p']
    print("\nExpressivity for each variable:")
    for node, exp in sorted(expressivity.items(), key=lambda x: x[1], reverse=True):
        print(f"{node}: {exp:.4f}")
    return models, best_total_loss, best_model, best_expected_p, best_output, best_new_v, best_node

def from_networkx_to_torch(graph, role_id):
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    x = torch.tensor(role_id, dtype=torch.float).view(-1, 1)  # Using role_id as node features
    return Data(x=x, edge_index=edge_index)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        ))

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


