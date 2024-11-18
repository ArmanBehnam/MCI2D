import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
from itertools import product

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
                self.u_i[v] = torch.tensor([data[v].iloc[0]], dtype=torch.float32)
                self.u_ij[v] = torch.tensor([data[v].iloc[0]], dtype=torch.float32)
            except KeyError:
                v_index = graph.v.index(v)
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
        f = self.model.forward(self.u)
        return torch.sigmoid(f)
