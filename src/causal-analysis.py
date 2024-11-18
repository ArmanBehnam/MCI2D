import torch
import numpy as np
from models.ncm import NCM
from collections import deque

def calculate_prob(graph, f, target_node):
    """Calculate probability based on graph structure."""
    nodes_n1_n2 = graph.fn[target_node] | graph.sn[target_node]
    if not nodes_n1_n2:
        return 0.0
    
    sum_prob = 0.0
    for v_j in nodes_n1_n2:
        if (target_node, v_j) in graph.p or (v_j, target_node) in graph.p:
            sum_prob += f.item()
    
    probability = sum_prob / len(nodes_n1_n2) if nodes_n1_n2 else 0.0
    return probability

def calculate_expected_prob(cg, P_do, label_probs):
    """Calculate expected probability."""
    expected_value = 0.0
    for y, y_prob in label_probs.items():
        inner_sum = 0.0
        for v_i in cg.one_hop_neighbors:
            inner_sum += P_do
        expected_value += y_prob * inner_sum
    
    return expected_value / len(cg.one_hop_neighbors) if cg.one_hop_neighbors else 0.0

def train(cg, learning_rate, h_size, h_layers, num_epochs, data, role_id, target_node):
    """Train the causal model."""
    # Initialize neighborhoods
    cg.target_node, cg.one_hop_neighbors, cg.two_hop_neighbors, cg.out_of_neighborhood = \
        cg.categorize_neighbors(target_node=target_node)
    
    # Create and train NCM
    ncm = NCM(cg, target_node, learning_rate, h_size, h_layers, data, role_id)
    optimizer = torch.optim.Adam(ncm.model.parameters(), lr=ncm.learning_rate)
    new_v = {cg.target_node}.union(cg.one_hop_neighbors)
    loss_history = []
    
    target_index = cg.v.index(target_node)
    
    # Training loop
    for i in range(num_epochs):
        f = ncm.ncm_forward(add_noise=True)
        P_do = calculate_prob(cg, f, cg.target_node)
        label_probs = compute_probability_of_node_label(cg, cg.target_node, role_id)
        expected_p = calculate_expected_prob(cg, P_do, label_probs)
        
        expected_p_tensor = torch.tensor([expected_p], dtype=torch.float32) \
            if isinstance(expected_p, float) else expected_p
        output = (expected_p_tensor.clone().detach() >= 0.05).float()
        
        loss = torch.nn.functional.binary_cross_entropy(
            f.view(1), 
            torch.tensor([role_id[target_index]], dtype=torch.float).view(1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    
    return loss_history, loss, ncm.model.state_dict(), expected_p, output, new_v

def implementation(G, y, num_epochs=None):
    """Implement the causal analysis pipeline."""
    if num_epochs is None:
        num_epochs = 100
    
    # Convert labels
    role_id = (y == 1).astype(int)
    
    # Create CausalGraph object
    cg = CausalGraph(V=G.nodes(), path=G.edges())
    
    # Create data DataFrame
    data = pd.DataFrame({node: [G.nodes[node]['mi_score_1']] 
                        for node in G.nodes()})
    
    # Run algorithm
    models, best_total_loss, best_model, best_expected_p, \
    best_output, best_new_v, best_node = alg_2(
        cg, num_epochs=num_epochs, data=data, role_id=role_id
    )
    
    # Calculate expressivity
    expressivity = {node: model_info['expected_p'] 
                   for node, model_info in models.items()}
    
    return (models, best_total_loss, best_model, best_expected_p,
            best_output, best_new_v, best_node, expressivity)
