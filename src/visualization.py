import matplotlib.pyplot as plt
import networkx as nx

def visualize(G):
    """Visualize the graph structure."""
    plt.figure(figsize=(15, 10), dpi=300)
    pos = nx.spring_layout(G)
    
    # Define color map
    color_map = {
        1: "#FF4444",  # Light red
        2: "#4444FF",  # Light blue
        3: "#44AA44"   # Light green
    }
    
    # Draw nodes
    node_colors = [color_map[G.nodes[n]['label']] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=600,
                          edgecolors='black',
                          linewidths=1)
    
    # Draw edges
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos,
                          width=edge_weights,
                          alpha=0.4,
                          edge_color='gray')
    
    # Add labels
    label_pos = {k: (v[0], v[1] + 0.05) for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos,
                          font_size=10,
                          font_weight='bold',
                          verticalalignment='bottom')
    
    plt.title("Variable Influence Graph\nRed: MCI stay, Blue: MCI2N, Green: MCI2D",
              fontsize=16,
              fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.margins(x=0.2, y=0.2)
    plt.savefig('figure1.png')
    plt.close()

def visualize_results(G, expressivity, y):
    """Visualize node expressivity distribution."""
    plt.figure(figsize=(12, 8), dpi=300)
    sorted_items = sorted(expressivity.items(), key=lambda x: x[1], reverse=True)
    sorted_nodes, sorted_values = zip(*sorted_items)
    
    colors = ['red' if G.nodes[node]['label'] == 1 else 'blue' 
              for node in sorted_nodes]
    
    plt.bar(range(len(sorted_nodes)), sorted_values, color=colors)
    plt.xticks(range(len(sorted_nodes)), sorted_nodes, rotation=90)
    plt.xlabel('Nodes')
    plt.ylabel('Expressivity')
    plt.title('Node Expressivity Distribution (Red: Group 1, Blue: Others)')
    plt.tight_layout()
    plt.savefig('expressivity_distribution.png')
    plt.close()
