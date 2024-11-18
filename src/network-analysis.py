import networkx as nx
from community import community_louvain

def comprehensive_network_analysis(G, target_node='sexc'):
    """Perform comprehensive network analysis for a target node."""
    results = {
        'direct_connections': [],
        'two_hop_neighbors': [],
        'centrality_measures': {},
        'paths': {},
        'community': {},
        'similar_nodes': [],
        'edge_weights': {}
    }
    
    # 1. Network Structure Analysis
    results['direct_connections'] = list(G.neighbors(target_node))
    two_hop_neighbors = set(
        n for nbr in results['direct_connections'] 
        for n in G.neighbors(nbr)
    ) - set(results['direct_connections']) - {target_node}
    results['two_hop_neighbors'] = list(two_hop_neighbors)
    
    # 2. Centrality Measures
    results['centrality_measures'] = {
        'degree': nx.degree_centrality(G)[target_node],
        'betweenness': nx.betweenness_centrality(G)[target_node],
        'closeness': nx.closeness_centrality(G)[target_node]
    }
    
    # 3. Path Analysis
    important_nodes = sorted(
        nx.degree_centrality(G),
        key=nx.degree_centrality(G).get,
        reverse=True
    )[:5]
    
    for node in important_nodes:
        if node != target_node:
            results['paths'][node] = nx.shortest_path(
                G, source=target_node, target=node
            )
    
    # 4. Community Detection
    communities = community_louvain.best_partition(G)
    target_community = communities[target_node]
    results['community'] = {
        'id': target_community,
        'members': [node for node, com in communities.items() 
                   if com == target_community]
    }
    
    # 5. Similar Nodes Analysis
    centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    
    similar_nodes = sorted(
        centrality.items(),
        key=lambda x: abs(x[1] - centrality[target_node])
    )[:6]
    
    for node, cent in similar_nodes:
        if node != target_node:
            results['similar_nodes'].append({
                'node': node,
                'centrality': cent,
                'betweenness': betweenness[node],
                'closeness': closeness[node]
            })
    
    # 6. Edge Weight Analysis
    if 'weight' in G.edges[list(G.edges())[0]]:
        for neighbor in G.neighbors(target_node):
            results['edge_weights'][neighbor] = G[target_node][neighbor]['weight']
    
    return results

def print_analysis_results(results):
    """Print the analysis results in a formatted way."""
    print(f"Network Analysis Results:")
    print("\n1. Direct Connections:")
    print(f"Number of direct connections: {len(results['direct_connections'])}")
    print(f"Nodes: {', '.join(results['direct_connections'])}")
    
    print