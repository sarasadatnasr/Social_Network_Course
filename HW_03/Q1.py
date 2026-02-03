import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random
from itertools import combinations, permutations
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Create output directory for figures
output_dir = Path('./HW_03/Q1/')
output_dir.mkdir(exist_ok=True)

# =============================================================================
# PART 1: SIGN PREDICTION (CORRECTED)
# =============================================================================
def load_balance_graph(filepath):
    try:
        df = pd.read_csv(filepath)
        # Standardize column names
        if 'source' in df.columns and 'target' in df.columns:
            df = df.rename(columns={'source': 'u', 'target': 'v'})
        elif 'node1' in df.columns and 'node2' in df.columns:
            df = df.rename(columns={'node1': 'u', 'node2': 'v'})
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def predict_missing_signs_corrected(df):
    """
    Corrected sign prediction using balance theory:
    - In balanced networks, triangles must be +++ or +--
    - Predict missing signs based on existing triangles
    """
    if df is None:
        return None
    
    G = nx.Graph()
    
    # Add edges with known signs
    for _, row in df.iterrows():
        if not pd.isna(row['sign']):
            G.add_edge(row['u'], row['v'], sign=int(row['sign']))
        else:
            G.add_edge(row['u'], row['v'], sign=None)
    
    def predict_sign_balance_theory(u, v):
        """Predict sign using balance theory and common neighbors"""
        common_neighbors = list(set(G.neighbors(u)) & set(G.neighbors(v)))
        
        predictions = []
        for w in common_neighbors:
            if G.has_edge(u, w) and G.has_edge(v, w):
                sign_uw = G[u][w].get('sign')
                sign_vw = G[v][w].get('sign')
                
                if sign_uw is not None and sign_vw is not None:
                    # In balanced triangle: sign(u,v) = sign(u,w) * sign(v,w)
                    predicted = sign_uw * sign_vw
                    predictions.append(predicted)
        
        if predictions:
            # Take majority vote
            avg_pred = np.mean(predictions)
            return 1 if avg_pred >= 0 else -1
        else:
            # No common neighbors with known signs
            # Check global positive/negative ratio
            positive_count = sum(1 for _, _, d in G.edges(data=True) if d.get('sign') == 1)
            negative_count = sum(1 for _, _, d in G.edges(data=True) if d.get('sign') == -1)
            return 1 if positive_count >= negative_count else -1
    
    predictions = []
    for _, row in df.iterrows():
        u, v, sign = row['u'], row['v'], row['sign']
        
        if pd.isna(sign):
            pred_sign = predict_sign_balance_theory(u, v)
            predictions.append((u, v, pred_sign))
        else:
            predictions.append((u, v, int(sign)))
    
    result_df = pd.DataFrame(predictions, columns=['u', 'v', 'predicted_sign'])
    return result_df

def visualize_sign_predictions(original_df, predicted_df, network_name="part1"):
    """Visualize original vs predicted signs and save as PNG"""
    if original_df is None or predicted_df is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original graph (known signs only)
    G_original = nx.Graph()
    known_edges = []
    for _, row in original_df.iterrows():
        if not pd.isna(row['sign']):
            G_original.add_edge(row['u'], row['v'], sign=int(row['sign']))
            known_edges.append((row['u'], row['v'], int(row['sign'])))
    
    if len(G_original.nodes()) == 0:
        print("No known signs to visualize")
        return
    
    pos = nx.spring_layout(G_original, seed=42)
    
    # Plot original
    ax = axes[0]
    nx.draw_networkx_nodes(G_original, pos, ax=ax, node_size=300, node_color='lightblue')
    
    # Draw edges by sign
    pos_edges = [(u, v) for u, v, s in known_edges if s == 1]
    neg_edges = [(u, v) for u, v, s in known_edges if s == -1]
    
    if pos_edges:
        nx.draw_networkx_edges(G_original, pos, edgelist=pos_edges, ax=ax, 
                              edge_color='green', width=2, label='Positive')
    if neg_edges:
        nx.draw_networkx_edges(G_original, pos, edgelist=neg_edges, ax=ax,
                              edge_color='red', width=2, style='dashed', label='Negative')
    
    nx.draw_networkx_labels(G_original, pos, ax=ax, font_size=8)
    ax.set_title("Original Graph (Known Signs Only)")
    ax.legend()
    ax.axis('off')
    
    # Plot predicted
    ax = axes[1]
    G_predicted = nx.Graph()
    predicted_edges = []
    for _, row in predicted_df.iterrows():
        G_predicted.add_edge(row['u'], row['v'], sign=row['predicted_sign'])
        predicted_edges.append((row['u'], row['v'], row['predicted_sign']))
    
    nx.draw_networkx_nodes(G_predicted, pos, ax=ax, node_size=300, node_color='lightblue')
    
    pos_edges = [(u, v) for u, v, s in predicted_edges if s == 1]
    neg_edges = [(u, v) for u, v, s in predicted_edges if s == -1]
    
    if pos_edges:
        nx.draw_networkx_edges(G_predicted, pos, edgelist=pos_edges, ax=ax,
                              edge_color='green', width=2, label='Positive')
    if neg_edges:
        nx.draw_networkx_edges(G_predicted, pos, edgelist=neg_edges, ax=ax,
                              edge_color='red', width=2, style='dashed', label='Negative')
    
    # Highlight predicted edges
    if 'sign' in original_df.columns:
        unknown_in_original = [(row['u'], row['v']) for _, row in original_df.iterrows() 
                              if pd.isna(row['sign'])]
        if unknown_in_original:
            nx.draw_networkx_edges(G_predicted, pos, edgelist=unknown_in_original, ax=ax,
                                  edge_color='orange', width=3, alpha=0.7, label='Predicted')
    
    nx.draw_networkx_labels(G_predicted, pos, ax=ax, font_size=8)
    ax.set_title("Graph with Predicted Signs")
    ax.legend()
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    filename = output_dir / f"{network_name}_sign_predictions.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filename}")
    plt.close(fig)

# =============================================================================
# PART 2: BALANCE TEST (CORRECTED)
# =============================================================================
def load_signed_network(filepath):
    try:
        df = pd.read_csv(filepath)
        # Standardize column names
        if 'source' in df.columns and 'target' in df.columns:
            df = df.rename(columns={'source': 'u', 'target': 'v'})
        elif 'node1' in df.columns and 'node2' in df.columns:
            df = df.rename(columns={'node1': 'u', 'node2': 'v'})
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def check_balance_corrected(df):
    """Corrected balance check with proper contradiction detection"""
    if df is None:
        return None
    
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['u'], row['v'], sign=int(row['sign']))
    
    # Step 1: Create super-nodes (connected components via positive edges)
    positive_G = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d['sign'] == 1:
            positive_G.add_edge(u, v)
    
    super_nodes = {}
    super_node_id = 0
    visited = set()
    
    for node in G.nodes():
        if node not in visited and node in positive_G:
            component = []
            queue = deque([node])
            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    for neighbor in positive_G.neighbors(current):
                        if neighbor not in visited:
                            queue.append(neighbor)
            
            for n in component:
                super_nodes[n] = super_node_id
            super_node_id += 1
    
    # Handle isolated nodes (no positive connections)
    for node in G.nodes():
        if node not in super_nodes:
            super_nodes[node] = super_node_id
            super_node_id += 1
    
    # Step 2: Build reduced graph and check for contradictions
    reduced_edges = defaultdict(list)
    contradictions = []
    
    for u, v, d in G.edges(data=True):
        super_u = super_nodes[u]
        super_v = super_nodes[v]
        
        if super_u != super_v:
            # Check for contradictions in reduced graph
            if (super_v, super_u) in reduced_edges:
                existing_signs = reduced_edges[(super_v, super_u)]
                if d['sign'] not in existing_signs and -d['sign'] in existing_signs:
                    contradictions.append(((super_u, super_v), d['sign'], existing_signs))
            reduced_edges[(super_u, super_v)].append(d['sign'])
    
    # Step 3: Check if balanced
    is_balanced = len(contradictions) == 0
    
    # Build reduced graph for visualization
    reduced_graph = nx.Graph()
    for (u, v), signs in reduced_edges.items():
        # Take majority sign if multiple edges
        avg_sign = np.mean(signs)
        sign = 1 if avg_sign >= 0 else -1
        reduced_graph.add_edge(u, v, sign=sign, original_signs=signs)
    
    return {
        'balanced': is_balanced,
        'super_nodes': super_nodes,
        'reduced_graph': reduced_graph,
        'contradictions': contradictions,
        'num_super_nodes': len(set(super_nodes.values()))
    }

def visualize_balance_test(df, result, network_name):
    """Visualize the balance test results and save as PNG"""
    if df is None or result is None:
        return
    
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['u'], row['v'], sign=int(row['sign']))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Original network with super-node colors
    ax = axes[0]
    pos = nx.spring_layout(G, seed=42)
    
    # Color nodes by super-node
    super_node_colors = {}
    colors = plt.cm.tab20(np.linspace(0, 1, result['num_super_nodes']))
    
    for node, super_id in result['super_nodes'].items():
        super_node_colors[node] = colors[super_id % len(colors)]
    
    node_colors = [super_node_colors[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=400, alpha=0.8)
    
    # Draw edges by sign
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == 1]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == -1]
    
    if pos_edges:
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, ax=ax,
                              edge_color='green', width=2, label='Positive')
    if neg_edges:
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, ax=ax,
                              edge_color='red', width=2, style='dashed', label='Negative')
    
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    
    ax.set_title(f"Original Network\n(Balanced: {result['balanced']})")
    ax.legend(loc='upper left')
    ax.axis('off')
    
    # Plot 2: Reduced graph
    ax = axes[1]
    if result['reduced_graph'].number_of_nodes() > 0:
        reduced_G = result['reduced_graph']
        pos_reduced = nx.spring_layout(reduced_G, seed=42)
        
        # Draw super-nodes (size proportional to number of original nodes)
        super_node_sizes = defaultdict(int)
        for node, super_id in result['super_nodes'].items():
            super_node_sizes[super_id] += 1
        
        node_sizes = [super_node_sizes[node] * 200 for node in reduced_G.nodes()]
        
        nx.draw_networkx_nodes(reduced_G, pos_reduced, ax=ax, 
                              node_color=colors[:len(reduced_G.nodes())],
                              node_size=node_sizes, alpha=0.8)
        
        # Draw edges in reduced graph (should all be negative if balanced)
        edges = list(reduced_G.edges(data=True))
        edge_colors = []
        for u, v, d in edges:
            if d.get('sign', -1) == 1:
                edge_colors.append('green')
            else:
                edge_colors.append('red')
        
        if edges:
            nx.draw_networkx_edges(reduced_G, pos_reduced, ax=ax,
                                  edge_color=edge_colors, width=2)
        
        # Label super-nodes with their member count
        labels = {node: f"S{node}\n({super_node_sizes[node]} nodes)" 
                 for node in reduced_G.nodes()}
        nx.draw_networkx_labels(reduced_G, pos_reduced, ax=ax, 
                               labels=labels, font_size=8)
        
        ax.set_title(f"Reduced Graph ({len(reduced_G.nodes())} super-nodes)")
    else:
        ax.text(0.5, 0.5, "No reduced graph\n(All nodes in one super-node)",
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save figure
    filename = output_dir / f"part2_balance_{network_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filename}")
    plt.close(fig)

# =============================================================================
# PART 3: CLUSTERABILITY (CORRECTED) - WITH VISUALIZATION
# =============================================================================
def detect_clusters_corrected(df):
    """Corrected cluster detection for weakly balanced networks"""
    if df is None:
        return None
    
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['u'], row['v'], sign=int(row['sign']))
    
    # Use correlation clustering approach
    # Start with each node in its own cluster
    clusters = {}
    node_to_cluster = {}
    
    for i, node in enumerate(G.nodes()):
        clusters[i] = {node}
        node_to_cluster[node] = i
    
    # Merge clusters with strong positive connections
    changed = True
    max_iterations = 100
    iteration = 0
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        # Try to merge clusters
        cluster_pairs = list(combinations(list(clusters.keys()), 2))
        for c1, c2 in cluster_pairs:
            if c1 not in clusters or c2 not in clusters:
                continue
            
            # Calculate connection quality between clusters
            positive_connections = 0
            negative_connections = 0
            
            for node1 in clusters[c1]:
                for node2 in clusters[c2]:
                    if G.has_edge(node1, node2):
                        if G[node1][node2]['sign'] == 1:
                            positive_connections += 1
                        else:
                            negative_connections += 1
            
            # Merge if positive connections dominate
            if positive_connections > negative_connections and negative_connections == 0:
                # Merge c2 into c1
                for node in clusters[c2]:
                    node_to_cluster[node] = c1
                clusters[c1].update(clusters[c2])
                del clusters[c2]
                changed = True
                break  # Restart merging process
    
    # Refine clusters by moving problematic nodes
    for _ in range(50):
        moved = False
        
        for node in G.nodes():
            current_cluster = node_to_cluster[node]
            
            # Calculate node's connection to each cluster
            cluster_scores = defaultdict(int)
            
            for neighbor in G.neighbors(node):
                neighbor_cluster = node_to_cluster[neighbor]
                edge_sign = G[node][neighbor]['sign']
                
                # Positive edges should be within clusters, negative between
                if edge_sign == 1:
                    cluster_scores[neighbor_cluster] += 1
                else:
                    cluster_scores[neighbor_cluster] -= 1
            
            # Find best cluster (exclude current if not the best)
            if cluster_scores:
                best_cluster, best_score = max(cluster_scores.items(), key=lambda x: x[1])
                
                if best_cluster != current_cluster and best_score > cluster_scores[current_cluster]:
                    # Move node
                    clusters[current_cluster].remove(node)
                    if not clusters[current_cluster]:
                        del clusters[current_cluster]
                    
                    clusters[best_cluster].add(node)
                    node_to_cluster[node] = best_cluster
                    moved = True
        
        if not moved:
            break
    
    # Convert to list format
    cluster_list = []
    for cluster_id, nodes in clusters.items():
        cluster_list.append(list(nodes))
    
    # Check for violations
    violations = []
    for u, v, d in G.edges(data=True):
        u_cluster = node_to_cluster[u]
        v_cluster = node_to_cluster[v]
        
        if d['sign'] == -1 and u_cluster == v_cluster:
            violations.append((u, v))
        elif d['sign'] == 1 and u_cluster != v_cluster:
            violations.append((u, v))
    
    return {
        'clusters': cluster_list,
        'node_to_cluster': node_to_cluster,
        'cluster_sizes': [len(c) for c in cluster_list],
        'violations': len(violations),
        'is_weakly_balanced': len(violations) == 0
    }

def visualize_clusters_corrected(df, clusters_data, network_name):
    """Visualize network with cluster structure and save as PNG"""
    if df is None or clusters_data is None:
        return
    
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['u'], row['v'], sign=int(row['sign']))
    
    clusters = clusters_data['clusters']
    node_to_cluster = clusters_data['node_to_cluster']
    
    fig = plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    
    # Assign colors to clusters
    colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))
    
    # Create mapping from node to color
    node_colors = {}
    for i, cluster in enumerate(clusters):
        for node in cluster:
            node_colors[node] = colors[i]
    
    # Draw nodes by cluster
    nx.draw_networkx_nodes(G, pos, 
                          node_color=[node_colors[node] for node in G.nodes()],
                          node_size=500, alpha=0.8)
    
    # Draw edges with different styles for positive/negative
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == 1]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == -1]
    
    # Draw positive edges (solid green)
    if positive_edges:
        nx.draw_networkx_edges(G, pos, edgelist=positive_edges, 
                              edge_color='green', width=2, alpha=0.7,
                              label='Positive edges')
    
    # Draw negative edges (dashed red)
    if negative_edges:
        nx.draw_networkx_edges(G, pos, edgelist=negative_edges, 
                              edge_color='red', width=1.5, alpha=0.5, 
                              style='dashed', label='Negative edges')
    
    # Highlight violations if any
    violations = []
    for u, v, d in G.edges(data=True):
        u_cluster = node_to_cluster[u]
        v_cluster = node_to_cluster[v]
        
        if (d['sign'] == -1 and u_cluster == v_cluster) or (d['sign'] == 1 and u_cluster != v_cluster):
            violations.append((u, v))
    
    if violations:
        nx.draw_networkx_edges(G, pos, edgelist=violations, 
                              edge_color='orange', width=3, alpha=0.8,
                              label=f'Violations ({len(violations)})')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Add cluster information
    cluster_info = "\n".join([f"Cluster {i}: {len(c)} nodes" for i, c in enumerate(clusters)])
    plt.figtext(0.02, 0.02, cluster_info, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    title = f"Network {network_name} Clustering\nWeakly Balanced: {clusters_data['is_weakly_balanced']}, Violations: {len(violations)}"
    plt.title(title, fontsize=14, pad=20)
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    filename = output_dir / f"part3_clusters_{network_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filename}")
    plt.close(fig)

# =============================================================================
# PART 4: LINE INDEX (CORRECTED)
# =============================================================================
def load_line_index_network(filepath):
    try:
        df = pd.read_csv(filepath)
        # Standardize column names
        if 'source' in df.columns and 'target' in df.columns:
            df = df.rename(columns={'source': 'u', 'target': 'v'})
        elif 'node1' in df.columns and 'node2' in df.columns:
            df = df.rename(columns={'node1': 'u', 'node2': 'v'})
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def compute_line_index_corrected(df, assignments, alpha=0.5):
    """More efficient line index computation"""
    P = 0  # Positive edges between clusters
    N = 0  # Negative edges within clusters
    
    for _, row in df.iterrows():
        u, v, sign = row['u'], row['v'], int(row['sign'])
        
        if assignments.get(u) != assignments.get(v):
            if sign == 1:
                P += 1
        else:
            if sign == -1:
                N += 1
    
    line_index = alpha * P + (1 - alpha) * N
    return line_index, P, N

def random_clustering(df, k=4):
    """Assign nodes randomly to clusters"""
    nodes = list(set(df['u'].tolist() + df['v'].tolist()))
    assignments = {}
    for node in nodes:
        assignments[node] = random.randint(0, k-1)
    return assignments

def improve_clustering_optimized(df, initial_assignments, alpha=0.5, iterations=1000):
    """Optimized clustering improvement with simulated annealing"""
    assignments = initial_assignments.copy()
    current_li, _, _ = compute_line_index_corrected(df, assignments, alpha)
    best_assignments = assignments.copy()
    best_li = current_li
    
    k = len(set(assignments.values()))
    nodes = list(assignments.keys())
    
    # Temperature for simulated annealing
    temperature = 1.0
    cooling_rate = 0.995
    
    for iteration in range(iterations):
        # Randomly choose a move
        move_type = random.choice(['move', 'swap'])
        
        if move_type == 'move':
            # Move a node to a different cluster
            node = random.choice(nodes)
            old_cluster = assignments[node]
            new_cluster = random.choice([c for c in range(k) if c != old_cluster])
            
            assignments[node] = new_cluster
            new_li, _, _ = compute_line_index_corrected(df, assignments, alpha)
            
            # Accept move based on simulated annealing
            delta = new_li - current_li
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_li = new_li
                if new_li < best_li:
                    best_li = new_li
                    best_assignments = assignments.copy()
            else:
                assignments[node] = old_cluster
        
        else:  # swap
            # Swap two nodes between clusters
            if len(nodes) >= 2:
                node1, node2 = random.sample(nodes, 2)
                if assignments[node1] != assignments[node2]:
                    # Swap clusters
                    cluster1, cluster2 = assignments[node1], assignments[node2]
                    assignments[node1] = cluster2
                    assignments[node2] = cluster1
                    
                    new_li, _, _ = compute_line_index_corrected(df, assignments, alpha)
                    
                    delta = new_li - current_li
                    if delta < 0 or random.random() < np.exp(-delta / temperature):
                        current_li = new_li
                        if new_li < best_li:
                            best_li = new_li
                            best_assignments = assignments.copy()
                    else:
                        assignments[node1] = cluster1
                        assignments[node2] = cluster2
        
        # Cool down
        temperature *= cooling_rate
        
        if iteration % 100 == 0 and temperature < 0.01:
            break
    
    return best_assignments, best_li

def visualize_line_index_results(df, random_assignments, improved_assignments, random_li, improved_li):
    """Visualize clustering results for line index and save as PNG"""
    if df is None:
        return
    
    # Create graph
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['u'], row['v'], sign=int(row['sign']))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    pos = nx.spring_layout(G, seed=42)
    
    # Plot 1: Random clustering
    ax = axes[0]
    
    # Get cluster assignments for visualization
    unique_clusters = len(set(random_assignments.values()))
    colors = plt.cm.tab10(np.linspace(0, 1, unique_clusters))
    
    node_colors = []
    for node in G.nodes():
        if node in random_assignments:
            node_colors.append(colors[random_assignments[node] % len(colors)])
        else:
            node_colors.append('gray')
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=300)
    
    # Draw edges by sign
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == 1]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == -1]
    
    if pos_edges:
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, ax=ax,
                              edge_color='green', width=1.5, alpha=0.7, label='Positive')
    if neg_edges:
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, ax=ax,
                              edge_color='red', width=1.5, alpha=0.7, style='dashed', label='Negative')
    
    ax.set_title(f"Random Clustering\nLine Index: {random_li:.4f}")
    ax.legend(loc='upper left')
    ax.axis('off')
    
    # Plot 2: Improved clustering
    ax = axes[1]
    
    unique_clusters = len(set(improved_assignments.values()))
    colors = plt.cm.tab10(np.linspace(0, 1, unique_clusters))
    
    node_colors = []
    for node in G.nodes():
        if node in improved_assignments:
            node_colors.append(colors[improved_assignments[node] % len(colors)])
        else:
            node_colors.append('gray')
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=300)
    
    if pos_edges:
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, ax=ax,
                              edge_color='green', width=1.5, alpha=0.7, label='Positive')
    if neg_edges:
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, ax=ax,
                              edge_color='red', width=1.5, alpha=0.7, style='dashed', label='Negative')
    
    ax.set_title(f"Improved Clustering\nLine Index: {improved_li:.4f}")
    ax.legend(loc='upper left')
    ax.axis('off')
    
    # Plot 3: Comparison bar chart
    ax = axes[2]
    
    # Calculate P and N for both clusterings
    _, P_rand, N_rand = compute_line_index_corrected(df, random_assignments, alpha=0.5)
    _, P_imp, N_imp = compute_line_index_corrected(df, improved_assignments, alpha=0.5)
    
    categories = ['P (Pos between)', 'N (Neg within)', 'Line Index']
    random_vals = [P_rand, N_rand, random_li]
    improved_vals = [P_imp, N_imp, improved_li]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, random_vals, width, label='Random', color='skyblue', alpha=0.8)
    ax.bar(x + width/2, improved_vals, width, label='Improved', color='lightcoral', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Clustering Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add value labels on bars
    for i, (rv, iv) in enumerate(zip(random_vals, improved_vals)):
        ax.text(i - width/2, rv + max(random_vals)/50, f'{rv:.1f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, iv + max(improved_vals)/50, f'{iv:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    filename = output_dir / "part4_line_index_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filename}")
    plt.close(fig)

# =============================================================================
# PART 5: TRANSITIVITY (CORRECTED)
# =============================================================================
def load_transitivity_network(filepath):
    try:
        df = pd.read_csv(filepath)
        # Ensure we have the right columns
        if 'node1' in df.columns and 'node2' in df.columns:
            df = df.rename(columns={'node1': 'source', 'node2': 'target'})
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_transitivity_corrected(df):
    """Corrected transitivity analysis"""
    if df is None:
        return None
    
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'])
    
    # Get all nodes
    nodes = list(G.nodes())
    
    # Count transitive triples
    transitive_triples = 0
    possible_triples = 0
    
    # Check all ordered triples
    for i in nodes:
        for j in nodes:
            if i != j and G.has_edge(i, j):
                for k in nodes:
                    if k != i and k != j and G.has_edge(j, k):
                        possible_triples += 1
                        if G.has_edge(i, k):
                            transitive_triples += 1
    
    # Compute transitivity ratio
    transitivity_ratio = transitive_triples / possible_triples if possible_triples > 0 else 0
    
    # Find missing edges to make network transitive
    edges_to_add = []
    
    # Make a copy to avoid modifying original during iteration
    G_copy = G.copy()
    
    # Check and add missing transitive edges
    for i in nodes:
        for j in nodes:
            if i != j and G_copy.has_edge(i, j):
                for k in nodes:
                    if k != i and k != j and G_copy.has_edge(j, k) and not G_copy.has_edge(i, k):
                        edges_to_add.append((i, k))
                        G_copy.add_edge(i, k)  # Add edge to check further transitivity
    
    # Remove duplicates
    edges_to_add = list(set(edges_to_add))
    
    # Verify after adding edges
    G_complete = G.copy()
    for u, v in edges_to_add:
        G_complete.add_edge(u, v)
    
    # Recompute transitivity
    transitive_after = 0
    possible_after = 0
    
    for i in nodes:
        for j in nodes:
            if i != j and G_complete.has_edge(i, j):
                for k in nodes:
                    if k != i and k != j and G_complete.has_edge(j, k):
                        possible_after += 1
                        if G_complete.has_edge(i, k):
                            transitive_after += 1
    
    transitivity_ratio_after = transitive_after / possible_after if possible_after > 0 else 0
    
    return {
        'transitive_triples': transitive_triples,
        'all_possible_triples': possible_triples,
        'transitivity_ratio': transitivity_ratio,
        'missing_edges_count': len(edges_to_add),
        'edges_to_add': edges_to_add,
        'min_edges_to_add': len(edges_to_add),
        'transitivity_ratio_after': transitivity_ratio_after,
        'is_transitive_after': transitivity_ratio_after == 1.0
    }

def visualize_transitivity(df, result):
    """Visualize transitivity results and save as PNG"""
    if df is None or result is None:
        return
    
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Original network
    ax = axes[0]
    pos = nx.spring_layout(G, seed=42)
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', 
                          node_size=400, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                          width=1.5, alpha=0.7, arrowsize=15, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    
    ax.set_title(f"Original Network\nTransitivity Ratio: {result['transitivity_ratio']:.4f}")
    ax.axis('off')
    
    # Plot 2: Network with added edges
    ax = axes[1]
    G_complete = G.copy()
    
    # Add the necessary edges
    for u, v in result['edges_to_add']:
        if not G_complete.has_edge(u, v):
            G_complete.add_edge(u, v)
    
    nx.draw_networkx_nodes(G_complete, pos, ax=ax, node_color='lightblue',
                          node_size=400, alpha=0.8)
    
    # Draw original edges in gray
    original_edges = list(G.edges())
    if original_edges:
        nx.draw_networkx_edges(G_complete, pos, edgelist=original_edges, ax=ax,
                              edge_color='gray', width=1.5, alpha=0.7, 
                              arrowsize=15, connectionstyle='arc3,rad=0.1')
    
    # Highlight added edges in orange
    added_edges = result['edges_to_add']
    if added_edges:
        nx.draw_networkx_edges(G_complete, pos, edgelist=added_edges, ax=ax,
                              edge_color='orange', width=3, alpha=0.8, 
                              arrowsize=20, connectionstyle='arc3,rad=0.1',
                              label=f'Added edges ({len(added_edges)})')
    
    nx.draw_networkx_labels(G_complete, pos, ax=ax, font_size=8)
    
    ax.set_title(f"Network with Added Edges\nTransitivity Ratio: {result['transitivity_ratio_after']:.4f}")
    if added_edges:
        ax.legend(loc='upper left')
    ax.axis('off')
    
    # Add statistics text
    stats_text = f"""
    Original Network:
    - Nodes: {G.number_of_nodes()}
    - Edges: {G.number_of_edges()}
    - Transitive Triples: {result['transitive_triples']}
    - Possible Triples: {result['all_possible_triples']}
    - Transitivity Ratio: {result['transitivity_ratio']:.4f}
    
    After Adding {len(added_edges)} Edges:
    - New Transitivity Ratio: {result['transitivity_ratio_after']:.4f}
    - {'✓ Fully Transitive' if result['is_transitive_after'] else '✗ Not Fully Transitive'}
    """
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    filename = output_dir / "part5_transitivity.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filename}")
    plt.close(fig)

# =============================================================================
# MAIN EXECUTION (UPDATED)
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("SIGNED NETWORK ANALYSIS - SAVING FIGURES AS PNG")
    print("=" * 70)
    print(f"Figures will be saved to: {output_dir}")
    
    # Use relative paths
    base_path = Path('../Networks/Part_A')
    
    # PART 1
    print("\n" + "=" * 70)
    print("PART 1: Sign Prediction")
    print("=" * 70)
    
    # Try different possible filenames for Part 1
    part1_files = [
        base_path / '1' / 'balance_graph.csv',
        base_path / '1' / 'balanced_graph.csv',
        base_path / '1' / 'graph.csv',
        base_path / '1' / 'network.csv'
    ]
    
    df1 = None
    for filepath in part1_files:
        if filepath.exists():
            print(f"Found file: {filepath}")
            df1 = load_balance_graph(str(filepath))
            if df1 is not None and not df1.empty:
                break
    
    if df1 is not None and not df1.empty:
        print(f"Loaded network with {len(df1)} edges")
        
        if 'sign' in df1.columns:
            missing_signs = df1['sign'].isna().sum()
            print(f"Known signs: {len(df1) - missing_signs}")
            print(f"Missing signs to predict: {missing_signs}")
            
            part1_result = predict_missing_signs_corrected(df1)
            print("\nSign Predictions Summary:")
            print(f"Total edges: {len(part1_result)}")
            print("Predicted signs distribution:")
            print(part1_result['predicted_sign'].value_counts())
            
            # Visualize and save
            visualize_sign_predictions(df1, part1_result, network_name="part1")
        else:
            print("No 'sign' column found in the data")
    else:
        print("Could not find or load Part 1 data")
    
    # PART 2
    print("\n" + "=" * 70)
    print("PART 2: Balance Test")
    print("=" * 70)
    
    network_files = []
    part2_path = base_path / '2'
    
    if part2_path.exists():
        # Try alphabetical naming first
        for letter in ['a', 'b', 'c', 'd', 'e']:
            for ext in ['csv', 'txt']:
                filepath = part2_path / f'network_{letter}.{ext}'
                if filepath.exists():
                    network_files.append((letter, str(filepath)))
                    break
        
        # If no alphabetical files, try numeric
        if not network_files:
            for i in range(1, 6):
                for ext in ['csv', 'txt']:
                    filepath = part2_path / f'network_{i}.{ext}'
                    if filepath.exists():
                        network_files.append((str(i), str(filepath)))
                        break
    
    if network_files:
        for network_id, filepath in network_files:
            print(f"\nAnalyzing Network {network_id}: {os.path.basename(filepath)}")
            try:
                df2 = load_signed_network(filepath)
                if df2 is not None and not df2.empty:
                    result = check_balance_corrected(df2)
                    if result:
                        print(f"  Balanced: {result['balanced']}")
                        print(f"  Number of super-nodes: {result['num_super_nodes']}")
                        print(f"  Contradictions found: {len(result['contradictions'])}")
                        
                        # Visualize and save
                        visualize_balance_test(df2, result, network_name=f"network_{network_id}")
                else:
                    print(f"  Could not load or process the file")
            except Exception as e:
                print(f"  Error: {e}")
    else:
        print(f"No network files found in {part2_path}")
    
    # PART 3
    print("\n" + "=" * 70)
    print("PART 3: Clusterability")
    print("=" * 70)
    
    cluster_files = []
    part3_path = base_path / '3'
    
    if part3_path.exists():
        for letter in ['a', 'b', 'c', 'd', 'e']:
            for ext in ['csv', 'txt']:
                filepath = part3_path / f'network_{letter}.{ext}'
                if filepath.exists():
                    cluster_files.append((letter, str(filepath)))
                    break
    
    if cluster_files:
        for network_id, filepath in cluster_files:
            print(f"\nAnalyzing Network {network_id}: {os.path.basename(filepath)}")
            try:
                df3 = load_signed_network(filepath)
                if df3 is not None and not df3.empty:
                    result = detect_clusters_corrected(df3)
                    if result:
                        print(f"  Number of clusters: {len(result['clusters'])}")
                        print(f"  Cluster sizes: {result['cluster_sizes']}")
                        print(f"  Violations: {result['violations']}")
                        print(f"  Weakly balanced: {result['is_weakly_balanced']}")
                        
                        # Visualize and save
                        visualize_clusters_corrected(df3, result, network_name=f"network_{network_id}")
                else:
                    print(f"  Could not load or process the file")
            except Exception as e:
                print(f"  Error: {e}")
    else:
        print(f"No network files found in {part3_path}")
    
    # PART 4
    print("\n" + "=" * 70)
    print("PART 4: Line Index")
    print("=" * 70)
    
    part4_files = [
        base_path / '4' / 'network_line_index.csv',
        base_path / '4' / 'line_index.csv',
        base_path / '4' / 'network.csv'
    ]
    
    df4 = None
    for filepath in part4_files:
        if filepath.exists():
            print(f"Found file: {filepath}")
            df4 = load_line_index_network(str(filepath))
            if df4 is not None and not df4.empty:
                break
    
    if df4 is not None and not df4.empty:
        print(f"Loaded network with {len(df4)} edges")
        print(f"Unique nodes: {len(set(df4['u'].tolist() + df4['v'].tolist()))}")
        
        # Random clustering
        random_assignments = random_clustering(df4, k=4)
        random_li, P_rand, N_rand = compute_line_index_corrected(df4, random_assignments)
        
        print("\nRandom Clustering Results:")
        print(f"  Line Index: {random_li:.4f}")
        print(f"  P (positive between clusters): {P_rand}")
        print(f"  N (negative within clusters): {N_rand}")
        
        # Improved clustering
        improved_assignments, improved_li = improve_clustering_optimized(df4, random_assignments)
        imp_li, P_imp, N_imp = compute_line_index_corrected(df4, improved_assignments)
        
        print("\nOptimized Clustering Results:")
        print(f"  Line Index: {imp_li:.4f}")
        print(f"  Improvement: {(random_li - imp_li):.4f} ({((random_li - imp_li)/random_li*100):.1f}%)")
        print(f"  P (positive between clusters): {P_imp}")
        print(f"  N (negative within clusters): {N_imp}")
        
        # Cluster sizes
        cluster_counts_rand = {}
        for cluster in random_assignments.values():
            cluster_counts_rand[cluster] = cluster_counts_rand.get(cluster, 0) + 1
        
        cluster_counts_imp = {}
        for cluster in improved_assignments.values():
            cluster_counts_imp[cluster] = cluster_counts_imp.get(cluster, 0) + 1
        
        print("\nRandom Cluster Distribution:")
        for cluster in sorted(cluster_counts_rand.keys()):
            print(f"  Cluster {cluster}: {cluster_counts_rand[cluster]} nodes")
        
        print("\nOptimized Cluster Distribution:")
        for cluster in sorted(cluster_counts_imp.keys()):
            print(f"  Cluster {cluster}: {cluster_counts_imp[cluster]} nodes")
        
        # Visualize and save
        visualize_line_index_results(df4, random_assignments, improved_assignments, random_li, imp_li)
    else:
        print("Could not find or load Part 4 data")
    
    # PART 5
    print("\n" + "=" * 70)
    print("PART 5: Transitivity")
    print("=" * 70)
    
    part5_files = [
        base_path / '5' / 'network_transitivity.csv',
        base_path / '5' / 'network_transivity.csv',
        base_path / '5' / 'transitivity.csv',
        base_path / '5' / 'network.csv'
    ]
    
    df5 = None
    for filepath in part5_files:
        if filepath.exists():
            print(f"Found file: {filepath}")
            df5 = load_transitivity_network(str(filepath))
            if df5 is not None and not df5.empty:
                break
    
    if df5 is not None and not df5.empty:
        print(f"Loaded network with {len(df5)} edges")
        
        result = analyze_transitivity_corrected(df5)
        
        print("\nNetwork Analysis:")
        nodes = set(df5['source'].tolist() + df5['target'].tolist())
        print(f"  Nodes: {len(nodes)}")
        print(f"  Edges: {len(df5)}")
        print(f"  Transitive triples: {result['transitive_triples']}")
        print(f"  All possible triples: {result['all_possible_triples']}")
        print(f"  Transitivity ratio: {result['transitivity_ratio']:.4f}")
        print(f"  Missing edges for full transitivity: {result['missing_edges_count']}")
        print(f"  Minimum edges to add: {result['min_edges_to_add']}")
        
        if result['edges_to_add']:
            print(f"\nFirst 5 edges to add (out of {len(result['edges_to_add'])}):")
            for i, edge in enumerate(result['edges_to_add'][:5], 1):
                print(f"  {i}. {edge[0]} → {edge[1]}")
        else:
            print("\nNo edges need to be added - network is already transitive")
        
        print(f"\nAfter adding {result['min_edges_to_add']} edges:")
        print(f"  Transitivity ratio: {result['transitivity_ratio_after']:.4f}")
        print(f"  Network is {'fully transitive' if result['is_transitive_after'] else 'not fully transitive'}")
        
        # Visualize and save
        visualize_transitivity(df5, result)
    else:
        print("Could not find or load Part 5 data")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"All figures saved to: {output_dir}")
    print("=" * 70)

# Run the main function
if __name__ == "__main__":
    main()