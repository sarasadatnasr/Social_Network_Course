import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import defaultdict


def girvan_newman_analysis(G, target_communities=2):

    results = {
        'removed_edges': [],
        'modularity_at_each_step': [],
        'num_components': [],
        'final_communities': None,
        'critical_edge': None,
        'max_modularity': -1,
        'optimal_step': 0
    }

    # Create a copy of the graph to work with
    G_copy = G.copy()
    
    # Track number of components
    components = list(nx.connected_components(G_copy))
    num_components = len(components)
    
    # Calculate initial modularity
    initial_communities = [set(comp) for comp in components]
    current_modularity = calculate_modularity(G, initial_communities)
    results['modularity_at_each_step'].append(current_modularity)
    results['num_components'].append(num_components)
    
    step = 0
    
    # Continue until we reach target number of communities
    while num_components < target_communities and G_copy.number_of_edges() > 0:
        # Calculate edge betweenness
        edge_betweenness = nx.edge_betweenness_centrality(G_copy)
        
        # Find edge with highest betweenness
        max_edge = max(edge_betweenness.items(), key=lambda x: x[1])
        edge_to_remove = max_edge[0]
        betweenness_value = max_edge[1]
        
        # Record the removed edge
        results['removed_edges'].append((edge_to_remove[0], edge_to_remove[1], betweenness_value))
        
        # Remove the edge
        G_copy.remove_edge(*edge_to_remove)
        
        # Check if this removal created a new component (critical edge)
        new_components = list(nx.connected_components(G_copy))
        new_num_components = len(new_components)
        
        # Record if this is the critical edge (first time number of components increases)
        if new_num_components > num_components and results['critical_edge'] is None:
            results['critical_edge'] = edge_to_remove
        
        num_components = new_num_components
        results['num_components'].append(num_components)
        
        # Calculate modularity for current partition
        communities = [set(comp) for comp in new_components]
        modularity = calculate_modularity(G, communities)
        results['modularity_at_each_step'].append(modularity)
        
        # Track maximum modularity
        if modularity > results['max_modularity']:
            results['max_modularity'] = modularity
            results['optimal_step'] = step
            results['final_communities'] = communities
        
        step += 1
    
    return results


def calculate_modularity(G, communities):

    m = G.number_of_edges()
    if m == 0:
        return 0
    
    # Create a dictionary mapping node to community index
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i
    
    Q = 0
    # Sum over all pairs of nodes
    for i in G.nodes():
        for j in G.nodes():
            # Check if nodes are in same community
            if node_to_comm.get(i, -1) == node_to_comm.get(j, -1):
                # A_ij: 1 if edge exists, 0 otherwise
                A_ij = 1 if G.has_edge(i, j) else 0
                # Expected edges by chance
                expected = (G.degree(i) * G.degree(j)) / (2 * m)
                Q += A_ij - expected
    
    Q = Q / (2 * m)
    return Q


def visualize_results(G, results, true_labels):

    
    # Create color mapping for true labels
    unique_labels = list(set(true_labels.values()))
    label_to_color_true = {label: i for i, label in enumerate(unique_labels)}
    node_colors_true = [label_to_color_true[true_labels[node]] for node in G.nodes()]
    
    # Fixed position for consistent layouts
    pos = nx.spring_layout(G, seed=42)
    
    # Plot 1: Modularity progression
    plt.figure(figsize=(10, 6))
    steps = range(len(results['modularity_at_each_step']))
    plt.plot(steps, results['modularity_at_each_step'], 'b-', linewidth=2, marker='o', markersize=4)
    plt.axhline(y=results['max_modularity'], color='r', linestyle='--', alpha=0.5, 
                label=f"Max Q = {results['max_modularity']:.3f}")
    plt.axvline(x=results['optimal_step'], color='g', linestyle='--', alpha=0.5,
                label=f"Optimal step = {results['optimal_step']}")
    plt.xlabel('Number of Edges Removed')
    plt.ylabel('Modularity (Q)')
    plt.title('Modularity Progression During Edge Removal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./images/Q4/modularity_progression.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/1_modularity_progression.png")
    
    # Plot 2: Number of components
    plt.figure(figsize=(10, 6))
    plt.plot(steps, results['num_components'], 'r-', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Number of Edges Removed')
    plt.ylabel('Number of Components')
    plt.title('Network Fragmentation Progression')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./images/Q4/fragmentation_progression.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/2_fragmentation_progression.png")
    
    # Plot 3: Ground truth network
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_color=node_colors_true, node_size=300, with_labels=True, 
            font_size=8, font_weight='bold', cmap=plt.cm.Set1, edge_color='gray')
    plt.title('Ground Truth Network\n(Mr. Hi vs Officer)')
    plt.tight_layout()
    plt.savefig('./images/Q4/ground_truth_network.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/3_ground_truth_network.png")
    
    # Plot 4: Detected communities
    plt.figure(figsize=(10, 8))
    # Map nodes to detected communities
    node_to_comm_detected = {}
    for i, comm in enumerate(results['final_communities']):
        for node in comm:
            node_to_comm_detected[node] = i
    node_colors_detected = [node_to_comm_detected[node] for node in G.nodes()]
    nx.draw(G, pos, node_color=node_colors_detected, node_size=300, with_labels=True,
            font_size=8, font_weight='bold', cmap=plt.cm.Set1, edge_color='gray')
    
    # Highlight critical edge
    if results['critical_edge']:
        nx.draw_networkx_edges(G, pos, edgelist=[results['critical_edge']], 
                               edge_color='r', width=3, alpha=0.7)
    plt.title('Detected Communities\n(Critical Edge Highlighted in Red)')
    plt.tight_layout()
    plt.savefig('./images/Q4/detected_communities.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/4_detected_communities.png")
    
    # Plot 5: Confusion matrix
    plt.figure(figsize=(8, 6))
    # Prepare labels for confusion matrix
    y_true = [true_labels[node] for node in sorted(G.nodes())]
    # Map detected communities to match true labels (handle label reversal)
    accuracy, aligned_detected = calculate_accuracy(results['final_communities'], true_labels, return_labels=True)
    
    cm = confusion_matrix(y_true, aligned_detected)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}')
    plt.ylabel('True Label')
    plt.xlabel('Detected Label')
    plt.tight_layout()
    plt.savefig('./images/Q4/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/5_confusion_matrix.png")
    
    # Plot 6: Critical edge analysis
    plt.figure(figsize=(10, 8))
    if results['critical_edge']:
        u, v = results['critical_edge']
        edge_info = f"Critical Edge: {u}-{v}\n"
        edge_info += f"Degree of {u}: {G.degree(u)}\n"
        edge_info += f"Degree of {v}: {G.degree(v)}\n"
        edge_info += f"Connects: {true_labels[u]} <-> {true_labels[v]}"
        
        # Create a subgraph showing the critical edge and its neighbors
        neighbors = set(G.neighbors(u)) | set(G.neighbors(v))
        subgraph_nodes = list(neighbors | {u, v})
        subgraph = G.subgraph(subgraph_nodes)
        
        pos_sub = nx.spring_layout(subgraph, seed=42)
        node_colors_sub = [label_to_color_true[true_labels[n]] for n in subgraph.nodes()]
        
        nx.draw(subgraph, pos_sub, node_color=node_colors_sub, node_size=500, 
                with_labels=True, font_size=10, font_weight='bold', cmap=plt.cm.Set1)
        nx.draw_networkx_edges(subgraph, pos_sub, edgelist=[(u, v)], 
                               edge_color='r', width=3)
        plt.title(edge_info, fontsize=12)
    else:
        plt.text(0.5, 0.5, 'No critical edge found yet', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Critical Edge Analysis')
    
    plt.tight_layout()
    plt.savefig('./images/Q4/critical_edge_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()



def calculate_accuracy(detected_communities, true_labels, return_labels=False):

    # Convert true labels to binary (0 and 1)
    unique_labels = list(set(true_labels.values()))
    true_label_to_int = {label: i for i, label in enumerate(unique_labels)}
    
    # Sort nodes for consistent ordering
    nodes = sorted(true_labels.keys())
    
    # Create true labels array
    y_true = [true_label_to_int[true_labels[node]] for node in nodes]
    
    # Create detected labels array (using community index)
    node_to_detected = {}
    for i, comm in enumerate(detected_communities):
        for node in comm:
            node_to_detected[node] = i
    
    y_detected = [node_to_detected[node] for node in nodes]
    
    # Try both alignments (original and swapped)
    accuracy1 = accuracy_score(y_true, y_detected)
    y_detected_swapped = [1 - label for label in y_detected]
    accuracy2 = accuracy_score(y_true, y_detected_swapped)
    
    if accuracy1 >= accuracy2:
        best_accuracy = accuracy1
        best_aligned = y_detected
    else:
        best_accuracy = accuracy2
        best_aligned = y_detected_swapped
    
    if return_labels:
        # Convert back to original label names for visualization
        int_to_true_label = {i: label for i, label in enumerate(unique_labels)}
        aligned_labels = [int_to_true_label[label] for label in best_aligned]
        return best_accuracy, aligned_labels
    
    return best_accuracy


if __name__ == "__main__":
    # Load Zachary's Karate Club network
    G = nx.karate_club_graph()

    # Extract ground truth labels
    true_labels = {}
    for node in G.nodes():
        true_labels[node] = G.nodes[node]['club']

    print("=" * 60)
    print("OPERATION: CHAIN BREAKER")
    print("=" * 60)
    print(f"Target Network: Zachary's Karate Club")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print()

    # Execute Girvan-Newman algorithm
    print("Executing Girvan-Newman algorithm...")
    results = girvan_newman_analysis(G, target_communities=2)

    # Display results
    print("\n" + "=" * 60)
    print("OPERATION RESULTS")
    print("=" * 60)
    print(f"Total edges removed: {len(results['removed_edges'])}")
    print(f"Maximum Modularity: {results['max_modularity']:.4f}")
    print(f"Optimal step: {results['optimal_step']}")
    print(f"Critical Edge: {results['critical_edge']}")
    if results['critical_edge']:
        u, v = results['critical_edge']
        print(f"  - Connects nodes {u} (degree {G.degree(u)}) and {v} (degree {G.degree(v)})")
        print(f"  - Factions: {true_labels[u]} <-> {true_labels[v]}")
    print(f"Final community sizes: {[len(c) for c in results['final_communities']]}")

    # Calculate accuracy
    accuracy = calculate_accuracy(results['final_communities'], true_labels)
    print(f"Accuracy vs Ground Truth: {accuracy:.2%}")

    # Generate visualizations
    print("\nGenerating intelligence report visualizations...")
    visualize_results(G, results, true_labels)

    print("\n✓ Mission Complete")
    
    # Additional analysis for written report
    print("\n" + "=" * 60)
    print("INTELLIGENCE ANALYSIS")
    print("=" * 60)
    
    # Modularity analysis
    max_q_step = results['optimal_step']
    print(f"\n1. MODULARITY PROGRESSION ANALYSIS:")
    print(f"   - Maximum modularity (Q={results['max_modularity']:.4f}) achieved at step {max_q_step}")
    if results['num_components'][max_q_step] == 2:
        print(f"   - ✓ Maximum modularity occurs exactly when network first splits into 2 components")
    else:
        print(f"   - Note: Maximum modularity occurs when network has {results['num_components'][max_q_step]} components")

    
    # Critical edge analysis
    print(f"\n2. CRITICAL EDGE ANALYSIS:")
    if results['critical_edge']:
        u, v = results['critical_edge']
        # Find the betweenness value for this edge
        betweenness = None
        for (x, y, b) in results['removed_edges']:
            if (x == u and y == v) or (x == v and y == u):
                betweenness = b
                break
        
        print(f"   - Critical edge: {u}-{v} (betweenness: {betweenness:.4f})")
        print(f"   - Node characteristics:")
        print(f"     * Node {u}: degree={G.degree(u)}, faction={true_labels[u]}")
        print(f"     * Node {v}: degree={G.degree(v)}, faction={true_labels[v]}")
        print(f"   - Position: This edge acts as a bridge between the two factions,")
        print(f"     connecting the '{true_labels[u]}' group (node {u}) to the '{true_labels[v]}' group (node {v})")
        if true_labels[u] != true_labels[v]:
            print(f"   - ✓ This edge directly connects the two real factions")
        else:
            print(f"   - Note: This edge connects nodes within the same faction")
    
    # Accuracy analysis
    print(f"\n3. GROUND TRUTH COMPARISON:")
    print(f"   - Classification accuracy: {accuracy:.2%}")
    
    # Find misclassified nodes
    if accuracy < 1.0:
        nodes = sorted(true_labels.keys())
        # Get detected communities in proper alignment
        _, aligned_labels = calculate_accuracy(results['final_communities'], true_labels, return_labels=True)
        
        misclassified = []
        for node, true, detected in zip(nodes, [true_labels[n] for n in nodes], aligned_labels):
            if true != detected:
                misclassified.append(node)
        
        print(f"   - Misclassified agents: {misclassified}")

    
    print("\n" + "=" * 60)
    print("STRATEGIC REPORT")
    print("=" * 60)
    print("\nOPERATION SUMMARY:")
    print(f"   - Total communication links severed before fragmentation: {results['optimal_step']}")
    print(f"   - Critical communication channel: Edge {results['critical_edge']}")
    if results['critical_edge']:
        u, v = results['critical_edge']
        print(f"     (Connects node {u} [{true_labels[u]}] to node {v} [{true_labels[v]}])")
    

    
    # Summary of findings
    print("\n" + "=" * 60)
    print("FINAL OPERATIONAL ASSESSMENT")
    print("=" * 60)
    print(f"✓ Successfully identified {len(results['final_communities'])} spy cells")
    print(f"✓ Critical communication link: {results['critical_edge']}")
    if results['critical_edge']:
        u, v = results['critical_edge']
        betweenness = next((b for (x, y, b) in results['removed_edges'] 
                           if (x == u and y == v) or (x == v and y == u)), None)
        print(f"  (betweenness: {betweenness:.4f})")
    print(f"✓ Classification accuracy: {accuracy:.2%}")
    if accuracy < 1.0 and 'misclassified' in locals():
        print(f"⚠ Ambiguous agents identified: {misclassified} - require further surveillance")
