# ===========================
# Q5.py (MODIFIED + FIXED)
# ===========================
# Key fixes:
# 1) Force non-interactive backend (Agg) BEFORE importing pyplot
# 2) Save figures via fig.savefig + bbox_inches="tight" + ALWAYS close figures
# 3) Reduce DPI for heavy network plots (avoid PNG writer stalls)
# 4) Make spring_layout cheaper for bridge subgraph (iterations)
# 5) Fix inter-community heatmap matrix logic
# 6) Ensure output folders exist (results/, images/Q5/)

import os
import random
import time
import numpy as np
import community as community_louvain
import matplotlib

matplotlib.use("Agg")  # IMPORTANT: set backend before importing pyplot

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from collections import defaultdict, Counter
from networkx.algorithms.community import greedy_modularity_communities


# --------------------------
# Helpers
# --------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, filepath: str, dpi: int = 200):
    """
    Robust save: uses fig.savefig, tight bbox, then closes to prevent memory leaks/hangs.
    """
    ensure_dir(os.path.dirname(filepath))
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------
# Algorithm comparison
# --------------------------
def algorithm_comparison(G, num_runs=5):
    """
    Deploy three community detection algorithms and compare performance.
    Returns a DataFrame with timing + modularity + community size stats.
    """
    results = []

    # ------------------ Louvain ------------------
    print("Deploying Algorithm 1: Louvain...")
    louvain_times, louvain_mods, louvain_num_comms, louvain_largest, louvain_smallest = [], [], [], [], []

    for run in range(num_runs):
        start_time = time.time()
        partition = community_louvain.best_partition(G, random_state=run)
        elapsed = time.time() - start_time

        mod = community_louvain.modularity(partition, G)

        comms = defaultdict(list)
        for node, comm_id in partition.items():
            comms[comm_id].append(node)

        comm_sizes = [len(members) for members in comms.values()]

        louvain_times.append(elapsed)
        louvain_mods.append(mod)
        louvain_num_comms.append(len(comms))
        louvain_largest.append(max(comm_sizes))
        louvain_smallest.append(min(comm_sizes))

    results.append({
        'algorithm': 'Louvain',
        'avg_time': float(np.mean(louvain_times)),
        'std_time': float(np.std(louvain_times)),
        'avg_num_communities': float(np.mean(louvain_num_comms)),
        'avg_modularity': float(np.mean(louvain_mods)),
        'std_modularity': float(np.std(louvain_mods)),
        'largest_community_size': float(np.mean(louvain_largest)),
        'smallest_community_size': float(np.mean(louvain_smallest))
    })

    # ------------------ Label Propagation (scratch) ------------------
    print("Deploying Algorithm 2: Label Propagation...")
    lp_times, lp_mods, lp_num_comms, lp_largest, lp_smallest = [], [], [], [], []

    for run in range(num_runs):
        start_time = time.time()
        labels, _ = label_propagation(G, seed=run)
        elapsed = time.time() - start_time

        mod = calculate_modularity_from_labels(G, labels)

        comms = defaultdict(list)
        for node, comm_id in labels.items():
            comms[comm_id].append(node)

        comm_sizes = [len(members) for members in comms.values()]

        lp_times.append(elapsed)
        lp_mods.append(mod)
        lp_num_comms.append(len(comms))
        lp_largest.append(max(comm_sizes))
        lp_smallest.append(min(comm_sizes))

    results.append({
        'algorithm': 'Label Propagation',
        'avg_time': float(np.mean(lp_times)),
        'std_time': float(np.std(lp_times)),
        'avg_num_communities': float(np.mean(lp_num_comms)),
        'avg_modularity': float(np.mean(lp_mods)),
        'std_modularity': float(np.std(lp_mods)),
        'largest_community_size': float(np.mean(lp_largest)),
        'smallest_community_size': float(np.mean(lp_smallest))
    })

    # ------------------ Greedy Modularity ------------------
    print("Deploying Algorithm 3: Greedy Modularity...")
    greedy_times, greedy_mods, greedy_num_comms, greedy_largest, greedy_smallest = [], [], [], [], []

    for run in range(num_runs):
        start_time = time.time()
        comms = list(greedy_modularity_communities(G))
        elapsed = time.time() - start_time

        mod = nx.community.modularity(G, comms)
        comm_sizes = [len(c) for c in comms] if comms else [0]

        greedy_times.append(elapsed)
        greedy_mods.append(mod)
        greedy_num_comms.append(len(comms))
        greedy_largest.append(max(comm_sizes))
        greedy_smallest.append(min(comm_sizes))

    results.append({
        'algorithm': 'Greedy Modularity',
        'avg_time': float(np.mean(greedy_times)),
        'std_time': float(np.std(greedy_times)),
        'avg_num_communities': float(np.mean(greedy_num_comms)),
        'avg_modularity': float(np.mean(greedy_mods)),
        'std_modularity': float(np.std(greedy_mods)),
        'largest_community_size': float(np.mean(greedy_largest)),
        'smallest_community_size': float(np.mean(greedy_smallest))
    })

    return pd.DataFrame(results)


def label_propagation(G, max_iter=100, seed=None):
    """
    Label Propagation (from scratch):
    1) Unique label per node
    2) Iterate random order; each node adopts most frequent neighbor label
    3) Stop on convergence
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    labels = {node: i for i, node in enumerate(G.nodes())}

    for iteration in range(max_iter):
        changed = False
        nodes = list(G.nodes())
        random.shuffle(nodes)

        for node in nodes:
            neighbor_labels = [labels[nbr] for nbr in G.neighbors(node)]
            if not neighbor_labels:
                continue

            label_counts = Counter(neighbor_labels)
            max_count = max(label_counts.values())
            max_labels = [lab for lab, cnt in label_counts.items() if cnt == max_count]

            new_label = random.choice(max_labels) if len(max_labels) > 1 else max_labels[0]

            if labels[node] != new_label:
                labels[node] = new_label
                changed = True

        if not changed:
            return labels, iteration + 1

    return labels, max_iter


def calculate_modularity_from_labels(G, labels):
    """Calculate modularity given node->community labels."""
    comms = defaultdict(set)
    for node, cid in labels.items():
        comms[cid].add(node)
    return nx.community.modularity(G, list(comms.values()))


# --------------------------
# Deep analysis
# --------------------------
def deep_analysis(G, best_partition):
    """
    Deep analysis:
    - Bridges (connected to 3+ different communities)
    - Central node per community (degree centrality within subgraph)
    - Largest community stats
    - Intra vs inter edges
    - Community size distribution
    """
    analysis = {}

    communities = defaultdict(list)
    for node, comm_id in best_partition.items():
        communities[comm_id].append(node)

    # Bridges
    bridges = []
    for node in G.nodes():
        neighbor_comms = set(best_partition[nbr] for nbr in G.neighbors(node))
        if len(neighbor_comms) >= 3:
            bridges.append({
                'node': node,
                'degree': G.degree(node),
                'communities_connected': len(neighbor_comms),
                'own_community': best_partition[node],
                'connected_comms': list(neighbor_comms)
            })
    bridges.sort(key=lambda x: x['communities_connected'], reverse=True)
    analysis['bridges'] = bridges

    # Central per community
    central_per_community = {}
    for comm_id, members in communities.items():
        subgraph = G.subgraph(members)
        dc = nx.degree_centrality(subgraph)
        if dc:
            top_node = max(dc, key=dc.get)
            central_per_community[comm_id] = {
                'node': top_node,
                'centrality': dc[top_node],
                'degree': subgraph.degree(top_node),
                'community_size': len(members)
            }
    analysis['central_per_community'] = central_per_community

    # Largest community
    largest_comm_id = max(communities, key=lambda c: len(communities[c]))
    largest_members = communities[largest_comm_id]
    largest_subgraph = G.subgraph(largest_members)

    analysis['largest_community'] = {
        'id': largest_comm_id,
        'size': len(largest_members),
        'members': sorted(largest_members),
        'degree_centrality': nx.degree_centrality(largest_subgraph),
        'betweenness_centrality': nx.betweenness_centrality(largest_subgraph)
    }

    # Edge distribution
    intra_edges, inter_edges = 0, 0
    for u, v in G.edges():
        if best_partition[u] == best_partition[v]:
            intra_edges += 1
        else:
            inter_edges += 1

    analysis['edge_distribution'] = {
        'intra': intra_edges,
        'inter': inter_edges,
        'ratio': intra_edges / inter_edges if inter_edges > 0 else float('inf')
    }

    # Community size distribution
    sizes = [len(m) for m in communities.values()]
    analysis['community_size_distribution'] = {
        'sizes': sizes,
        'num_communities': len(communities),
        'max': max(sizes),
        'min': min(sizes),
        'mean': float(np.mean(sizes)),
        'std': float(np.std(sizes))
    }

    return analysis


# --------------------------
# Visualization
# --------------------------
def visualize_communities(G, partition, analysis, outdir="./images/Q5", dpi=200):
    """
    Save each required plot as a separate PNG:
    1) network communities
    2) community size histogram
    3) inter-community heatmap
    4) bridge subgraph
    5) degree distribution
    6) summary text image

    Uses Agg backend + fig.savefig to avoid PIL save issues/hangs.
    """
    ensure_dir(outdir)

    # Community -> nodes mapping
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    unique_comms = list(communities.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_comms), 1)))
    node_colors = [colors[unique_comms.index(partition[node])] for node in G.nodes()]

    # Precompute positions once for the full graph
    pos = nx.spring_layout(G, seed=42, iterations=50)
    bridge_nodes = [b['node'] for b in analysis['bridges']]

    # 1) Network with community colors
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8, ax=ax)
    if bridge_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=bridge_nodes, node_color='red',
            node_size=200, node_shape='s', ax=ax
        )

    key_chars = ['Valjean', 'Javert', 'Marius', 'Cosette', 'Fantine', 'Thenardier']
    labels = {c: c for c in key_chars if c in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    ax.set_title("Les Misérables Character Network\n(Red squares = Bridge characters)")
    ax.axis('off')
    save_fig(fig, os.path.join(outdir, "network_communities.png"), dpi=dpi)

    # 2) Community size distribution
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    comm_sizes = analysis['community_size_distribution']['sizes']
    ax.hist(comm_sizes, bins=15, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Community Size")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Community Size Distribution\n({analysis['community_size_distribution']['num_communities']} communities)")
    ax.axvline(np.mean(comm_sizes), color='red', linestyle='--', label=f"Mean: {np.mean(comm_sizes):.1f}")
    ax.legend()
    save_fig(fig, os.path.join(outdir, "community_size_distribution.png"), dpi=dpi)

    # 3) Inter-community connection heatmap (fixed logic)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    comm_ids = sorted(unique_comms)
    comm_index = {c: i for i, c in enumerate(comm_ids)}
    n = len(comm_ids)
    connection_matrix = np.zeros((n, n), dtype=float)

    for u, v in G.edges():
        i = comm_index[partition[u]]
        j = comm_index[partition[v]]
        # keep symmetric counts
        connection_matrix[i, j] += 1
        connection_matrix[j, i] += 1

    im = ax.imshow(connection_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Community ID")
    ax.set_title("Inter-Community Connections (symmetric edge counts)")
    fig.colorbar(im, ax=ax, label="Number of edges")
    save_fig(fig, os.path.join(outdir, "intercommunity_heatmap.png"), dpi=dpi)

    # 4) Bridge nodes highlighted (subgraph) — cheaper layout
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    bridge_set = set(bridge_nodes)
    neighbor_set = set()
    for b in bridge_nodes:
        neighbor_set.update(G.neighbors(b))

    nodes_of_interest = bridge_set.union(neighbor_set)
    subG = G.subgraph(nodes_of_interest).copy()

    # If for some reason there are no bridge nodes, still plot something stable:
    if subG.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No bridge nodes detected.", ha='center', va='center')
        ax.axis('off')
        save_fig(fig, os.path.join(outdir, "bridge_subgraph.png"), dpi=dpi)
    else:
        sub_pos = nx.spring_layout(subG, seed=42, iterations=50)
        sub_colors = [colors[unique_comms.index(partition[node])] for node in subG.nodes()]

        nx.draw_networkx_edges(subG, sub_pos, alpha=0.3, ax=ax)
        nx.draw_networkx_nodes(subG, sub_pos, node_color=sub_colors, node_size=200, alpha=0.8, ax=ax)
        if bridge_nodes:
            nx.draw_networkx_nodes(
                subG, sub_pos, nodelist=[n for n in bridge_nodes if n in subG],
                node_color='red', node_size=300, node_shape='s', ax=ax
            )
            bridge_labels = {node: node for node in bridge_nodes if node in subG}
            nx.draw_networkx_labels(subG, sub_pos, bridge_labels, font_size=8, ax=ax)

        ax.set_title("Bridge Characters and Their Connections")
        ax.axis('off')
        # Bridge plot is the one that often hangs at 300 DPI; keep this lower.
        save_fig(fig, os.path.join(outdir, "bridge_subgraph.png"), dpi=min(dpi, 200))

    # 5) Degree distribution
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    degrees = [G.degree(node) for node in G.nodes()]
    ax.hist(degrees, bins=15, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")
    ax.set_title("Degree Distribution")
    ax.axvline(np.mean(degrees), color='red', linestyle='--', label=f"Mean: {np.mean(degrees):.1f}")
    ax.legend()
    save_fig(fig, os.path.join(outdir, "degree_distribution.png"), dpi=dpi)

    # 6) Summary text image
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')

    mod = community_louvain.modularity(partition, G)
    summary_text = f"""
NETWORK INTELLIGENCE SUMMARY
----------------------------
Network Statistics:
• Nodes: {G.number_of_nodes()}
• Edges: {G.number_of_edges()}
• Density: {nx.density(G):.4f}

Community Structure:
• Communities: {analysis['community_size_distribution']['num_communities']}
• Modularity: {mod:.4f}
• Largest Community: {analysis['largest_community']['size']} nodes
• Smallest Community: {analysis['community_size_distribution']['min']} nodes

Edge Distribution:
• Intra-community: {analysis['edge_distribution']['intra']}
• Inter-community: {analysis['edge_distribution']['inter']}
• Ratio: {analysis['edge_distribution']['ratio']:.2f}

Bridge Characters:
• Total: {len(analysis['bridges'])}
• Top Bridges: {', '.join([b['node'] for b in analysis['bridges'][:3]]) if analysis['bridges'] else 'None'}
""".strip()

    ax.text(0.05, 0.95, summary_text, va='top', fontsize=11)
    save_fig(fig, os.path.join(outdir, "summary_report.png"), dpi=dpi)

    print(f"✓ Saved 6 separate plots to {outdir}")


def create_comparison_table(df):
    """Create formatted comparison table for printing."""
    display_df = df.copy()
    display_df['avg_time'] = display_df['avg_time'].apply(lambda x: f"{x:.4f}")
    display_df['std_time'] = display_df['std_time'].apply(lambda x: f"{x:.4f}")
    display_df['avg_num_communities'] = display_df['avg_num_communities'].apply(lambda x: f"{x:.1f}")
    display_df['avg_modularity'] = display_df['avg_modularity'].apply(lambda x: f"{x:.4f}")
    display_df['std_modularity'] = display_df['std_modularity'].apply(lambda x: f"{x:.4f}")
    display_df['largest_community_size'] = display_df['largest_community_size'].apply(lambda x: f"{x:.1f}")
    display_df['smallest_community_size'] = display_df['smallest_community_size'].apply(lambda x: f"{x:.1f}")

    display_df.columns = ['Algorithm', 'Time(s)', 'Std(s)', '#Comm', 'Modularity',
                          'Std(Mod)', 'Largest', 'Smallest']
    return display_df


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    ensure_dir("./images/Q5")
    ensure_dir("./results")

    # Load Les Misérables network
    G = nx.les_miserables_graph()

    print("=" * 80)
    print("OPERATION: RAPID DETECTION")
    print("=" * 80)
    print("Target Network: Les Misérables")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")
    print()

    # Execute multi-algorithm comparison
    comparison_df = algorithm_comparison(G, num_runs=5)

    # Display comparison table
    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON RESULTS")
    print("=" * 80)
    display_df = create_comparison_table(comparison_df)
    print(display_df.to_string(index=False))

    # Save results
    comparison_df.to_csv("./results/algorithm_comparison.csv", index=False)

    # Identify best algorithm
    best_idx = comparison_df['avg_modularity'].idxmax()
    best_algo = comparison_df.iloc[best_idx]

    print("\n" + "=" * 80)
    print(f"BEST ALGORITHM: {best_algo['algorithm']}")
    print("=" * 80)
    print(f"Modularity: {best_algo['avg_modularity']:.4f} (±{best_algo['std_modularity']:.4f})")
    print(f"Time: {best_algo['avg_time']:.4f}s (±{best_algo['std_time']:.4f}s)")
    print(f"Communities: {best_algo['avg_num_communities']:.1f}")

    # Best partition + deep analysis (use Louvain for partition)
    print("\nPerforming deep network analysis...")
    best_partition = community_louvain.best_partition(G, random_state=42)
    analysis = deep_analysis(G, best_partition)

    # Print bridge characters
    print("\n" + "=" * 80)
    print("BRIDGE CHARACTERS (Connected to 3+ Communities)")
    print("=" * 80)
    for bridge in analysis['bridges']:
        print(f"• {bridge['node']}: Degree {bridge['degree']}, "
              f"Connects to {bridge['communities_connected']} communities")

    # Largest community analysis
    print("\n" + "=" * 80)
    print("LARGEST COMMUNITY ANALYSIS")
    print("=" * 80)
    largest = analysis['largest_community']
    print(f"Community ID: {largest['id']}")
    print(f"Size: {largest['size']} characters")

    if largest['degree_centrality']:
        dc_max_node = max(largest['degree_centrality'], key=largest['degree_centrality'].get)
        bc_max_node = max(largest['betweenness_centrality'], key=largest['betweenness_centrality'].get)
        print(f"\nMost central (degree): {dc_max_node} ({largest['degree_centrality'][dc_max_node]:.3f})")
        print(f"Most central (betweenness): {bc_max_node} ({largest['betweenness_centrality'][bc_max_node]:.3f})")

    # Network disruption simulation
    print("\n" + "=" * 80)
    print("NETWORK DISRUPTION SIMULATION")
    print("=" * 80)

    G_disrupted = G.copy()
    bridge_nodes = [b['node'] for b in analysis['bridges']]
    G_disrupted.remove_nodes_from(bridge_nodes)

    print(f"Removing {len(bridge_nodes)} bridge characters...")
    print(f"Original components: {nx.number_connected_components(G)}")
    print(f"After removal: {nx.number_connected_components(G_disrupted)} components")
    print(f"Network fragmentation: {nx.number_connected_components(G_disrupted) - 1} new components")

    # Visualizations
    print("\nGenerating visualizations...")
    visualize_communities(G, best_partition, analysis, outdir="./images/Q5", dpi=200)

    print("\n✓ Mission Complete")