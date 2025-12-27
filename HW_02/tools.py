import networkx as nx

# Define the edges from your network
edges = [
    (1, 2), (1, 3), (1, 4), (1, 12),
    (2, 3), (2, 13),
    (3, 4), (3, 7),
    (4, 5),
    (5, 6), (5, 7), (5, 8),
    (6, 7), (6, 13),
    (7, 8),
    (8, 9),
    (9, 10), (9, 11), (9, 12),
    (10, 11), (10, 13),
    (11, 12),
    (12, 1),
    (13, 6), (13, 10), (13, 14),
    (14, 15)
]

G = nx.Graph()
G.add_edges_from(edges)

# --- 1. Radio Broadcast Station (Center of Graph) ---
# Goal: Minimize maximum distance (Eccentricity)
ecc = nx.eccentricity(G)
center_nodes = nx.center(G)
print(f"Scenario 1: Radio Station")
print(f"Nodes with minimum eccentricity: {center_nodes} (Value: {ecc[center_nodes[0]]})")

# --- 2. Bookstore (Median of Graph) ---
# Goal: Minimize sum of distances (Max Closeness Centrality)
closeness = nx.closeness_centrality(G)
# Sorting to find the highest closeness
sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
print(f"\nScenario 3: Bookstore (Top 3 Closeness)")
for node, val in sorted_closeness[:3]:
    print(f"Node {node}: {val:.4f}")

# --- 3. Competitive Stores (Market Share) ---
def get_market_share(node_a, node_b):
    a_count = 0
    b_count = 0
    for node in G.nodes():
        dist_a = nx.shortest_path_length(G, node, node_a)
        dist_b = nx.shortest_path_length(G, node, node_b)
        if dist_a < dist_b:
            a_count += 1
        elif dist_b < dist_a:
            b_count += 1
        else:
            a_count += 0.5
            b_count += 0.5
    return a_count, b_count

# Check your proposed A=13, B=3
share_a, share_b = get_market_share(13, 3)
print(f"\nScenario 2: Competitive Stores")
print(f"With Store A at 13 and Store B at 3:")
print(f"Store A captures: {share_a} nodes")
print(f"Store B captures: {share_b} nodes")