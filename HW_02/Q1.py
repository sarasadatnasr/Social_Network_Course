import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.sparse.linalg import eigsh
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Load data
nodes = pd.read_csv('./HW_02/data/q1/politician_nodes.csv')
edges = pd.read_csv('./HW_02/data/q1/politician_edges.csv')

# Create graph
G = nx.Graph()
G.add_nodes_from(nodes['id'].values)
G.add_edges_from(edges[['id_1', 'id_2']].values)

# Merge node attributes
node_attrs = nodes.set_index('id').to_dict('index')
nx.set_node_attributes(G, node_attrs)

print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ============================================================================
# PART (a): Power Geometry
# ============================================================================

print("\n" + "="*60)
print("PART (a): POWER GEOMETRY")
print("="*60)

# 1. Centrality Calculations
def normalize_centrality(cent_dict):
    """Normalize centrality values to [0,1]"""
    if not cent_dict:
        return cent_dict
    max_val = max(cent_dict.values())
    min_val = min(cent_dict.values())
    if max_val == min_val:
        return {k: 0.5 for k in cent_dict}
    return {k: (v - min_val) / (max_val - min_val) for k, v in cent_dict.items()}

degree = dict(G.degree())
degree_norm = normalize_centrality(degree)

try:
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
    eigenvector_norm = normalize_centrality(eigenvector)
except:
    eigenvector_norm = degree_norm.copy()

closeness = nx.closeness_centrality(G)
closeness_norm = normalize_centrality(closeness)

# Store in DataFrame
centrality_df = pd.DataFrame({
    'id': list(G.nodes()),
    'degree': [degree_norm.get(n, 0) for n in G.nodes()],
    'eigenvector': [eigenvector_norm.get(n, 0) for n in G.nodes()],
    'closeness': [closeness_norm.get(n, 0) for n in G.nodes()]
})

# Merge with names
centrality_df = centrality_df.merge(nodes[['id', 'page_name']], on='id')

# Get top 10 for each
print("\n1. TOP 10 NODES BY CENTRALITY:")
for metric in ['degree', 'eigenvector', 'closeness']:
    top_10 = centrality_df.nlargest(10, metric)[['page_name', metric]]
    print(f"\n{metric.upper()} Top 10:")
    print(top_10.round(4).to_string(index=False))

# 2. Gap Analysis
plt.figure(figsize=(10, 8))
plt.scatter(centrality_df['degree'], centrality_df['eigenvector'], 
           alpha=0.5, s=20, c='steelblue')

# Fit line
z = np.polyfit(centrality_df['degree'], centrality_df['eigenvector'], 1)
p = np.poly1d(z)
x_range = np.linspace(centrality_df['degree'].min(), centrality_df['degree'].max(), 100)
plt.plot(x_range, p(x_range), 'r--', alpha=0.8, linewidth=2, label='Correlation Line')

# Find outliers (low degree, high eigenvector)
centrality_df['predicted_eigen'] = p(centrality_df['degree'])
centrality_df['eigen_residual'] = centrality_df['eigenvector'] - centrality_df['predicted_eigen']

# Get outliers
outliers = centrality_df.nlargest(20, 'eigen_residual')
plt.scatter(outliers['degree'], outliers['eigenvector'], 
           color='red', s=50, alpha=0.8, label='Outliers')

plt.xlabel('Normalized Degree', fontsize=12)
plt.ylabel('Normalized Eigenvector Centrality', fontsize=12)
plt.title('Degree vs Eigenvector Centrality Gap Analysis', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('./HW_02/Q1/gap_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Three-Way Case Study
print("\n" + "-"*60)
print("3. THREE-WAY CASE STUDY: Low Degree, High Eigenvector")
print("-"*60)

# Get low degree (outside top 100) but high eigenvector (top 50)
degree_ranks = centrality_df['degree'].rank(ascending=False, method='min')
eigen_ranks = centrality_df['eigenvector'].rank(ascending=False, method='min')
closeness_ranks = centrality_df['closeness'].rank(ascending=False, method='min')

centrality_df['degree_rank'] = degree_ranks
centrality_df['eigen_rank'] = eigen_ranks
centrality_df['closeness_rank'] = closeness_ranks

# Criteria: degree rank > 100, eigen rank <= 50
case_study = centrality_df[(centrality_df['degree_rank'] > 100) & 
                          (centrality_df['eigen_rank'] <= 50)].copy()

if len(case_study) >= 3:
    selected = case_study.nsmallest(3, 'degree_rank')
    print("\nSelected Politicians:")
    for idx, row in selected.iterrows():
        closeness_status = "HIGH (Geometric Heart)" if row['closeness_rank'] <= 100 else "LOW (Marginal Attachment)"
        print(f"\n{row['page_name']}:")
        print(f"  Degree Rank: {int(row['degree_rank'])}")
        print(f"  Eigenvector Rank: {int(row['eigen_rank'])}")
        print(f"  Closeness Rank: {int(row['closeness_rank'])} - {closeness_status}")
else:
    print("Not enough candidates found. Using top outliers instead.")
    selected = centrality_df.nlargest(3, 'eigen_residual').head(3)

# ============================================================================
# PART (b): Information Bottlenecks
# ============================================================================

print("\n\n" + "="*60)
print("PART (b): INFORMATION BOTTLENECKS")
print("="*60)

# 1. Betweenness Calculation
print("\n1. BETWEENNESS CENTRALITY")
betweenness = nx.betweenness_centrality(G, normalized=True, k=min(1000, len(G)))
centrality_df['betweenness'] = [betweenness.get(n, 0) for n in centrality_df['id']]
centrality_df['betweenness_rank'] = centrality_df['betweenness'].rank(ascending=False, method='min')

# Top 10 betweenness
top_betweenness = centrality_df.nlargest(10, 'betweenness')[['page_name', 'betweenness', 'degree_rank']]
print("\nTop 10 Betweenness Centrality:")
top_betweenness_display = top_betweenness.copy()
top_betweenness_display['degree_rank'] = top_betweenness_display['degree_rank'].astype(int)
print(top_betweenness_display.round(4).to_string(index=False))

# 2. Rank Gap Analysis
print("\n2. BRIDGES VS HUBS ANALYSIS")
print("\nMathematical Bridges (High Betweenness, Lower Degree):")
bridges = top_betweenness[top_betweenness['degree_rank'] > 10]
if not bridges.empty:
    print(bridges[['page_name', 'degree_rank']].to_string(index=False))
    print("\nStructural Difference:")
    print("• Bridges: Connect different communities, control information flow")
    print("• Hubs: Have many connections but within similar circles")
else:
    print("No strong bridges found in top 10")

# ============================================================================
# PART (c): Power in Local Structures
# ============================================================================

print("\n\n" + "="*60)
print("PART (c): POWER IN LOCAL STRUCTURES")
print("="*60)

# 1. & 2. Closeness Analysis
print("\n1. TOP 10 CLOSENESS CENTRALITY:")
top_closeness = centrality_df.nsmallest(10, 'closeness_rank')[['page_name', 'closeness']]
print(top_closeness.round(4).to_string(index=False))

# Scatter plot: Degree vs Closeness
plt.figure(figsize=(12, 8))
scatter = plt.scatter(centrality_df['degree'], centrality_df['closeness'], 
                     alpha=0.6, s=30, c=centrality_df['betweenness'], cmap='viridis')

# Find Top-Left quadrant (High Closeness, Low Degree)
top_closeness_ids = centrality_df.nsmallest(20, 'closeness_rank')['id'].values
low_degree_ids = centrality_df[centrality_df['degree_rank'] > 100]['id'].values
tl_candidates = set(top_closeness_ids) & set(low_degree_ids)

tl_df = centrality_df[centrality_df['id'].isin(tl_candidates)]
if len(tl_df) >= 3:
    tl_selected = tl_df.nsmallest(3, 'closeness_rank')
    
    # Annotate on plot
    for idx, row in tl_selected.iterrows():
        plt.annotate(row['page_name'][:20], 
                    xy=(row['degree'], row['closeness']),
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                    fontsize=9)
        plt.scatter(row['degree'], row['closeness'], color='red', s=100, marker='*')

plt.xlabel('Normalized Degree', fontsize=12)
plt.ylabel('Closeness Centrality', fontsize=12)
plt.title('Degree vs Closeness: Efficient Monitors', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Betweenness Centrality')
plt.tight_layout()
plt.savefig('./HW_02/Q1/closeness_vs_degree.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Ego Network Visualization
print("\n3. EGO NETWORK VISUALIZATION")
if len(tl_selected) > 0:
    central_node = tl_selected.iloc[0]['id']
    central_name = tl_selected.iloc[0]['page_name']
    
    # Create ego network (distance 1)
    ego = nx.ego_graph(G, central_node, radius=1)
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(ego, k=2, iterations=50, seed=42)
    
    # Node sizes proportional to degree
    node_sizes = [ego.degree(n) * 50 for n in ego.nodes()]
    node_colors = ['red' if n == central_node else 'steelblue' for n in ego.nodes()]
    
    nx.draw_networkx_nodes(ego, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(ego, pos, alpha=0.3, edge_color='gray')
    
    # Label only central node
    nx.draw_networkx_labels(ego, pos, labels={central_node: central_name[:15]}, 
                           font_size=10, font_weight='bold')
    
    plt.title(f'Ego Network: {central_name}\n(High Closeness, Low Degree)', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./HW_02/Q1/ego_network.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Morphological Analysis
    print(f"\nMorphological Analysis for {central_name}:")
    print(f"• Number of direct connections: {ego.degree(central_node)}")
    print(f"• Clustering coefficient: {nx.clustering(G, central_node):.3f}")
    
    # Check if neighbors are connected to each other
    neighbors = list(ego.neighbors(central_node))
    if len(neighbors) > 1:
        subgraph = G.subgraph(neighbors)
        density = nx.density(subgraph)
        print(f"• Neighbor density: {density:.3f}")
        if density < 0.3:
            print("• Structure: Neighbors dispersed (bridge-like)")
        else:
            print("• Structure: Dense cluster")

# ============================================================================
# PART (d): Bonacich Power Dynamics
# ============================================================================

print("\n\n" + "="*60)
print("PART (d): BONACICH POWER DYNAMICS")
print("="*60)

# Get adjacency matrix
adj_matrix = nx.to_numpy_array(G)

# Calculate largest eigenvalue for convergence check
try:
    eigenvalues, _ = eigsh(adj_matrix, k=1, which='LM')
    lambda_max = eigenvalues[0]
    beta_max = 1 / lambda_max
    print(f"Largest eigenvalue: {lambda_max:.4f}")
    print(f"Maximum beta for convergence: {beta_max:.4f}")
except:
    lambda_max = 2.0  # Conservative estimate
    beta_max = 0.5

# Bonacich power function
def bonacich_power(beta, max_iter=100, tol=1e-6):
    """Calculate Bonacich centrality for given beta"""
    n = len(G)
    I = np.eye(n)
    try:
        if beta >= 0 and abs(beta) < beta_max:
            # For positive beta
            power = np.linalg.inv(I - beta * adj_matrix) @ np.ones(n)
        else:
            # For negative beta or small magnitudes
            power = np.zeros(n)
            x = np.ones(n)
            for _ in range(max_iter):
                x_new = np.ones(n) + beta * adj_matrix @ x
                if np.linalg.norm(x_new - x) < tol:
                    break
                x = x_new
            power = x
    except:
        power = np.ones(n)
    
    return {node: power[i] for i, node in enumerate(G.nodes())}

# Calculate for three regimes
print("\n1. BONACICH POWER REGIMES:")
betas = {
    'neutral': 0.01,      # Near zero
    'supportive': min(0.8 * beta_max, 0.3),  # Positive
    'suppressive': -0.1   # Negative
}

bonacich_scores = {}
for regime, beta in betas.items():
    scores = bonacich_power(beta)
    # Normalize
    max_score = max(scores.values()) if scores else 1
    scores_norm = {k: v/max_score for k, v in scores.items()}
    bonacich_scores[regime] = scores_norm
    
    # Store in DataFrame
    centrality_df[f'bonacich_{regime}'] = [scores_norm.get(n, 0) for n in centrality_df['id']]
    centrality_df[f'bonacich_{regime}_rank'] = centrality_df[f'bonacich_{regime}'].rank(
        ascending=False, method='min')

print(f"Beta values: Neutral={betas['neutral']:.3f}, "
      f"Supportive={betas['supportive']:.3f}, "
      f"Suppressive={betas['suppressive']:.3f}")

# 2. Slope Chart for Top 20 nodes
print("\n2. SLOPE CHART: Rank Trajectories")

# Select top 20 from neutral regime for tracking
top_nodes = centrality_df.nsmallest(20, 'bonacich_neutral_rank')['id'].tolist()
track_df = centrality_df[centrality_df['id'].isin(top_nodes)].copy()

# Prepare data for plotting
regimes = ['neutral', 'supportive', 'suppressive']
ranks_data = []

for node_id in track_df['id'].head(10):  # Plot top 10 for clarity
    node_name = track_df[track_df['id'] == node_id]['page_name'].iloc[0]
    ranks = [track_df[track_df['id'] == node_id][f'bonacich_{r}_rank'].iloc[0] for r in regimes]
    ranks_data.append((node_name, ranks))

# Plot slope chart
plt.figure(figsize=(14, 8))
x_pos = range(len(regimes))
colors = cm.rainbow(np.linspace(0, 1, len(ranks_data)))

for idx, (name, ranks) in enumerate(ranks_data):
    plt.plot(x_pos, ranks, marker='o', linewidth=2, markersize=8, 
            color=colors[idx], label=name[:15])
    
    # Add labels at end points
    plt.annotate(f"{int(ranks[0])}", xy=(x_pos[0], ranks[0]), 
                xytext=(-20, 5), textcoords='offset points', fontsize=9)
    plt.annotate(f"{int(ranks[-1])}", xy=(x_pos[-1], ranks[-1]), 
                xytext=(10, 5), textcoords='offset points', fontsize=9)

plt.xticks(x_pos, ['Neutral\n(β≈0)', 'Supportive\n(β>0)', 'Suppressive\n(β<0)'], fontsize=12)
plt.xlabel('Bonacich Regime', fontsize=12)
plt.ylabel('Rank (lower = more powerful)', fontsize=12)
plt.title('Bonacich Power Dynamics: Rank Trajectories', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # Invert so rank 1 is at top
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./HW_02/Q1/bonacich_slope.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Role Classification
print("\n3. ROLE CLASSIFICATION BASED ON RANK SHIFTS")

# Calculate rank changes
centrality_df['rank_change_supportive'] = (
    centrality_df['bonacich_supportive_rank'] - centrality_df['bonacich_neutral_rank']
)
centrality_df['rank_change_suppressive'] = (
    centrality_df['bonacich_suppressive_rank'] - centrality_df['bonacich_neutral_rank']
)

# Classify nodes
power_amplifiers = centrality_df.nsmallest(10, 'rank_change_supportive')
power_inhibitors = centrality_df.nlargest(10, 'rank_change_supportive')
stable = centrality_df[abs(centrality_df['rank_change_supportive']) < 5].nsmallest(10, 'bonacich_neutral_rank')

print("\nPower Amplifiers (rise in rank with β>0):")
print(power_amplifiers[['page_name', 'bonacich_neutral_rank', 
                       'bonacich_supportive_rank', 'rank_change_supportive']]
      .head(3).round(2).to_string(index=False))

print("\nPower Inhibitors (drop in rank with β>0):")
print(power_inhibitors[['page_name', 'bonacich_neutral_rank', 
                       'bonacich_supportive_rank', 'rank_change_supportive']]
      .head(3).round(2).to_string(index=False))

print("\nStable Actors (minimal change):")
print(stable[['page_name', 'bonacich_neutral_rank', 
             'bonacich_supportive_rank', 'rank_change_supportive']]
      .head(3).round(2).to_string(index=False))

# ============================================================================
# Save Results
# ============================================================================

# Save comprehensive results
centrality_df.to_csv('centrality_results.csv', index=False)
print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Results saved to:")
print(f"1. centrality_results.csv - All centrality metrics")
print(f"2. gap_analysis.png - Degree vs Eigenvector plot")
print(f"3. closeness_vs_degree.png - Efficiency monitor plot")
print(f"4. ego_network.png - Local structure visualization")
print(f"5. bonacich_slope.png - Power dynamics visualization")