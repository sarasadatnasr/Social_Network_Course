import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import networkx as nx

# ---------- helpers ----------
def int_to_bin(n, b):
    return format(n, f"0{b}b")

# ---------- (a) deterministic model ----------
def build_deterministic_sf(b=10):
    N = 2**b
    A = np.zeros((N, N), dtype=bool)

    for i in range(b + 1):
        # Si: i leading zeros
        src_nodes = [u for u in range(N)
                     if int_to_bin(u, b).startswith('0' * i)]

        # Di: (b-i) trailing ones
        if b - i == 0:
            # pattern XXXXX...X (all wildcards) → all nodes
            dst_nodes = list(range(N))
        else:
            pattern = '1' * (b - i)
            dst_nodes = [v for v in range(N)
                         if int_to_bin(v, b).endswith(pattern)]

        for u in src_nodes:
            A[u, dst_nodes] = True

    return A

# Generate random graph model based on parameters
def generate_rg(b, x, r, seed=None):
    if seed is not None:
        np.random.seed(seed)
    N = 2**b
    A = np.zeros((N, N), dtype=bool)

    # Precompute binary strings for speed
    labels = [format(i, f"0{b}b") for i in range(N)]

    for _ in range(r):
        # source pattern
        positions = np.arange(b)
        x_pos = np.random.choice(positions, size=x, replace=False)
        fixed_pos = np.setdiff1d(positions, x_pos)
        fixed_bits = np.random.randint(0, 2, size=fixed_pos.size)

        def match_nodes(fixed_pos, fixed_bits):
            nodes = []
            for idx, lab in enumerate(labels):
                ok = True
                for p, bit in zip(fixed_pos, fixed_bits):
                    if lab[p] != str(bit):
                        ok = False
                        break
                if ok:
                    nodes.append(idx)
            return nodes

        src_nodes = match_nodes(fixed_pos, fixed_bits)

        # destination pattern (independent)
        positions = np.arange(b)
        x_pos_d = np.random.choice(positions, size=x, replace=False)
        fixed_pos_d = np.setdiff1d(positions, x_pos_d)
        fixed_bits_d = np.random.randint(0, 2, size=fixed_pos_d.size)
        dst_nodes = match_nodes(fixed_pos_d, fixed_bits_d)

        for u in src_nodes:
            A[u, dst_nodes] = True

    return A

# Calculate density of graph
def density(A):
    N = A.shape[0]
    return A.sum() / (N * N)

# --------- Main part of the code ---------
b = 10  # Size parameter
x_vals = np.arange(1, 9)  # 1..8
r_vals = np.unique(np.round(np.logspace(np.log10(2), np.log10(160), 10)).astype(int))

# Initialize dens_sim matrix
dens_sim = np.zeros((len(x_vals), len(r_vals)))
dens_theo = np.zeros_like(dens_sim)

# Calculate density for each (x, r) combination
for i, x in enumerate(x_vals):
    pi = 1.0 / (2 ** (b - x))  # π = 2^{x-b}
    for j, r in enumerate(r_vals):
        A_rg = generate_rg(b, x, r)
        dens_sim[i, j] = density(A_rg)
        dens_theo[i, j] = 1 - (1 - pi**2)**(2 * r)


# degrees from deterministic scale-free model
A = build_deterministic_sf(b)
out_deg = A.sum(axis=1)
in_deg = A.sum(axis=0)

# Log-log degree distribution plot for out-degree and in-degree
def loglog_degree_fit(deg, title):
    deg = deg[deg > 0]
    vals, counts = np.unique(deg, return_counts=True)
    pk = counts / counts.sum()

    plt.figure()
    plt.loglog(vals, pk, 'o')
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.title(title)

    # Linear fit on log–log
    x = np.log10(vals)
    y = np.log10(pk)
    slope, intercept, r, p, se = linregress(x, y)
    gamma = -slope
    print(f"{title}: slope={slope:.2f}, gamma≈{gamma:.2f}, R²≈{r**2:.3f}")

    kfit = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 50)
    yfit = intercept + slope * np.log10(kfit)
    plt.loglog(kfit, 10**yfit, '-r')
    plt.savefig(f"Q2/{title.replace(' ', '_').lower()}.png")  # Save plot as .png

loglog_degree_fit(out_deg, "Out-degree distribution")
loglog_degree_fit(in_deg, "In-degree distribution")
plt.show()

# Heatmap of network density (Simulated)
plt.figure(figsize=(6, 4))
plt.imshow(dens_sim, origin='lower',
           extent=[r_vals[0], r_vals[-1], x_vals[0], x_vals[-1]],
           aspect='auto')
plt.colorbar(label="density")
plt.xlabel("r (rules, log-spaced)")
plt.ylabel("x (wildcards)")
plt.title("RG network density (simulation)")
plt.tight_layout()
plt.savefig("Q2/rg_network_density_simulation.png")  # Save plot as .png
plt.show()

# Plot panels for (d)–(f): Degree distribution and example networks
def plot_deg_panel(b, x, r, title):
    A = generate_rg(b, x, r)
    G = nx.from_numpy_array(A)

    # Degree distribution (CCDF)
    degs = np.array([d for _, d in G.degree()])
    k_vals, counts = np.unique(degs, return_counts=True)
    ccdf = np.array([np.sum(degs >= k) / len(degs) for k in k_vals])

    plt.figure(figsize=(5, 3))
    plt.loglog(k_vals, ccdf, 'o', label="sim")
    plt.xlabel("k")
    plt.ylabel("1-CDF(k)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"Q2/{title.replace(' ', '_').lower()}_deg_dist.png")  # Save plot as .png

    # Network inset
    plt.figure(figsize=(3, 3))
    pos = nx.spring_layout(G, seed=1)
    nx.draw_networkx_nodes(G, pos, node_size=10)
    nx.draw_networkx_edges(G, pos, width=0.3)
    plt.axis('off')
    plt.title(f"{title} (example network)")
    plt.tight_layout()
    plt.savefig(f"Q2/{title.replace(' ', '_').lower()}_network.png")  # Save plot as .png

# Choose parameter sets representing subcritical / supercritical / connected
plot_deg_panel(b=11, x=3, r=15, title="d) Subcritical")
plot_deg_panel(b=11, x=4, r=30, title="e) Supercritical")
plot_deg_panel(b=11, x=5, r=70, title="f) Connected")

# Plot robustness curves for RG vs ER
def robustness_curves(G, mode="random"):
    """Return sizes of giant component as nodes removed (fraction)."""
    G = G.copy()
    N = G.number_of_nodes()
    order = list(G.nodes())

    if mode == "targeted":
        # repeatedly choose current highest-degree node
        order = []
        H = G.copy()
        while H.number_of_nodes() > 0:
            v = max(H.degree, key=lambda x: x[1])[0]
            order.append(v)
            H.remove_node(v)

    sizes = []
    Gtemp = G.copy()
    for i, v in enumerate(order):
        if Gtemp.number_of_nodes() == 0:
            sizes.append(0.0)
            continue
        comps = nx.connected_components(Gtemp)
        lcc_size = max(len(c) for c in comps)
        sizes.append(lcc_size / N)
        Gtemp.remove_node(v)
    x = np.linspace(0, 1, len(sizes))
    return x, np.array(sizes)

def plot_robustness_pair(b, x, r, title_suffix):
    A_rg = generate_rg(b, x, r)
    G_rg = nx.from_numpy_array(A_rg)

    # ER with same N and density
    N = 2**b
    p = density(A_rg)
    G_er = nx.erdos_renyi_graph(N, p)

    fig, axes = plt.subplots(2, 1, figsize=(4, 5), sharex=True)
    for mode, row in zip(["random", "targeted"], [0, 1]):
        xr, yr = robustness_curves(G_rg, mode=mode)
        xe, ye = robustness_curves(G_er, mode=mode)
        axes[row].plot(xr, yr, color='blue', label='RG')
        axes[row].plot(xe, ye, color='orange', label='ER')
        axes[row].set_ylabel("Size of Giant Component")
        axes[row].set_title(f"{'Random' if mode == 'random' else 'Targeted'} Removal")
        axes[row].set_ylim(0, 1.05)

    axes[1].set_xlabel("Proportion of Nodes Removed")
    axes[0].legend(loc='upper right')
    fig.suptitle(title_suffix)
    plt.tight_layout()
    plt.savefig(f"Q2/robustness_{title_suffix.replace(' ', '_').lower()}.png")  # Save plot as .png
    plt.show()

plot_robustness_pair(b=11, x=3, r=15, title_suffix="g) Subcritical")
plot_robustness_pair(b=11, x=4, r=30, title_suffix="g) Supercritical")
plot_robustness_pair(b=11, x=5, r=70, title_suffix="g) Connected")


# ---------- Spy plot for RG adjacency matrix ----------
def plot_rg_adjacency_spy(b=10, x=5, r=32, seed=42, figsize=(8, 6)):
    """
    Generate and plot spy plot (heatmap) of RG adjacency matrix.
    Similar to the provided image: 'RG sample adjacency (spy) - b=10, x=5, R=32'
    """
    # Generate RG network
    A = generate_rg(b, x, r, seed=seed)
    N = 2**b
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use colormap similar to first plots
    # Using 'viridis' colormap which is colorful and matches matplotlib defaults
    cmap = plt.cm.viridis
    
    # Plot the adjacency matrix
    im = ax.imshow(A, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Set labels and title
    ax.set_xlabel('Destination node index', fontsize=12)
    ax.set_ylabel('Source node index', fontsize=12)
    
    # Add grid lines for better visibility
    ax.set_xticks(np.linspace(0, N-1, 6, dtype=int))
    ax.set_yticks(np.linspace(0, N-1, 6, dtype=int))
    
    # Format tick labels
    ax.set_xticklabels([f'{int(tick)}' for tick in ax.get_xticks()])
    ax.set_yticklabels([f'{int(tick)}' for tick in ax.get_yticks()])
    
    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Edge presence', fontsize=10)
    
    # Add title with parameters
    ax.set_title(f'RG Adjacency Matrix (Spy Plot)\nb={b}, x={x}, r={r}, N={N}', 
                 fontsize=14, pad=15)
    
    # Add grid for better readability
    ax.grid(True, which='both', color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'Density: {density(A):.4f}\n'
    stats_text += f'Total edges: {A.sum():,}'
    
    # Place text box in upper left
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig, ax, A

# ---------- Alternative spy plot using scatter (for sparse matrices) ----------
def plot_rg_adjacency_sparse_spy(b=10, x=5, r=32, seed=42, figsize=(8, 6), marker_size=1):
    """
    Generate sparse spy plot using scatter plot - more efficient for large sparse matrices.
    This style is closer to traditional spy plots in MATLAB.
    """
    # Generate RG network
    A = generate_rg(b, x, r, seed=seed)
    N = 2**b
    
    # Get coordinates of non-zero entries
    rows, cols = np.where(A)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use a color from viridis colormap (matching the first plots)
    color_map = plt.cm.viridis(0.7)  # Get a specific color from viridis colormap
    
    # Plot only the non-zero entries
    ax.scatter(cols, rows, s=marker_size, c=[color_map], marker='s', alpha=0.7)
    
    # Invert y-axis to match matrix convention
    ax.invert_yaxis()
    
    # Set labels and title
    ax.set_xlabel('Destination node index', fontsize=12)
    ax.set_ylabel('Source node index', fontsize=12)
    
    # Set limits
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(N-0.5, -0.5)
    
    # Add grid
    ax.grid(True, which='both', color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Set ticks
    tick_positions = np.linspace(0, N-1, 6, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    
    # Format tick labels
    ax.set_xticklabels([f'{pos}' for pos in tick_positions])
    ax.set_yticklabels([f'{pos}' for pos in tick_positions])
    
    ax.set_title(f'RG Adjacency Spy Plot\nb={b}, x={x}, r={r}, N={N}', 
                 fontsize=14, pad=15)
    
    # Add statistics
    density_val = density(A)
    stats_text = f'Density: {density_val:.4f}\n'
    stats_text += f'Edges: {A.sum():,}\n'
    stats_text += f'Non-zero: {len(rows):,}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig, ax, A

# ---------- Compare multiple RG networks ----------
def plot_multiple_rg_spy_plots(params_list, figsize=(15, 10)):
    """
    Plot multiple RG adjacency matrices for comparison.
    params_list: list of tuples (b, x, r, seed, title_suffix)
    """
    n_plots = len(params_list)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                             constrained_layout=True)
    
    # Flatten axes array for easier indexing
    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Use a colormap that matches your first plots
    cmap = plt.cm.viridis  # Change to your preferred colormap
    
    for idx, (b, x, r, seed, title_suffix) in enumerate(params_list):
        ax = axes[idx]
        A = generate_rg(b, x, r, seed=seed)
        N = 2**b
        
        # Use imshow with colormap
        im = ax.imshow(A, cmap=cmap, aspect='auto', interpolation='nearest')
        
        # Simple grid
        ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        
        # Set title with parameters
        ax.set_title(f'{title_suffix}\nb={b}, x={x}, r={r}, ρ={density(A):.4f}', 
                     fontsize=10)
        
        # Only show ticks for first row and first column
        if idx >= n_plots - n_cols:  # Last row
            ax.set_xlabel('Dest idx')
            tick_positions = np.linspace(0, N-1, 3, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f'{int(t)}' for t in tick_positions])
        else:
            ax.set_xticks([])
            
        if idx % n_cols == 0:  # First column
            ax.set_ylabel('Src idx')
            tick_positions = np.linspace(0, N-1, 3, dtype=int)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels([f'{int(t)}' for t in tick_positions])
        else:
            ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    fig.suptitle('RG Network Adjacency Matrices Comparison', fontsize=16, y=1.02)
    
    # Add a single colorbar for all subplots
    fig.colorbar(im, ax=axes, shrink=0.8, label='Edge presence')
    
    return fig, axes

# ---------- Add to your main execution ----------
# Add this to your main code after generating other plots:

print("\n=== Generating RG Adjacency Spy Plots ===")

# Plot 1: The specific case from your image (b=10, x=5, r=32)
print("Generating spy plot for b=10, x=5, r=32...")
fig1, ax1, A1 = plot_rg_adjacency_spy(b=10, x=5, r=32, seed=42)
plt.savefig("Q2/rg_adjacency_spy_b10_x5_r32.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 2: Sparse version
print("Generating sparse spy plot...")
fig2, ax2, A2 = plot_rg_adjacency_sparse_spy(b=10, x=5, r=32, seed=42, marker_size=0.5)
plt.savefig("Q2/rg_adjacency_sparse_spy_b10_x5_r32.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 3: Comparison of different parameter sets
print("Generating comparison plots...")
params_comparison = [
    (10, 3, 16, 1, "Sparse RG"),
    (10, 5, 32, 2, "Medium RG"),
    (10, 7, 64, 3, "Dense RG"),
    (10, 4, 20, 4, "Low wildcards"),
    (10, 6, 40, 5, "High wildcards"),
    (10, 5, 100, 6, "Many rules")
]

fig3, axes3 = plot_multiple_rg_spy_plots(params_comparison, figsize=(15, 8))
plt.savefig("Q2/rg_adjacency_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# ---------- Additional analysis: Pattern visualization ----------
def visualize_pattern_rule(b=10, x=5, rule_idx=0, seed=42):
    """
    Visualize a specific rule's matching nodes.
    """
    if seed is not None:
        np.random.seed(seed)
    
    N = 2**b
    labels = [format(i, f"0{b}b") for i in range(N)]
    
    # Generate a specific rule
    positions = np.arange(b)
    x_pos = np.random.choice(positions, size=x, replace=False)
    fixed_pos = np.setdiff1d(positions, x_pos)
    fixed_bits = np.random.randint(0, 2, size=fixed_pos.size)
    
    # Create pattern strings
    pattern_chars = ['X'] * b
    for p, bit in zip(fixed_pos, fixed_bits):
        pattern_chars[p] = str(bit)
    pattern = ''.join(pattern_chars)
    
    print(f"Rule Pattern: {pattern}")
    print(f"Wildcard positions: {sorted(x_pos)}")
    print(f"Fixed positions: {sorted(fixed_pos)} with bits {fixed_bits}")
    
    # Find matching nodes
    matching_nodes = []
    for idx, lab in enumerate(labels):
        ok = True
        for p in range(b):
            if pattern[p] != 'X' and pattern[p] != lab[p]:
                ok = False
                break
        if ok:
            matching_nodes.append(idx)
    
    print(f"\nMatching nodes: {len(matching_nodes)} out of {N} ({len(matching_nodes)/N*100:.2f}%)")
    print(f"Sample matching nodes (first 5): {matching_nodes[:5]}")
    print(f"Corresponding binary strings: {[labels[i] for i in matching_nodes[:5]]}")
    
    # Visualize the matching pattern
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Create a matrix showing the pattern
    pattern_matrix = np.zeros((1, b))
    for i in range(b):
        if pattern[i] == 'X':
            pattern_matrix[0, i] = 0.5  # Gray for wildcard
        elif pattern[i] == '0':
            pattern_matrix[0, i] = 0.0  # Black for 0
        else:
            pattern_matrix[0, i] = 1.0  # White for 1
    
    im = ax.imshow(pattern_matrix, cmap='gray', aspect='auto')
    ax.set_xticks(range(b))
    ax.set_xticklabels([str(i) for i in range(b)])
    ax.set_yticks([])
    ax.set_title(f"Pattern Visualization: {pattern}\n(Black=0, White=1, Gray=X)")
    
    # Add text annotations
    for i in range(b):
        text_color = 'white' if pattern_matrix[0, i] < 0.3 else 'black'
        ax.text(i, 0, pattern[i], ha='center', va='center', 
                color=text_color, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("Q2/pattern_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return pattern, matching_nodes

# Visualize a sample rule pattern
print("\n=== Visualizing a Sample Rule Pattern ===")
pattern, matches = visualize_pattern_rule(b=10, x=5, rule_idx=0, seed=42)

# ---------- Degree correlation analysis for RG networks ----------
def analyze_rg_degree_correlation(b=10, x=5, r=32, seed=42):
    """Analyze degree correlations in RG network."""
    A = generate_rg(b, x, r, seed=seed)
    
    # Calculate degrees
    out_deg = A.sum(axis=1)
    in_deg = A.sum(axis=0)
    
    # Create NetworkX graph for analysis
    G = nx.DiGraph(A)
    
    # Calculate degree correlations
    if len(G.edges()) > 0:
        # In-in degree correlation
        in_in_pairs = [(in_deg[u], in_deg[v]) for u, v in G.edges()]
        in_in_corr = np.corrcoef([p[0] for p in in_in_pairs], 
                                 [p[1] for p in in_in_pairs])[0, 1]
        
        # Out-out degree correlation
        out_out_pairs = [(out_deg[u], out_deg[v]) for u, v in G.edges()]
        out_out_corr = np.corrcoef([p[0] for p in out_out_pairs], 
                                   [p[1] for p in out_out_pairs])[0, 1]
    else:
        in_in_corr = out_out_corr = np.nan
    
    print(f"\n=== Degree Correlation Analysis (b={b}, x={x}, r={r}) ===")
    print(f"In-degree correlation: {in_in_corr:.4f}")
    print(f"Out-degree correlation: {out_out_corr:.4f}")
    print(f"Average in-degree: {np.mean(in_deg):.2f}")
    print(f"Average out-degree: {np.mean(out_deg):.2f}")
    print(f"In-degree variance: {np.var(in_deg):.2f}")
    print(f"Out-degree variance: {np.var(out_deg):.2f}")
    
    # Plot degree-degree scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if len(G.edges()) > 0:
        # In-in scatter
        in_sources = [in_deg[u] for u, v in G.edges()]
        in_targets = [in_deg[v] for u, v in G.edges()]
        axes[0].scatter(in_sources, in_targets, alpha=0.5, s=10)
        axes[0].set_xlabel('Source node in-degree')
        axes[0].set_ylabel('Target node in-degree')
        axes[0].set_title(f'In-In Degree Correlation: {in_in_corr:.3f}')
        
        # Out-out scatter
        out_sources = [out_deg[u] for u, v in G.edges()]
        out_targets = [out_deg[v] for u, v in G.edges()]
        axes[1].scatter(out_sources, out_targets, alpha=0.5, s=10, color='red')
        axes[1].set_xlabel('Source node out-degree')
        axes[1].set_ylabel('Target node out-degree')
        axes[1].set_title(f'Out-Out Degree Correlation: {out_out_corr:.3f}')
    
    plt.tight_layout()
    plt.savefig("Q2/rg_degree_correlation.png", dpi=150, bbox_inches='tight')
    plt.show()

# Analyze degree correlations
analyze_rg_degree_correlation(b=10, x=5, r=32, seed=42)

print("\n=== All plots have been generated and saved to Q2/ directory ===")