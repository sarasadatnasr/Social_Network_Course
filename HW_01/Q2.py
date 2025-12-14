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
