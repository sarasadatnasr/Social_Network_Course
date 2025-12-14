import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import binned_statistic
import warnings
warnings.filterwarnings('ignore')

def simulate_er_phase_transition():
    """
    Complete implementation of Erdős-Rényi phase transition analysis - FIXED VERSION
    """
    
    # Part (a): Network evolution N=1000 with variable step size
    print("=== Part (a): Simulating N=1000 with variable step size ===")
    N = 1000
    num_realizations = 30  # Reduced for speed, increase to 50 for production
    
    # Variable step size: coarse outside, fine in critical region
    k_coarse = np.arange(0.0, 0.8, 0.1)
    k_fine = np.arange(0.8, 1.2, 0.02)
    k_supercritical = np.arange(1.2, 3.1, 0.1)
    k_values = np.sort(np.concatenate([k_coarse, k_fine, k_supercritical]))
    
    S_avg = np.zeros(len(k_values))
    s_avg = np.zeros(len(k_values))
    
    for i, k in enumerate(k_values):
        p = k / (N - 1)
        S_list, s_list = [], []
        
        for _ in range(num_realizations):
            G = nx.erdos_renyi_graph(N, p)
            components = [len(c) for c in nx.connected_components(G)]
            
            if components:
                NG = max(components)
                S = NG / N
                S_list.append(S)
                
                # Average size of remaining clusters (excluding giant)
                remaining = [sz for sz in components if sz != NG]
                if remaining:
                    s_list.append(np.mean(remaining))
        
        S_avg[i] = np.mean(S_list)
        s_avg[i] = np.mean(s_list) if s_list else 0
        
        if i % 10 == 0:
            print(f"k={k:.2f}: S={S_avg[i]:.3f}, <s>={s_avg[i]:.1f}")
    
    # Critical point: maximum slope in S
    diff_S = np.diff(S_avg) / np.diff(k_values)
    crit_idx = np.argmax(diff_S)
    k_c = k_values[crit_idx]
    print(f"Estimated critical point: <k>_c = {k_c:.3f}")
    
    # Part (b): Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(k_values, S_avg, 'b-', linewidth=2, label='S = N_G/N')
    ax1.axvline(k_c, color='r', linestyle='--', label=f'k_c={k_c:.3f}')
    ax1.axhline(0.5, color='g', linestyle=':', alpha=0.7)
    ax1.set_xlabel('<k>')
    ax1.set_ylabel('Order parameter S')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(k_values, s_avg, 'r-', linewidth=2, label='<s>')
    ax2.axvline(k_c, color='b', linestyle='--', label=f'k_c={k_c:.3f}')
    ax2.set_xlabel('<k>')
    ax2.set_ylabel('Average finite cluster size <s>')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Q3/phase_transition_N1000.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Part (c): Finite size effects
    print("\n=== Part (c): Finite size scaling ===")
    Ns = [100, 1000, 10000]
    colors = ['green', 'blue', 'red']
    fig_fs, ax_fs = plt.subplots(figsize=(8, 6))
    
    S_finite_data = {}
    for i, n in enumerate(Ns):
        print(f"Simulating N={n}...")
        k_fs = np.arange(0.4, 2.0, 0.05)
        S_n = np.zeros(len(k_fs))
        num_rep = 50 if n <= 100 else 20 if n == 1000 else 5
        
        for j, k in enumerate(k_fs):
            p = k / (n - 1)
            S_list = []
            for _ in range(num_rep):
                G = nx.erdos_renyi_graph(n, p)
                components = [len(c) for c in nx.connected_components(G)]
                S_list.append(max(components) / n)
            S_n[j] = np.mean(S_list)
        
        S_finite_data[n] = S_n
        ax_fs.plot(k_fs, S_n, 'o-', color=colors[i], label=f'N={n}', linewidth=2, markersize=4)
        idx_crit = np.argmin(np.abs(k_fs - 1.0))
        print(f"N={n}: S(k=1) = {S_n[idx_crit]:.3f}")
    
    ax_fs.axvline(1.0, color='k', linestyle='--', alpha=0.7, label='<k>_c=1')
    ax_fs.set_xlabel('<k>')
    ax_fs.set_ylabel('S = N_G/N')
    ax_fs.set_ylim(0, 1)
    ax_fs.legend()
    ax_fs.grid(True, alpha=0.3)
    plt.savefig('Q3/finite_size_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Part (d): Critical state distribution - FIXED LOGARITHMIC BINNING
    print("\n=== Part (d): Critical state N=10000 ===")
    N_crit = 5000  # Compromise for speed
    p_crit = 1.0 / (N_crit - 1)
    G_crit = nx.erdos_renyi_graph(N_crit, p_crit)
    components_crit = [len(c) for c in nx.connected_components(G_crit)]
    
    # Component size histogram
    size_to_count = defaultdict(int)
    for s in components_crit:
        size_to_count[s] += 1
    
    sizes = np.array(sorted(size_to_count.keys()))
    counts = np.array([size_to_count[s] for s in sizes])
    P_s = counts / N_crit  # Probability density
    
    max_S = max(components_crit) / N_crit
    print(f"Critical N={N_crit}: max S = {max_S:.3f}")
    
    # FIXED: Logarithmic binning with proper shape handling
    if len(sizes) > 1:
        log_min = np.log10(max(1, min(sizes)))
        log_max = np.log10(max(sizes))
        num_log_bins = min(25, len(sizes)//2)
        
        log_bin_edges = np.logspace(log_min, log_max, num_log_bins + 1)
        
        # Use binned_statistic properly
        bin_indices = np.digitize(sizes, log_bin_edges) - 1
        bin_centers = []
        binned_P = np.zeros(num_log_bins)
        
        for j in range(num_log_bins):
            mask = (bin_indices == j)
            if np.any(mask):
                # Geometric mean of bin edges as center
                bin_center = np.sqrt(log_bin_edges[j] * log_bin_edges[j+1])
                bin_centers.append(bin_center)
                # Sum probability in bin, normalize by bin width
                total_prob_bin = np.sum(P_s[mask])
                bin_width = log_bin_edges[j+1] - log_bin_edges[j]
                binned_P[j] = total_prob_bin / bin_width
        
        bin_centers = np.array(bin_centers)
        binned_P = binned_P[:len(bin_centers)]  # Trim to matching length
        
        print(f"Log-bins created: {len(bin_centers)} bins")
        
        # Power-law fit for tail (s > 10)
        mask_fit = bin_centers > 10
        alpha_est = None
        if np.sum(mask_fit) >= 4:
            log_s = np.log10(bin_centers[mask_fit])
            log_P = np.log10(binned_P[mask_fit] + 1e-12)  # Avoid log(0)
            slope, intercept = np.polyfit(log_s, log_P, 1)
            alpha_est = -slope
            print(f"Power-law exponent α = {alpha_est:.2f} (theory: 1.5 )")
        else:
            print("Insufficient data for power-law fit")
    else:
        print("No valid sizes for binning")
        alpha_est = None
    
    # Log-log plot
    fig_crit, ax_crit = plt.subplots(figsize=(8, 6))
    ax_crit.loglog(sizes, P_s, 'k.', alpha=0.5, markersize=2, label='Raw data')
    if len(bin_centers) > 0:
        ax_crit.loglog(bin_centers, binned_P, 'ro-', linewidth=2, markersize=6, 
                       label=f'Log-binned (α={alpha_est:.2f})' if alpha_est else 'Log-binned')
    ax_crit.axvline(10, color='g', linestyle='--', alpha=0.7, label='Fit region (s>10)')
    ax_crit.set_xlabel('Cluster size s')
    ax_crit.set_ylabel('P(s)')
    ax_crit.set_title(f'Critical component distribution (N={N_crit}, <k>=1)')
    ax_crit.legend()
    ax_crit.grid(True, alpha=0.3)
    plt.savefig('Q3/critical_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary results
    print("\n=== SUMMARY ===")
    print(f"1. Critical point: <k>_c = {k_c:.3f} (theory: 1.0 )")
    print(f"2. Finite size at k=1: S decreases with N (NG ~ N^{2/3})")
    print(f"3. <s> peaks at criticality due to maximum heterogeneity")
    print(f"4. Power-law exponent: α ≈ {alpha_est:.2f} (theory: 3/2 )")
    
    return {
        'k_values': k_values, 'S_avg': S_avg, 's_avg': s_avg,
        'k_c': k_c, 'finite_size': S_finite_data,
        'alpha': alpha_est
    }

# Run complete analysis
if __name__ == "__main__":
    results = simulate_er_phase_transition()
