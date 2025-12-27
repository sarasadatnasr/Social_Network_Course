import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
import time
import warnings
import os
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')

# Create output directory for plots
os.makedirs('Q1', exist_ok=True)

def create_1d_lattice(N: int, k: int = 2) -> nx.Graph:
    """Create 1D ring lattice with k nearest neighbors"""
    G = nx.cycle_graph(N)
    return G

def create_2d_lattice(N: int) -> nx.Graph:
    """Create 2D square lattice on a torus"""
    # Find dimensions closest to square
    side = int(np.sqrt(N))
    while N % side != 0:
        side -= 1
    rows, cols = side, N // side
    
    # Create grid graph with periodic boundaries
    G = nx.grid_2d_graph(rows, cols, periodic=True)
    G = nx.convert_node_labels_to_integers(G)
    return G

def create_3d_lattice(N: int) -> nx.Graph:
    """Create 3D cubic lattice"""
    # Find cube-like dimensions
    side = int(np.cbrt(N))
    for s in range(side, 0, -1):
        if N % (s**2) == 0:
            dim3 = N // (s**2)
            if s * s * dim3 == N:
                dim1, dim2, dim3 = s, s, dim3
                break
    else:
        # Approximate factorization
        for d1 in range(int(np.cbrt(N)), 0, -1):
            if N % d1 == 0:
                remaining = N // d1
                for d2 in range(int(np.sqrt(remaining)), 0, -1):
                    if remaining % d2 == 0:
                        dim1, dim2, dim3 = d1, d2, remaining // d2
                        break
                break
    
    G = nx.Graph()
    
    # Add nodes and edges with periodic boundaries
    for x in range(dim1):
        for y in range(dim2):
            for z in range(dim3):
                node_id = x * dim2 * dim3 + y * dim3 + z
                
                # Add edges to 6 neighbors
                # x-direction
                right_id = ((x + 1) % dim1) * dim2 * dim3 + y * dim3 + z
                G.add_edge(node_id, right_id)
                
                # y-direction
                down_id = x * dim2 * dim3 + ((y + 1) % dim2) * dim3 + z
                G.add_edge(node_id, down_id)
                
                # z-direction
                back_id = x * dim2 * dim3 + y * dim3 + ((z + 1) % dim3)
                G.add_edge(node_id, back_id)
    
    return G

def create_random_network(N: int, k_avg: float = 4) -> nx.Graph:
    """Create Erdős–Rényi random graph with fixed average degree"""
    p = k_avg / (N - 1)
    
    # Create connected graph
    for _ in range(5):  # Try 5 times
        G = nx.erdos_renyi_graph(N, p)
        if nx.is_connected(G):
            return G
    
    # If not connected, use largest component
    G = nx.erdos_renyi_graph(N, p)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    return G

def compute_average_path_length(G: nx.Graph, sample_nodes: int = 200) -> float:
    """Compute average shortest path length efficiently"""
    if len(G) <= 1000:
        try:
            return nx.average_shortest_path_length(G)
        except:
            pass
    
    # Sample nodes for larger graphs
    nodes = list(G.nodes())
    if sample_nodes > len(nodes):
        sample_nodes = len(nodes)
    
    sampled_nodes = np.random.choice(nodes, size=min(sample_nodes, 100), replace=False)
    total_length = 0
    total_paths = 0
    
    for source in sampled_nodes:
        lengths = nx.single_source_shortest_path_length(G, source)
        total_length += sum(lengths.values())
        total_paths += len(lengths) - 1
    
    return total_length / total_paths if total_paths > 0 else 0

def run_simulations(N_values: List[int]) -> Dict[str, List[float]]:
    """Run simulations for all network types and sizes"""
    results = {
        '1D': [],
        '2D': [],
        '3D': [],
        'Random': []
    }
    
    for N in N_values:
        print(f"Processing N = {N:4d}", end=" | ")
        
        # 1D Lattice
        G = create_1d_lattice(N, k=2)
        d1 = compute_average_path_length(G)
        results['1D'].append(d1)
        print(f"1D: {d1:6.2f}", end=" | ")
        
        # 2D Lattice
        G = create_2d_lattice(N)
        d2 = compute_average_path_length(G)
        results['2D'].append(d2)
        print(f"2D: {d2:6.2f}", end=" | ")
        
        # 3D Lattice
        G = create_3d_lattice(N)
        d3 = compute_average_path_length(G)
        results['3D'].append(d3)
        print(f"3D: {d3:6.2f}", end=" | ")
        
        # Random Network (average of 3 samples)
        d_vals = []
        for _ in range(3):
            G = create_random_network(N, k_avg=4)
            d_vals.append(compute_average_path_length(G))
        d_rand = np.mean(d_vals)
        results['Random'].append(d_rand)
        print(f"RN: {d_rand:6.2f}")
    
    return results

def save_plot(fig, filename: str):
    """Save figure as PNG in full screen mode"""
    # Get screen dimensions
    screen_dpi = 100  # Default DPI
    screen_width = 1920  # Default screen width in pixels
    screen_height = 1080  # Default screen height in pixels
    
    # Convert to inches for matplotlib
    width_inches = screen_width / screen_dpi
    height_inches = screen_height / screen_dpi
    
    # Resize figure to full screen
    fig.set_size_inches(width_inches, height_inches)
    
    # Save with high resolution
    filepath = os.path.join('Q1', filename)
    fig.savefig(filepath, dpi=screen_dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filepath}")

def create_individual_plots(N_values: List[int], results: Dict[str, List[float]]):
    """Create and save individual plots as separate PNG files"""
    
    N_array = np.array(N_values)
    log_N = np.log10(N_array)
    
    colors = {
        '1D': '#E41A1C',    # Red
        '2D': '#377EB8',    # Blue
        '3D': '#4DAF4A',    # Green
        'Random': '#984EA3' # Purple
    }
    
    markers = {
        '1D': 'o',
        '2D': 's',
        '3D': '^',
        'Random': 'D'
    }
    
    # 1. 1D Lattice Plot
    fig1, ax1 = plt.subplots(figsize=(16, 12))
    ax1.plot(N_array, results['1D'], 'o-', color=colors['1D'], linewidth=2, markersize=6, label='Simulation')
    
    # Add theoretical scaling line
    N_fit = np.linspace(min(N_array), max(N_array), 100)
    coeff = np.polyfit(N_array, results['1D'], 1)
    theoretical_line = coeff[0] * N_fit
    ax1.plot(N_fit, theoretical_line, '--', color='black', linewidth=1.5, alpha=0.7, 
             label=r'$\langle d \rangle \sim N$')
    
    ax1.set_xlabel('$N$', fontsize=14)
    ax1.set_ylabel(r'$\langle d \rangle$', fontsize=14)
    ax1.set_title('1D Lattice: Average Path Length Scaling', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=12, loc='upper left')
    ax1.tick_params(labelsize=12)
    
    save_plot(fig1, '1D_lattice_scaling.png')
    plt.close(fig1)
    
    # 2. 2D Lattice Plot
    fig2, ax2 = plt.subplots(figsize=(16, 12))
    ax2.plot(N_array, results['2D'], 's-', color=colors['2D'], linewidth=2, markersize=6, label='Simulation')
    
    # Add theoretical scaling
    log_d_2D = np.log10(results['2D'])
    coeff_2D = np.polyfit(log_N, log_d_2D, 1)
    theoretical_2D = 10**(coeff_2D[1]) * N_fit**(1/2)
    ax2.plot(N_fit, theoretical_2D, '--', color='black', linewidth=1.5, alpha=0.7, 
             label=r'$\langle d \rangle \sim N^{1/2}$')
    
    ax2.set_xlabel('$N$', fontsize=14)
    ax2.set_ylabel(r'$\langle d \rangle$', fontsize=14)
    ax2.set_title('2D Lattice: Average Path Length Scaling', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=12, loc='upper left')
    ax2.tick_params(labelsize=12)
    
    save_plot(fig2, '2D_lattice_scaling.png')
    plt.close(fig2)
    
    # 3. 3D Lattice Plot
    fig3, ax3 = plt.subplots(figsize=(16, 12))
    ax3.plot(N_array, results['3D'], '^-', color=colors['3D'], linewidth=2, markersize=6, label='Simulation')
    
    # Add theoretical scaling
    log_d_3D = np.log10(results['3D'])
    coeff_3D = np.polyfit(log_N, log_d_3D, 1)
    theoretical_3D = 10**(coeff_3D[1]) * N_fit**(1/3)
    ax3.plot(N_fit, theoretical_3D, '--', color='black', linewidth=1.5, alpha=0.7,
             label=r'$\langle d \rangle \sim N^{1/3}$')
    
    ax3.set_xlabel('$N$', fontsize=14)
    ax3.set_ylabel(r'$\langle d \rangle$', fontsize=14)
    ax3.set_title('3D Lattice: Average Path Length Scaling', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=12, loc='upper left')
    ax3.tick_params(labelsize=12)
    
    save_plot(fig3, '3D_lattice_scaling.png')
    plt.close(fig3)
    
    # 4. Random Network Plot
    fig4, ax4 = plt.subplots(figsize=(16, 12))
    ax4.plot(N_array, results['Random'], 'D-', color=colors['Random'], linewidth=2, markersize=6, label='Simulation')
    
    # Add theoretical scaling
    coeff_log = np.polyfit(log_N, results['Random'], 1)
    theoretical_log = coeff_log[0] * np.log10(N_fit) + coeff_log[1]
    ax4.plot(N_fit, theoretical_log, '--', color='black', linewidth=1.5, alpha=0.7,
             label=r'$\langle d \rangle \sim \log N$')
    
    ax4.set_xlabel('$N$', fontsize=14)
    ax4.set_ylabel(r'$\langle d \rangle$', fontsize=14)
    ax4.set_title('Random Network: Average Path Length Scaling', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=12, loc='upper left')
    ax4.tick_params(labelsize=12)
    
    save_plot(fig4, 'random_network_scaling.png')
    plt.close(fig4)
    
    # 5. LOG-LOG PLOT: log⟨d⟩ vs logN (All networks)
    fig5, ax5 = plt.subplots(figsize=(16, 12))
    
    # Plot all networks on log-log scale
    for network_type in ['1D', '2D', '3D', 'Random']:
        log_d = np.log10(results[network_type])
        ax5.plot(log_N, log_d, 
                marker=markers[network_type], 
                color=colors[network_type], 
                linestyle='-', 
                linewidth=2, 
                markersize=7,
                label=f'{network_type} {"Lattice" if network_type != "Random" else "Network"}',
                alpha=0.8)
    
    # Add theoretical lines with specific slopes
    x_log_fit = np.linspace(min(log_N), max(log_N), 100)
    
    # Theoretical slopes for lattices
    theoretical_slopes = {
        '1D': 1,
        '2D': 0.5,
        '3D': 1/3
    }
    
    # Plot theoretical lines
    for network_type, slope in theoretical_slopes.items():
        middle_idx = len(log_N) // 2
        intercept = np.log10(results[network_type][middle_idx]) - slope * log_N[middle_idx]
        theoretical_line = slope * x_log_fit + intercept
        
        ax5.plot(x_log_fit, theoretical_line, '--', 
                color=colors[network_type], 
                linewidth=1.5, 
                alpha=0.6,
                label=rf'$\alpha = {slope}$' + (r'$(N^{' + f'{slope:.2f}' + r'})$' if network_type != 'Random' else ''))
    
    # Add reference line for Random network
    middle_idx = len(log_N) // 2
    random_intercept = np.log10(results['Random'][middle_idx]) - 0.2 * log_N[middle_idx]
    random_theoretical = 0.2 * x_log_fit + random_intercept
    ax5.plot(x_log_fit, random_theoretical, '--', 
            color=colors['Random'], 
            linewidth=1.5, 
            alpha=0.6,
            label=r'Random ($\alpha \approx 0.2$)')
    
    ax5.set_xlabel(r'$\log_{10} N$', fontsize=16)
    ax5.set_ylabel(r'$\log_{10} \langle d \rangle$', fontsize=16)
    ax5.set_title(r'Log-Log Plot: $\log\langle d\rangle$ vs $\log N$ for All Networks', 
                 fontsize=18, fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.legend(fontsize=12, loc='upper left', ncol=2)
    ax5.tick_params(labelsize=14)
    
    save_plot(fig5, 'loglog_all_networks.png')
    plt.close(fig5)
    
    # 6. Combined Log-Log Plot (Lattices only)
    fig6, ax6 = plt.subplots(figsize=(16, 12))
    
    # Plot all lattices on log-log scale
    for network_type in ['1D', '2D', '3D']:
        ax6.loglog(N_array, results[network_type], 
                  marker=markers[network_type], 
                  color=colors[network_type], 
                  linestyle='-', 
                  linewidth=2, 
                  markersize=8,
                  label=f'{network_type} Lattice',
                  alpha=0.8)
    
    ax6.set_xlabel('$N$', fontsize=16)
    ax6.set_ylabel(r'$\langle d \rangle$', fontsize=16)
    ax6.set_title('Lattices: Log-Log Scale Comparison', fontsize=18, fontweight='bold')
    ax6.grid(True, alpha=0.3, linestyle='--', which='both')
    ax6.legend(fontsize=12, loc='upper left')
    ax6.tick_params(labelsize=14)
    
    save_plot(fig6, 'loglog_lattices_only.png')
    plt.close(fig6)
    
    # 7. Combined Semi-Log Plot (All networks)
    fig7, ax7 = plt.subplots(figsize=(16, 12))
    
    for network_type in ['1D', '2D', '3D', 'Random']:
        ax7.semilogx(N_array, results[network_type], 
                    marker=markers[network_type], 
                    color=colors[network_type], 
                    linestyle='-', 
                    linewidth=2, 
                    markersize=8,
                    label=f'{network_type} {"Lattice" if network_type != "Random" else "Network"}',
                    alpha=0.8)
    
    ax7.set_xlabel('$N$', fontsize=16)
    ax7.set_ylabel(r'$\langle d \rangle$', fontsize=16)
    ax7.set_title('All Networks: Semi-Log Scale Comparison', fontsize=18, fontweight='bold')
    ax7.grid(True, alpha=0.3, linestyle='--', which='both')
    ax7.legend(fontsize=12, loc='upper left')
    ax7.tick_params(labelsize=14)
    
    save_plot(fig7, 'semilog_all_networks.png')
    plt.close(fig7)
    
    # 8. Linear regression fits
    fig8, ax8 = plt.subplots(figsize=(16, 12))
    
    # Perform and plot linear regressions
    for network_type in ['1D', '2D', '3D']:
        log_d = np.log10(results[network_type])
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_N, log_d)
        
        # Plot data
        ax8.plot(log_N, log_d, 
               marker=markers[network_type], 
               color=colors[network_type], 
               markersize=10,
               label=f'{network_type}: α={slope:.3f}±{std_err:.3f} (R²={r_value**2:.3f})',
               alpha=0.7,
               linestyle='')
        
        # Plot regression line
        regression_line = slope * log_N + intercept
        ax8.plot(log_N, regression_line, 
               color=colors[network_type], 
               linewidth=2,
               alpha=0.5)
    
    ax8.set_xlabel(r'$\log_{10} N$', fontsize=16)
    ax8.set_ylabel(r'$\log_{10} \langle d \rangle$', fontsize=16)
    ax8.set_title('Linear Regression Analysis on Log-Log Data', fontsize=18, fontweight='bold')
    ax8.grid(True, alpha=0.3, linestyle='--')
    ax8.legend(fontsize=12, loc='upper left')
    ax8.tick_params(labelsize=14)
    
    save_plot(fig8, 'linear_regression_analysis.png')
    plt.close(fig8)
    
    # 9. Scaling exponents comparison
    fig9, ax9 = plt.subplots(figsize=(16, 12))
    
    dimensions = [1, 2, 3]
    theoretical_exponents = [1, 0.5, 1/3]
    
    # Calculate actual exponents from regression
    simulated_exponents = []
    errors = []
    for network_type in ['1D', '2D', '3D']:
        log_d = np.log10(results[network_type])
        slope, _, _, _, std_err = stats.linregress(log_N, log_d)
        simulated_exponents.append(slope)
        errors.append(std_err)
    
    x_pos = np.arange(len(dimensions))
    width = 0.35
    
    bars1 = ax9.bar(x_pos - width/2, theoretical_exponents, width, 
                   label='Theoretical (α = 1/D)', color='#2E86AB', alpha=0.8)
    bars2 = ax9.bar(x_pos + width/2, simulated_exponents, width, 
                   label='Simulated', color='#A23B72', alpha=0.8, yerr=errors, capsize=10)
    
    ax9.set_xlabel('Lattice Dimension', fontsize=16)
    ax9.set_ylabel('Scaling Exponent α', fontsize=16)
    ax9.set_title('Comparison of Theoretical vs Simulated Scaling Exponents', 
                 fontsize=18, fontweight='bold')
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels([f'{d}D' for d in dimensions], fontsize=14)
    ax9.set_ylim(0, 1.2)
    ax9.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax9.legend(fontsize=12, loc='upper right')
    ax9.tick_params(labelsize=14)
    
    # Add value labels
    for i, (th, sim) in enumerate(zip(theoretical_exponents, simulated_exponents)):
        ax9.text(i - width/2, th + 0.02, f'{th:.3f}', ha='center', va='bottom', fontsize=12)
        ax9.text(i + width/2, sim + 0.02, f'{sim:.3f}', ha='center', va='bottom', fontsize=12)
    
    save_plot(fig9, 'scaling_exponents_comparison.png')
    plt.close(fig9)
    
    # 10. Overview dashboard
    fig10 = plt.figure(figsize=(20, 12))
    
    # Create subplots
    gs = fig10.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Subplot 1: 1D lattice
    ax1 = fig10.add_subplot(gs[0, 0])
    ax1.plot(N_array, results['1D'], 'o-', color=colors['1D'], linewidth=2, markersize=5)
    ax1.set_xlabel('$N$', fontsize=10)
    ax1.set_ylabel(r'$\langle d \rangle$', fontsize=10)
    ax1.set_title('1D Lattice', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # Subplot 2: 2D lattice
    ax2 = fig10.add_subplot(gs[0, 1])
    ax2.plot(N_array, results['2D'], 's-', color=colors['2D'], linewidth=2, markersize=5)
    ax2.set_xlabel('$N$', fontsize=10)
    ax2.set_ylabel(r'$\langle d \rangle$', fontsize=10)
    ax2.set_title('2D Lattice', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2, linestyle='--')
    
    # Subplot 3: 3D lattice
    ax3 = fig10.add_subplot(gs[0, 2])
    ax3.plot(N_array, results['3D'], '^-', color=colors['3D'], linewidth=2, markersize=5)
    ax3.set_xlabel('$N$', fontsize=10)
    ax3.set_ylabel(r'$\langle d \rangle$', fontsize=10)
    ax3.set_title('3D Lattice', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.2, linestyle='--')
    
    # Subplot 4: Random network
    ax4 = fig10.add_subplot(gs[1, 0])
    ax4.plot(N_array, results['Random'], 'D-', color=colors['Random'], linewidth=2, markersize=5)
    ax4.set_xlabel('$N$', fontsize=10)
    ax4.set_ylabel(r'$\langle d \rangle$', fontsize=10)
    ax4.set_title('Random Network', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.2, linestyle='--')
    
    # Subplot 5: Log-log all networks
    ax5 = fig10.add_subplot(gs[1, 1])
    for network_type in ['1D', '2D', '3D', 'Random']:
        log_d = np.log10(results[network_type])
        ax5.plot(log_N, log_d, marker=markers[network_type], 
                color=colors[network_type], markersize=5, linestyle='', alpha=0.7)
    ax5.set_xlabel(r'$\log_{10} N$', fontsize=10)
    ax5.set_ylabel(r'$\log_{10} \langle d \rangle$', fontsize=10)
    ax5.set_title('Log-Log All Networks', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.2, linestyle='--')
    ax5.legend(['1D', '2D', '3D', 'Random'], fontsize=8)
    
    # Subplot 6: Summary statistics
    ax6 = fig10.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Calculate and display summary statistics
    summary_text = "SCALING ANALYSIS SUMMARY\n\n"
    summary_text += f"Network Sizes: {N_values}\n\n"
    
    for network_type in ['1D', '2D', '3D']:
        log_d = np.log10(results[network_type])
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_N, log_d)
        theoretical = 1 if network_type == '1D' else 0.5 if network_type == '2D' else 1/3
        error_pct = abs(slope - theoretical) / theoretical * 100
        
        summary_text += f"{network_type} Lattice:\n"
        summary_text += f"  α = {slope:.3f} ± {std_err:.3f}\n"
        summary_text += f"  Theoretical: {theoretical:.3f}\n"
        summary_text += f"  Error: {error_pct:.1f}%\n"
        summary_text += f"  R² = {r_value**2:.3f}\n\n"
    
    # Random network
    log_d_rand = np.log10(results['Random'])
    slope_rand, _, r_value_rand, _, _ = stats.linregress(log_N, results['Random'])
    summary_text += f"Random Network:\n"
    summary_text += f"  Slope = {slope_rand:.3f}\n"
    summary_text += f"  R² = {r_value_rand**2:.3f}\n"
    
    ax6.text(0.1, 0.95, summary_text, fontsize=9, fontfamily='monospace',
             verticalalignment='top', transform=ax6.transAxes)
    
    fig10.suptitle('Small-World Phenomena: Network Scaling Dashboard', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    save_plot(fig10, 'network_scaling_dashboard.png')
    plt.close(fig10)

def create_simple_plots(N_values: List[int], results: Dict[str, List[float]]):
    """Create and save simple plots as separate PNG files"""
    N_array = np.array(N_values)
    log_N = np.log10(N_array)
    
    colors = {'1D': 'red', '2D': 'blue', '3D': 'green', 'Random': 'purple'}
    markers = {'1D': 'o', '2D': 's', '3D': '^', 'Random': 'D'}
    
    # Plot 1: Linear scale
    fig1, ax1 = plt.subplots(figsize=(16, 12))
    for network in results:
        ax1.plot(N_array, results[network], marker=markers[network], 
                color=colors[network], label=network, linewidth=2, markersize=8)
    ax1.set_xlabel('$N$', fontsize=16)
    ax1.set_ylabel('$\\langle d \\rangle$', fontsize=16)
    ax1.set_title('Average Path Length vs Network Size', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=14)
    ax1.tick_params(labelsize=14)
    
    save_plot(fig1, 'simple_linear_all_networks.png')
    plt.close(fig1)
    
    # Plot 2: Log-log scale
    fig2, ax2 = plt.subplots(figsize=(16, 12))
    for network in results:
        log_d = np.log10(results[network])
        ax2.plot(log_N, log_d, marker=markers[network], 
                color=colors[network], label=network, linewidth=2, markersize=8)
    ax2.set_xlabel(r'$\log_{10} N$', fontsize=16)
    ax2.set_ylabel(r'$\log_{10} \langle d \rangle$', fontsize=16)
    ax2.set_title('Log-Log Plot: All Networks', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=14)
    ax2.tick_params(labelsize=14)
    
    save_plot(fig2, 'simple_loglog_all_networks.png')
    plt.close(fig2)

def main():
    """Main function to run simulations and generate plots"""
    print("="*70)
    print("SMALL-WORLD PHENOMENA: LOG-LOG SCALING ANALYSIS")
    print("Investigating ⟨d⟩ ∼ N^α scaling in different network topologies")
    print("="*70)
    print(f"\nSaving plots to directory: {os.path.abspath('Q1')}")
    
    # Define network sizes
    N_values = [500, 750, 1000, 1500, 2000, 3000, 5000]
    print(f"\nNetwork sizes: {N_values}")
    print("Generating networks and computing average path lengths...\n")
    
    # Run simulations
    start_time = time.time()
    results = run_simulations(N_values)
    elapsed_time = time.time() - start_time
    
    print(f"\nSimulation completed in {elapsed_time:.1f} seconds")
    
    # Create and save individual plots
    print("\n" + "="*70)
    print("GENERATING AND SAVING VISUALIZATIONS")
    print("="*70)
    
    create_individual_plots(N_values, results)
    
    # Print analysis summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    N_array = np.array(N_values)
    log_N = np.log10(N_array)
    
    print(f"\n{'Network':<12} {'Slope (α)':<12} {'Intercept':<12} {'R²':<10}")
    print("-"*50)
    
    for network_type in ['1D', '2D', '3D']:
        log_d = np.log10(results[network_type])
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_N, log_d)
        print(f"{network_type:<12} {slope:.4f}±{std_err:.4f}  {intercept:.4f}      {r_value**2:.6f}")
    
    print(f"\nAll plots saved successfully to 'Q1' directory!")
    print(f"Total plots created: 10")
    print("="*70)

if __name__ == "__main__":
    # Run the main function
    try:
        main()
    except Exception as e:
        print(f"\nError in main function: {e}")
        print("Trying simplified version...")
        
        # Try with smaller network sizes for testing
        N_values_simple = [500, 1000, 2000, 3000]
        print(f"\nRunning with simplified network sizes: {N_values_simple}")
        
        try:
            results_simple = run_simulations(N_values_simple)
            create_simple_plots(N_values_simple, results_simple)
            print(f"Plots saved successfully!")
        except Exception as e2:
            print(f"Error in simplified version: {e2}")