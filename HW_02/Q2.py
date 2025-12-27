import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy import sparse
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class NetworkRankingAnalyzer:
    def __init__(self, file_path='./HW_02/data/q2/Wiki-Vote.txt'):
        """
        Initialize the analyzer with the Wiki-Vote dataset
        """
        self.file_path = file_path
        self.graph = None
        self.nodes = None
        self.authority_scores = None
        self.hub_scores = None
        self.pagerank_scores = None
        self.authority_ranks = None
        self.pagerank_ranks = None
        
    def load_data(self):
        """
        Load the Wiki-Vote dataset and create a directed graph
        """
        print("Loading Wiki-Vote dataset...")
        
        # Read the data (skip comment lines)
        edges = []
        with open(self.file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                u, v = map(int, line.strip().split())
                edges.append((u, v))
        
        # Create directed graph
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edges)
        
        print(f"Graph loaded with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        print(f"Average in-degree: {np.mean([d for n, d in self.graph.in_degree()]):.2f}")
        print(f"Average out-degree: {np.mean([d for n, d in self.graph.out_degree()]):.2f}")
        
        self.nodes = list(self.graph.nodes())
        return self.graph
    
    def hits_algorithm(self, max_iter=100, tol=1e-6):
        """
        Implement HITS algorithm to compute authority and hub scores
        """
        print("\nRunning HITS algorithm...")
        
        # Initialize scores
        n = len(self.nodes)
        auth = np.ones(n) / n
        hub = np.ones(n) / n
        
        # Create adjacency matrix
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        A = np.zeros((n, n))
        
        for u, v in self.graph.edges():
            i = node_to_idx[u]
            j = node_to_idx[v]
            A[i, j] = 1
        
        A = csr_matrix(A)
        
        # Power iteration
        for iteration in range(max_iter):
            # Update authority scores: a = A^T * h
            auth_new = A.T.dot(hub)
            
            # Update hub scores: h = A * a
            hub_new = A.dot(auth_new)
            
            # Normalize
            auth_new = auth_new / np.linalg.norm(auth_new, 2)
            hub_new = hub_new / np.linalg.norm(hub_new, 2)
            
            # Check convergence
            auth_diff = np.linalg.norm(auth_new - auth, 2)
            hub_diff = np.linalg.norm(hub_new - hub, 2)
            
            auth, hub = auth_new, hub_new
            
            if max(auth_diff, hub_diff) < tol:
                print(f"HITS converged after {iteration + 1} iterations")
                break
        
        # Store scores
        self.authority_scores = {node: auth[node_to_idx[node]] for node in self.nodes}
        self.hub_scores = {node: hub[node_to_idx[node]] for node in self.nodes}
        
        return self.authority_scores, self.hub_scores
    
    def pagerank_algorithm(self, alpha=0.85, max_iter=100, tol=1e-6):
        """
        Implement PageRank algorithm
        """
        print(f"\nRunning PageRank algorithm with alpha={alpha}...")
        
        n = len(self.nodes)
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        # Create adjacency matrix and handle dangling nodes
        A = np.zeros((n, n))
        out_degrees = np.zeros(n)
        
        for u, v in self.graph.edges():
            i = node_to_idx[u]
            j = node_to_idx[v]
            A[i, j] = 1
            out_degrees[i] += 1
        
        # Normalize by out-degree (avoid division by zero)
        for i in range(n):
            if out_degrees[i] > 0:
                A[i, :] = A[i, :] / out_degrees[i]
        
        # Handle dangling nodes (columns sum to 0)
        dangling_nodes = np.where(out_degrees == 0)[0]
        
        # Add teleportation
        E = np.ones((n, n)) / n
        
        # PageRank matrix
        M = alpha * A + (1 - alpha) * E
        
        # Adjust for dangling nodes
        for i in dangling_nodes:
            M[i, :] = 1.0 / n
        
        # Power iteration
        pr = np.ones(n) / n
        
        for iteration in range(max_iter):
            pr_new = M.T.dot(pr)
            
            # Normalize
            pr_new = pr_new / np.linalg.norm(pr_new, 1)
            
            # Check convergence
            diff = np.linalg.norm(pr_new - pr, 1)
            pr = pr_new
            
            if diff < tol:
                print(f"PageRank converged after {iteration + 1} iterations")
                break
        
        self.pagerank_scores = {node: pr[node_to_idx[node]] for node in self.nodes}
        return self.pagerank_scores
    
    def compute_ranks(self):
        """
        Convert scores to ranks (Rank 1 = highest score)
        """
        print("\nConverting scores to ranks...")
        
        # Authority ranks
        sorted_auth = sorted(self.authority_scores.items(), key=lambda x: x[1], reverse=True)
        self.authority_ranks = {}
        for rank, (node, _) in enumerate(sorted_auth, 1):
            self.authority_ranks[node] = rank
        
        # PageRank ranks
        sorted_pr = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        self.pagerank_ranks = {}
        for rank, (node, _) in enumerate(sorted_pr, 1):
            self.pagerank_ranks[node] = rank
        
        return self.authority_ranks, self.pagerank_ranks
    
    def compare_ranks(self):
        """
        Create scatter plot comparing Authority Rank vs PageRank Rank
        """
        print("\nCreating rank comparison plot...")
        
        # Prepare data for scatter plot
        nodes_list = []
        auth_ranks_list = []
        pr_ranks_list = []
        
        for node in self.nodes:
            auth_rank = self.authority_ranks[node]
            pr_rank = self.pagerank_ranks[node]
            nodes_list.append(node)
            auth_ranks_list.append(auth_rank)
            pr_ranks_list.append(pr_rank)
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame({
            'node': nodes_list,
            'authority_rank': auth_ranks_list,
            'pagerank_rank': pr_ranks_list
        })
        
        # Create scatter plot
        plt.figure(figsize=(12, 10))
        
        # Log-log scale
        plt.loglog(df['authority_rank'], df['pagerank_rank'], 'o', 
                  alpha=0.6, markersize=4, label='Nodes')
        
        # Add y=x line for reference
        min_rank = min(df['authority_rank'].min(), df['pagerank_rank'].min())
        max_rank = max(df['authority_rank'].max(), df['pagerank_rank'].max())
        plt.loglog([min_rank, max_rank], [min_rank, max_rank], 
                  'r--', alpha=0.7, label='y = x (perfect agreement)')
        
        plt.xlabel('Authority Rank (HITS)', fontsize=14)
        plt.ylabel('PageRank Rank', fontsize=14)
        plt.title('Comparative Analysis: HITS Authority vs PageRank (Log-Log Scale)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Identify significant outliers
        df['rank_diff'] = abs(df['pagerank_rank'] - df['authority_rank'])
        df['rank_ratio'] = df['pagerank_rank'] / df['authority_rank']
        
        # Find nodes with largest differences
        significant_outliers = df.nlargest(10, 'rank_diff')
        
        # Highlight outliers on plot
        plt.scatter(significant_outliers['authority_rank'], 
                   significant_outliers['pagerank_rank'],
                   color='red', s=100, alpha=0.8, 
                   label='Top 10 divergent nodes', zorder=5)
        
        # Annotate some outliers
        for idx, row in significant_outliers.head(5).iterrows():
            plt.annotate(f'Node {row["node"]}', 
                        xy=(row['authority_rank'], row['pagerank_rank']),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('./HW_02/Q2/rank_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df, significant_outliers
    
    def analyze_divergent_nodes(self, df, top_n=5):
        """
        Analyze structural reasons for divergent rankings
        """
        print("\n" + "="*60)
        print("ANALYSIS OF DIVERGENT NODES")
        print("="*60)
        
        # Get top divergent nodes
        divergent_nodes = df.nlargest(top_n, 'rank_diff')
        
        for idx, row in divergent_nodes.iterrows():
            node = row['node']
            auth_rank = row['authority_rank']
            pr_rank = row['pagerank_rank']
            
            print(f"\nNode {node}:")
            print(f"  - Authority Rank: {auth_rank}")
            print(f"  - PageRank Rank: {pr_rank}")
            print(f"  - Rank Difference: {abs(pr_rank - auth_rank)}")
            
            # Get network properties
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            auth_score = self.authority_scores[node]
            pr_score = self.pagerank_scores[node]
            
            print(f"  - In-degree: {in_degree}")
            print(f"  - Out-degree: {out_degree}")
            print(f"  - Authority Score: {auth_score:.6f}")
            print(f"  - PageRank Score: {pr_score:.6f}")
            
            # Get predecessors (nodes that vote for this node)
            predecessors = list(self.graph.predecessors(node))
            print(f"  - Number of endorsers: {len(predecessors)}")
            
            if predecessors:
                # Analyze endorsers
                endorser_hub_scores = [self.hub_scores[p] for p in predecessors]
                avg_hub_score = np.mean(endorser_hub_scores)
                print(f"  - Average hub score of endorsers: {avg_hub_score:.6f}")
                
                # Check if endorsers are also highly ranked
                high_hub_endorsers = [p for p in predecessors if self.hub_scores[p] > 0.01]
                print(f"  - Number of high-hub endorsers: {len(high_hub_endorsers)}")
            
            # Structural interpretation
            if pr_rank < auth_rank:  # PageRank ranks it higher
                print(f"  - INTERPRETATION: PageRank ranks this node higher than HITS.")
                print(f"    This suggests the node receives votes from nodes that themselves")
                print(f"    receive many votes (high PageRank), creating a virtuous cycle.")
            else:  # HITS ranks it higher
                print(f"  - INTERPRETATION: HITS ranks this node higher than PageRank.")
                print(f"    This suggests the node is endorsed by active hubs,")
                print(f"    even if those hubs don't have high PageRank themselves.")
            
            print("-" * 40)
    
    def pagerank_sensitivity_analysis(self, alpha_values=None, top_n=10):
        """
        Analyze how PageRank changes with different alpha values
        """
        print("\n" + "="*60)
        print("PAGERANK SENSITIVITY ANALYSIS")
        print("="*60)
        
        if alpha_values is None:
            alpha_values = np.linspace(0.50, 0.99, 20)
        
        # Get top nodes from initial analysis
        top_nodes = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_node_ids = [node for node, _ in top_nodes]
        
        # Get some divergent nodes from previous analysis
        divergent_nodes = []
        if hasattr(self, 'significant_outliers'):
            divergent_nodes = self.significant_outliers['node'].head(5).tolist()
        
        # Combine nodes to track
        nodes_to_track = list(set(top_node_ids + divergent_nodes))
        
        # Store results
        rank_trajectories = {node: [] for node in nodes_to_track}
        
        print(f"Tracking rank trajectories for {len(nodes_to_track)} nodes across {len(alpha_values)} alpha values...")
        
        # Compute PageRank for each alpha
        for alpha in alpha_values:
            pr_scores = self.pagerank_algorithm_custom_alpha(alpha)
            
            # Convert to ranks
            sorted_nodes = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
            ranks = {node: rank+1 for rank, (node, _) in enumerate(sorted_nodes)}
            
            # Store ranks for tracked nodes
            for node in nodes_to_track:
                if node in ranks:
                    rank_trajectories[node].append(ranks[node])
                else:
                    rank_trajectories[node].append(len(self.nodes) + 1)  # Bottom rank
        
        # Plot trajectories
        self.plot_rank_trajectories(rank_trajectories, alpha_values, nodes_to_track)
        
        return rank_trajectories, alpha_values
    
    def pagerank_algorithm_custom_alpha(self, alpha=0.85, max_iter=100, tol=1e-6):
        """
        PageRank implementation that accepts custom alpha
        """
        n = len(self.nodes)
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        # Create adjacency matrix
        A = np.zeros((n, n))
        out_degrees = np.zeros(n)
        
        for u, v in self.graph.edges():
            i = node_to_idx[u]
            j = node_to_idx[v]
            A[i, j] = 1
            out_degrees[i] += 1
        
        # Normalize by out-degree
        for i in range(n):
            if out_degrees[i] > 0:
                A[i, :] = A[i, :] / out_degrees[i]
        
        # Handle dangling nodes
        dangling_nodes = np.where(out_degrees == 0)[0]
        
        # Add teleportation
        E = np.ones((n, n)) / n
        
        # PageRank matrix
        M = alpha * A + (1 - alpha) * E
        
        # Adjust for dangling nodes
        for i in dangling_nodes:
            M[i, :] = 1.0 / n
        
        # Power iteration
        pr = np.ones(n) / n
        
        for iteration in range(max_iter):
            pr_new = M.T.dot(pr)
            pr_new = pr_new / np.linalg.norm(pr_new, 1)
            
            diff = np.linalg.norm(pr_new - pr, 1)
            pr = pr_new
            
            if diff < tol:
                break
        
        return {node: pr[node_to_idx[node]] for node in self.nodes}
    
    def plot_rank_trajectories(self, rank_trajectories, alpha_values, nodes_to_track):
        """
        Plot rank trajectories for different alpha values
        """
        print("\nCreating rank trajectory plot...")
        
        plt.figure(figsize=(14, 10))
        
        # Color map for better visualization
        colors = plt.cm.tab20(np.linspace(0, 1, len(nodes_to_track)))
        
        for idx, (node, trajectory) in enumerate(rank_trajectories.items()):
            plt.plot(alpha_values, trajectory, '-o', 
                    label=f'Node {node}', 
                    linewidth=2, markersize=6,
                    color=colors[idx])
        
        plt.xlabel('Damping Factor (α)', fontsize=14)
        plt.ylabel('Rank (lower is better)', fontsize=14)
        plt.title('PageRank Sensitivity Analysis: Rank Trajectories vs Damping Factor', fontsize=16)
        
        # Reverse y-axis so rank 1 (best) is at top
        plt.gca().invert_yaxis()
        
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Add horizontal lines for reference
        for rank in [1, 10, 100, 1000]:
            if rank < len(self.nodes):
                plt.axhline(y=rank, color='gray', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./HW_02/Q2/rank_trajectories.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze and interpret trajectories
        self.interpret_trajectories(rank_trajectories, alpha_values, nodes_to_track)
    
    def interpret_trajectories(self, rank_trajectories, alpha_values, nodes_to_track):
        """
        Interpret the rank trajectories
        """
        print("\n" + "="*60)
        print("TRAJECTORY INTERPRETATION")
        print("="*60)
        
        for node in nodes_to_track[:10]:  # Analyze top 10 nodes
            trajectory = rank_trajectories[node]
            
            # Calculate trend
            start_rank = trajectory[0]
            end_rank = trajectory[-1]
            rank_change = end_rank - start_rank
            
            # Classify trajectory
            if abs(rank_change) < 5:
                trend = "STABLE"
            elif rank_change < -5:
                trend = "IMPROVING (rank decreasing)"
            else:
                trend = "DECLINING (rank increasing)"
            
            print(f"\nNode {node}:")
            print(f"  - Initial rank (α=0.50): {start_rank}")
            print(f"  - Final rank (α=0.99): {end_rank}")
            print(f"  - Trend: {trend}")
            
            # Interpret based on trend
            if "IMPROVING" in trend:
                print(f"  - INTERPRETATION: This node benefits from higher α (more link-following).")
                print(f"    Its influence comes from network structure, not random jumps.")
                print(f"    Likely has strong, well-connected endorsers.")
            elif "DECLINING" in trend:
                print(f"  - INTERPRETATION: This node suffers from higher α.")
                print(f"    Its ranking depends more on random teleportation.")
                print(f"    May have isolated endorsements or be in a less connected region.")
            else:
                print(f"  - INTERPRETATION: This node's rank is robust to α changes.")
                print(f"    It maintains consistent importance regardless of random surfer patience.")
                print(f"    Likely a core, well-connected node in the network.")
            
            # Network statistics
            in_degree = self.graph.in_degree(node)
            auth_rank = self.authority_ranks.get(node, "N/A")
            
            print(f"  - In-degree: {in_degree}")
            print(f"  - Authority Rank: {auth_rank}")
            
            # Check for local vs global influence
            if in_degree > 50 and "STABLE" in trend:
                print(f"  - This node appears to have GLOBAL influence (high degree, stable rank)")
            elif in_degree < 10 and "DECLINING" in trend:
                print(f"  - This node appears to have LOCAL influence (low degree, α-sensitive)")
            
            print("-" * 40)
    
    def comprehensive_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("="*60)
        print("COMPARATIVE ANALYSIS OF RANKING ALGORITHMS")
        print("="*60)
        
        # Part A: Comparative Analysis
        print("\nPART A: RANKING COMPARISON (HITS vs. PageRank)")
        print("-" * 40)
        
        # Load data
        self.load_data()
        
        # Run HITS algorithm
        self.hits_algorithm()
        
        # Run PageRank with default alpha
        self.pagerank_algorithm(alpha=0.85)
        
        # Compute ranks
        self.compute_ranks()
        
        # Compare ranks
        df, outliers = self.compare_ranks()
        self.significant_outliers = outliers
        
        # Analyze divergent nodes
        self.analyze_divergent_nodes(df, top_n=5)
        
        # Part B: Sensitivity Analysis
        print("\n\nPART B: RANK STABILITY ANALYSIS")
        print("-" * 40)
        
        # Perform sensitivity analysis
        rank_trajectories, alpha_values = self.pagerank_sensitivity_analysis()
        
        # Summary statistics
        self.print_summary_statistics(df)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nOutput files saved:")
        print("1. rank_comparison.png - Scatter plot of Authority vs PageRank ranks")
        print("2. rank_trajectories.png - Line chart of rank trajectories vs alpha")
    
    def print_summary_statistics(self, df):
        """
        Print summary statistics of the analysis
        """
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        # Calculate correlation between ranks
        from scipy.stats import spearmanr, kendalltau
        
        auth_ranks = [self.authority_ranks[node] for node in self.nodes]
        pr_ranks = [self.pagerank_ranks[node] for node in self.nodes]
        
        spearman_corr, _ = spearmanr(auth_ranks, pr_ranks)
        kendall_corr, _ = kendalltau(auth_ranks, pr_ranks)
        
        print(f"\nRank Correlation Metrics:")
        print(f"  - Spearman Rank Correlation: {spearman_corr:.4f}")
        print(f"  - Kendall's Tau: {kendall_corr:.4f}")
        
        # Top nodes comparison
        top_auth = sorted(self.authority_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        top_pr = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        auth_top_nodes = [node for node, _ in top_auth]
        pr_top_nodes = [node for node, _ in top_pr]
        
        overlap = len(set(auth_top_nodes) & set(pr_top_nodes))
        
        print(f"\nTop 10 Nodes Overlap: {overlap}/10 ({overlap/10*100:.1f}%)")
        
        print(f"\nTop 5 Authority Nodes (HITS):")
        for node, score in top_auth[:5]:
            print(f"  Node {node}: Authority Score = {score:.6f}, Rank = {self.authority_ranks[node]}")
        
        print(f"\nTop 5 PageRank Nodes:")
        for node, score in top_pr[:5]:
            print(f"  Node {node}: PageRank Score = {score:.6f}, Rank = {self.pagerank_ranks[node]}")
        
        # Divergence statistics
        df['abs_rank_diff'] = abs(df['pagerank_rank'] - df['authority_rank'])
        
        print(f"\nRank Divergence Statistics:")
        print(f"  - Average absolute rank difference: {df['abs_rank_diff'].mean():.2f}")
        print(f"  - Median absolute rank difference: {df['abs_rank_diff'].median():.2f}")
        print(f"  - Maximum absolute rank difference: {df['abs_rank_diff'].max():.2f}")
        print(f"  - Nodes with rank difference > 100: {(df['abs_rank_diff'] > 100).sum()}")
        
        # Degree statistics for top nodes
        print(f"\nNetwork Properties of Top-Ranked Nodes:")
        for node_list, algorithm in [(auth_top_nodes[:5], "HITS"), (pr_top_nodes[:5], "PageRank")]:
            print(f"\n  {algorithm} Top 5 Nodes:")
            for node in node_list:
                in_deg = self.graph.in_degree(node)
                out_deg = self.graph.out_degree(node)
                print(f"    Node {node}: In-degree={in_deg}, Out-degree={out_deg}")


# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = NetworkRankingAnalyzer('./HW_02/data/q2/Wiki-Vote.txt')
    
    # Run comprehensive analysis
    analyzer.comprehensive_analysis()
    
    # Additional analysis: Compare top nodes more deeply
    print("\n" + "="*60)
    print("DETAILED TOP NODES COMPARISON")
    print("="*60)
    
    # Get top 20 from each algorithm
    top_20_auth = sorted(analyzer.authority_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    top_20_pr = sorted(analyzer.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    
    auth_set = set([node for node, _ in top_20_auth])
    pr_set = set([node for node, _ in top_20_pr])
    
    print(f"\nNodes in HITS top 20 but NOT in PageRank top 20:")
    for node in auth_set - pr_set:
        auth_rank = analyzer.authority_ranks[node]
        pr_rank = analyzer.pagerank_ranks[node]
        print(f"  Node {node}: HITS Rank={auth_rank}, PageRank Rank={pr_rank}")
    
    print(f"\nNodes in PageRank top 20 but NOT in HITS top 20:")
    for node in pr_set - auth_set:
        auth_rank = analyzer.authority_ranks[node]
        pr_rank = analyzer.pagerank_ranks[node]
        print(f"  Node {node}: HITS Rank={auth_rank}, PageRank Rank={pr_rank}")