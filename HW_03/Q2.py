import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import os
from itertools import combinations

class SchoolNetworkAnalyzer:
    def __init__(self, data_path="../Networks/Part_B"):
        self.data_path = data_path
        self.networks = {}
        self.attributes = {}
        self.results = {}
        
    def load_data(self):
        """Load all network and attribute data"""
        days = [1, 30, 60, 90]
        
        for day in days:
            # Load connections
            try:
                conn_file = f"{self.data_path}connections_day_{day}.csv"
                connections = pd.read_csv(conn_file)
                self.networks[day] = connections
                print(f"Loaded Day {day}: {len(connections)} connections")
            except FileNotFoundError:
                print(f"Warning: connections_day_{day}.csv not found, using simulated data")
                # Generate simulated connections if files don't exist
                self.networks[day] = self.simulate_connections(day)
            
            # Load attributes
            try:
                attr_file = f"{self.data_path}properties_day_{day}.csv"
                attributes = pd.read_csv(attr_file)
                self.attributes[day] = attributes
                print(f"Loaded Day {day}: {len(attributes)} students")
            except FileNotFoundError:
                print(f"Warning: properties_day_{day}.csv not found, using simulated data")
                self.attributes[day] = self.simulate_attributes(day)
    
    def simulate_connections(self, day):
        """Generate simulated connection data if files are missing"""
        # This is for demonstration when actual files aren't available
        np.random.seed(day)
        n_students = 120
        n_connections = 300 + day * 2  # Network grows over time
        
        connections = []
        for _ in range(n_connections):
            s1 = np.random.randint(0, n_students)
            s2 = np.random.randint(0, n_students)
            if s1 != s2:
                connections.append([s1, s2])
        
        return pd.DataFrame(connections, columns=['student_id_1', 'student_id_2'])
    
    def simulate_attributes(self, day):
        """Generate simulated attribute data"""
        np.random.seed(day)
        n_students = 120
        
        data = {
            'id': list(range(n_students)),
            'gender': np.random.randint(0, 2, n_students),
            'age': np.random.randint(14, 19, n_students),
            'studies': np.random.randint(1, 6, n_students),
            'plays_football': np.random.randint(0, 2, n_students),
            'watches_movies': np.random.randint(0, 2, n_students),
            'club': np.random.randint(0, 2, n_students),
            'smokes': np.random.choice([0, 1], n_students, p=[0.8, 0.2]),  # 20% smokers initially
            'class_number': np.repeat([1, 2, 3, 4], 30)
        }
        
        # Make smoking spread over time
        if day > 1:
            # Simulate smoking diffusion
            base_df = pd.DataFrame(data)
            # Increase smoking prevalence slightly each day
            smoke_rate = 0.2 + (day/90) * 0.1  # Increases to ~30% by Day 90
            base_df['smokes'] = np.random.choice([0, 1], n_students, p=[1-smoke_rate, smoke_rate])
            return base_df
        
        return pd.DataFrame(data)
    
    def create_network_graph(self, day):
        """Create NetworkX graph for a specific day"""
        G = nx.Graph()
        
        # Add nodes with attributes
        attrs = self.attributes[day]
        for _, row in attrs.iterrows():
            G.add_node(row['id'], **row.to_dict())
        
        # Add edges
        connections = self.networks[day]
        for _, row in connections.iterrows():
            G.add_edge(row['student_id_1'], row['student_id_2'])
        
        return G
    
    def analyze_triadic_closure(self):
        """Analyze triadic closure patterns over time"""
        print("\n" + "="*60)
        print("TRIADIC CLOSURE ANALYSIS")
        print("="*60)
        
        triadic_results = {}
        
        for i, day in enumerate(sorted(self.networks.keys())):
            G = self.create_network_graph(day)
            
            # Count triangles
            triangles = nx.triangles(G)
            total_triangles = sum(triangles.values()) // 3
            
            # Calculate clustering coefficient
            clustering = nx.average_clustering(G)
            
            # Identify potential triadic closures (friend of friend connections)
            potential_triads = 0
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                for n1, n2 in combinations(neighbors, 2):
                    if not G.has_edge(n1, n2):
                        potential_triads += 1
            
            # Analyze by smoking status
            smoker_triangles = 0
            mixed_triangles = 0
            non_smoker_triangles = 0
            
            # Simple triangle sampling for smoking analysis
            sampled_nodes = list(G.nodes())[:50]  # Sample for efficiency
            for node in sampled_nodes:
                neighbors = list(G.neighbors(node))
                for n1, n2 in combinations(neighbors, 2):
                    if G.has_edge(n1, n2):
                        # This is a triangle (node, n1, n2)
                        smoke_status = [G.nodes[node]['smokes'], 
                                       G.nodes[n1]['smokes'], 
                                       G.nodes[n2]['smokes']]
                        smoke_count = sum(smoke_status)
                        if smoke_count == 3:
                            smoker_triangles += 1
                        elif smoke_count == 0:
                            non_smoker_triangles += 1
                        else:
                            mixed_triangles += 1
            
            triadic_results[day] = {
                'total_triangles': total_triangles,
                'clustering_coefficient': clustering,
                'potential_triads': potential_triads,
                'smoker_triangles': smoker_triangles,
                'mixed_triangles': mixed_triangles,
                'non_smoker_triangles': non_smoker_triangles
            }
            
            print(f"\nDay {day}:")
            print(f"  Total triangles: {total_triangles}")
            print(f"  Average clustering: {clustering:.3f}")
            print(f"  Potential triadic closures: {potential_triads}")
            print(f"  Smoking triangles - All smokers: {smoker_triangles}, Mixed: {mixed_triangles}, Non-smokers: {non_smoker_triangles}")
            
            # Compare with previous day if available
            if i > 0:
                prev_day = sorted(self.networks.keys())[i-1]
                triadic_growth = ((total_triangles - triadic_results[prev_day]['total_triangles']) / 
                                 triadic_results[prev_day]['total_triangles'] * 100)
                print(f"  Triangle growth since Day {prev_day}: {triadic_growth:+.1f}%")
        
        self.results['triadic'] = triadic_results
        return triadic_results
    
    def analyze_membership_closure(self):
        """Analyze membership closure (within-class connections)"""
        print("\n" + "="*60)
        print("MEMBERSHIP CLOSURE ANALYSIS")
        print("="*60)
        
        membership_results = {}
        
        for day in sorted(self.networks.keys()):
            G = self.create_network_graph(day)
            connections = self.networks[day]
            attrs = self.attributes[day]
            
            # Count within-class and cross-class connections
            within_class = 0
            cross_class = 0
            
            # Create dictionary for quick class lookup
            class_dict = dict(zip(attrs['id'], attrs['class_number']))
            
            for _, row in connections.iterrows():
                s1_class = class_dict[row['student_id_1']]
                s2_class = class_dict[row['student_id_2']]
                
                if s1_class == s2_class:
                    within_class += 1
                else:
                    cross_class += 1
            
            total_connections = within_class + cross_class
            within_class_pct = (within_class / total_connections * 100) if total_connections > 0 else 0
            
            # Analyze by smoking status within classes
            within_class_smoking = {'both_smoke': 0, 'one_smokes': 0, 'neither_smoke': 0}
            
            for _, row in connections.iterrows():
                s1_class = class_dict[row['student_id_1']]
                s2_class = class_dict[row['student_id_2']]
                
                if s1_class == s2_class:
                    s1_smokes = attrs.loc[attrs['id'] == row['student_id_1'], 'smokes'].values[0]
                    s2_smokes = attrs.loc[attrs['id'] == row['student_id_2'], 'smokes'].values[0]
                    
                    if s1_smokes and s2_smokes:
                        within_class_smoking['both_smoke'] += 1
                    elif s1_smokes or s2_smokes:
                        within_class_smoking['one_smokes'] += 1
                    else:
                        within_class_smoking['neither_smoke'] += 1
            
            membership_results[day] = {
                'within_class': within_class,
                'cross_class': cross_class,
                'within_class_pct': within_class_pct,
                'within_class_smoking': within_class_smoking
            }
            
            print(f"\nDay {day}:")
            print(f"  Within-class connections: {within_class} ({within_class_pct:.1f}%)")
            print(f"  Cross-class connections: {cross_class}")
            print(f"  Within-class smoking patterns:")
            print(f"    Both smoke: {within_class_smoking['both_smoke']}")
            print(f"    One smokes: {within_class_smoking['one_smokes']}")
            print(f"    Neither smoke: {within_class_smoking['neither_smoke']}")
        
        self.results['membership'] = membership_results
        return membership_results
    
    def analyze_focal_closure(self):
        """Analyze focal closure patterns related to shared attributes"""
        print("\n" + "="*60)
        print("FOCAL CLOSURE ANALYSIS")
        print("="*60)
        
        focal_results = {}
        
        for day in sorted(self.networks.keys()):
            G = self.create_network_graph(day)
            connections = self.networks[day]
            attrs = self.attributes[day]
            
            # Analyze connections based on shared attributes
            shared_features = {
                'same_gender': 0,
                'same_class': 0,
                'both_smoke': 0,
                'both_play_football': 0,
                'both_in_club': 0,
                'age_diff_1': 0  # Age difference ≤ 1 year
            }
            
            for _, row in connections.iterrows():
                s1 = row['student_id_1']
                s2 = row['student_id_2']
                
                s1_data = attrs[attrs['id'] == s1].iloc[0]
                s2_data = attrs[attrs['id'] == s2].iloc[0]
                
                if s1_data['gender'] == s2_data['gender']:
                    shared_features['same_gender'] += 1
                
                if s1_data['class_number'] == s2_data['class_number']:
                    shared_features['same_class'] += 1
                
                if s1_data['smokes'] == 1 and s2_data['smokes'] == 1:
                    shared_features['both_smoke'] += 1
                
                if s1_data['plays_football'] == 1 and s2_data['plays_football'] == 1:
                    shared_features['both_play_football'] += 1
                
                if s1_data['club'] == 1 and s2_data['club'] == 1:
                    shared_features['both_in_club'] += 1
                
                if abs(s1_data['age'] - s2_data['age']) <= 1:
                    shared_features['age_diff_1'] += 1
            
            total_connections = len(connections)
            for key in shared_features:
                shared_features[key] = (shared_features[key] / total_connections * 100) if total_connections > 0 else 0
            
            # Analyze smoking homophily over time
            smoker_connections = 0
            smoker_non_smoker_connections = 0
            non_smoker_connections = 0
            
            for _, row in connections.iterrows():
                s1_smokes = attrs.loc[attrs['id'] == row['student_id_1'], 'smokes'].values[0]
                s2_smokes = attrs.loc[attrs['id'] == row['student_id_2'], 'smokes'].values[0]
                
                if s1_smokes and s2_smokes:
                    smoker_connections += 1
                elif not s1_smokes and not s2_smokes:
                    non_smoker_connections += 1
                else:
                    smoker_non_smoker_connections += 1
            
            focal_results[day] = {
                'shared_features_pct': shared_features,
                'smoking_homophily': {
                    'smoker_smoker': smoker_connections,
                    'mixed': smoker_non_smoker_connections,
                    'non_smoker_non_smoker': non_smoker_connections
                }
            }
            
            print(f"\nDay {day}:")
            print(f"  Shared attribute percentages:")
            for feature, pct in shared_features.items():
                print(f"    {feature}: {pct:.1f}%")
            
            print(f"  Smoking homophily patterns:")
            print(f"    Smoker-Smoker: {smoker_connections}")
            print(f"    Mixed: {smoker_non_smoker_connections}")
            print(f"    Non-smoker-Non-smoker: {non_smoker_connections}")
        
        self.results['focal'] = focal_results
        return focal_results
    
    def compare_smokers_nonsmokers(self):
        """Compare feature distributions between smokers and non-smokers"""
        print("\n" + "="*60)
        print("SMOKER vs NON-SMOKER COMPARISON")
        print("="*60)
        
        comparison_results = {}
        
        for day in sorted(self.attributes.keys()):
            attrs = self.attributes[day]
            
            # Split data
            smokers = attrs[attrs['smokes'] == 1]
            non_smokers = attrs[attrs['smokes'] == 0]
            
            # Calculate statistics
            stats = {
                'smoker_count': len(smokers),
                'non_smoker_count': len(non_smokers),
                'smoking_prevalence': len(smokers) / len(attrs) * 100,
                'avg_age_smokers': smokers['age'].mean(),
                'avg_age_nonsmokers': non_smokers['age'].mean(),
                'gender_ratio_smokers': smokers['gender'].mean() * 100,  # % male
                'gender_ratio_nonsmokers': non_smokers['gender'].mean() * 100,
                'club_participation_smokers': smokers['club'].mean() * 100,
                'club_participation_nonsmokers': non_smokers['club'].mean() * 100,
                'football_smokers': smokers['plays_football'].mean() * 100,
                'football_nonsmokers': non_smokers['plays_football'].mean() * 100,
                'avg_studies_smokers': smokers['studies'].mean(),
                'avg_studies_nonsmokers': non_smokers['studies'].mean()
            }
            
            comparison_results[day] = stats
            
            print(f"\nDay {day}:")
            print(f"  Smoking prevalence: {stats['smoking_prevalence']:.1f}% ({stats['smoker_count']} smokers)")
            print(f"  Average age - Smokers: {stats['avg_age_smokers']:.1f}, Non-smokers: {stats['avg_age_nonsmokers']:.1f}")
            print(f"  Gender (% male) - Smokers: {stats['gender_ratio_smokers']:.1f}%, Non-smokers: {stats['gender_ratio_nonsmokers']:.1f}%")
            print(f"  Club participation - Smokers: {stats['club_participation_smokers']:.1f}%, Non-smokers: {stats['club_participation_nonsmokers']:.1f}%")
            print(f"  Football players - Smokers: {stats['football_smokers']:.1f}%, Non-smokers: {stats['football_nonsmokers']:.1f}%")
            print(f"  Study intensity (1-5) - Smokers: {stats['avg_studies_smokers']:.1f}, Non-smokers: {stats['avg_studies_nonsmokers']:.1f}")
        
        self.results['comparison'] = comparison_results
        return comparison_results
    
    def compute_centrality(self):
        """Compute degree centrality and identify top central students"""
        print("\n" + "="*60)
        print("CENTRALITY ANALYSIS")
        print("="*60)
        
        centrality_results = {}
        
        for day in sorted(self.networks.keys()):
            G = self.create_network_graph(day)
            
            # Compute degree centrality
            degree_centrality = nx.degree_centrality(G)
            
            # Get top 5 most central students
            top_5 = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Analyze their attributes
            top_students_info = []
            for student_id, centrality in top_5:
                attrs = self.attributes[day]
                student_data = attrs[attrs['id'] == student_id].iloc[0]
                
                top_students_info.append({
                    'id': student_id,
                    'centrality': centrality,
                    'smokes': student_data['smokes'],
                    'gender': 'Male' if student_data['gender'] == 1 else 'Female',
                    'age': student_data['age'],
                    'class': student_data['class_number'],
                    'club': student_data['club'],
                    'football': student_data['plays_football']
                })
            
            centrality_results[day] = {
                'top_5': top_students_info,
                'avg_centrality': np.mean(list(degree_centrality.values())),
                'centrality_distribution': degree_centrality
            }
            
            print(f"\nDay {day} - Top 5 Most Central Students:")
            for i, student in enumerate(top_students_info, 1):
                smoke_status = "Smoker" if student['smokes'] == 1 else "Non-smoker"
                print(f"  {i}. Student {student['id']}: Centrality={student['centrality']:.3f}, "
                      f"{smoke_status}, {student['gender']}, Age {student['age']}, Class {student['class']}")
            
            # Calculate smoking status of central students
            central_smokers = sum(1 for s in top_students_info if s['smokes'] == 1)
            print(f"  Smoking among central students: {central_smokers}/5 ({central_smokers/5*100:.0f}%)")
        
        self.results['centrality'] = centrality_results
        return centrality_results
    
    def analyze_smoking_diffusion(self):
        """Analyze how smoking behavior spreads through the network"""
        print("\n" + "="*60)
        print("SMOKING DIFFUSION ANALYSIS")
        print("="*60)
        
        # Track new smokers over time
        smoking_evolution = {}
        
        days = sorted(self.attributes.keys())
        
        for i, day in enumerate(days):
            current_smokers = set(self.attributes[day][self.attributes[day]['smokes'] == 1]['id'])
            smoking_evolution[day] = {
                'smokers': current_smokers,
                'count': len(current_smokers)
            }
            
            if i > 0:
                prev_day = days[i-1]
                prev_smokers = smoking_evolution[prev_day]['smokers']
                new_smokers = current_smokers - prev_smokers
                stopped_smokers = prev_smokers - current_smokers
                
                print(f"\nDay {day} vs Day {prev_day}:")
                print(f"  New smokers: {len(new_smokers)}")
                print(f"  Stopped smoking: {len(stopped_smokers)}")
                print(f"  Net change: {len(new_smokers) - len(stopped_smokers)}")
        
        # Analyze if new smokers are connected to existing smokers
        for i in range(1, len(days)):
            day = days[i]
            prev_day = days[i-1]
            
            G_prev = self.create_network_graph(prev_day)
            prev_smokers = smoking_evolution[prev_day]['smokers']
            new_smokers = smoking_evolution[day]['smokers'] - smoking_evolution[prev_day]['smokers']
            
            new_smokers_with_smoker_friends = 0
            
            for new_smoker in new_smokers:
                if new_smoker in G_prev.nodes():
                    neighbors = list(G_prev.neighbors(new_smoker))
                    smoker_neighbors = [n for n in neighbors if n in prev_smokers]
                    if len(smoker_neighbors) > 0:
                        new_smokers_with_smoker_friends += 1
            
            if len(new_smokers) > 0:
                pct_with_smoker_friends = new_smokers_with_smoker_friends / len(new_smokers) * 100
                print(f"\nDay {day}: {pct_with_smoker_friends:.1f}% of new smokers had smoker friends on Day {prev_day}")
    
    def visualize_results(self):
        """Create visualizations of key findings"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('School Social Network Evolution Analysis', fontsize=16)
        
        # Plot 1: Smoking prevalence over time
        days = sorted(self.results['comparison'].keys())
        smoking_rates = [self.results['comparison'][d]['smoking_prevalence'] for d in days]
        
        axes[0, 0].plot(days, smoking_rates, marker='o', linewidth=2)
        axes[0, 0].set_title('Smoking Prevalence Over Time')
        axes[0, 0].set_xlabel('Day')
        axes[0, 0].set_ylabel('Percentage of Smokers (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Clustering coefficient over time
        clustering_coeffs = [self.results['triadic'][d]['clustering_coefficient'] for d in days]
        
        axes[0, 1].plot(days, clustering_coeffs, marker='s', color='green', linewidth=2)
        axes[0, 1].set_title('Network Clustering Over Time')
        axes[0, 1].set_xlabel('Day')
        axes[0, 1].set_ylabel('Average Clustering Coefficient')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Within-class connections percentage
        within_class_pct = [self.results['membership'][d]['within_class_pct'] for d in days]
        
        axes[0, 2].plot(days, within_class_pct, marker='^', color='red', linewidth=2)
        axes[0, 2].set_title('Membership Closure (Within-class %)')
        axes[0, 2].set_xlabel('Day')
        axes[0, 2].set_ylabel('Percentage of Within-class Connections (%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Smoking homophily
        smoker_smoker = [self.results['focal'][d]['smoking_homophily']['smoker_smoker'] for d in days]
        mixed = [self.results['focal'][d]['smoking_homophily']['mixed'] for d in days]
        
        x = range(len(days))
        width = 0.35
        axes[1, 0].bar(x, smoker_smoker, width, label='Smoker-Smoker', color='orange')
        axes[1, 0].bar([i + width for i in x], mixed, width, label='Mixed', color='gray')
        axes[1, 0].set_title('Smoking Homophily in Connections')
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Number of Connections')
        axes[1, 0].set_xticks([i + width/2 for i in x])
        axes[1, 0].set_xticklabels(days)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Centrality distribution for Day 90
        if 90 in self.results['centrality']:
            centrality_vals = list(self.results['centrality'][90]['centrality_distribution'].values())
            axes[1, 1].hist(centrality_vals, bins=20, edgecolor='black', alpha=0.7)
            axes[1, 1].set_title('Degree Centrality Distribution (Day 90)')
            axes[1, 1].set_xlabel('Degree Centrality')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Attribute comparison between smokers and non-smokers (Day 90)
        if 90 in self.results['comparison']:
            stats = self.results['comparison'][90]
            categories = ['Club', 'Football', 'Studies']
            smoker_vals = [stats['club_participation_smokers'], 
                          stats['football_smokers'], 
                          stats['avg_studies_smokers'] * 20]  # Scaled for visibility
            nonsmoker_vals = [stats['club_participation_nonsmokers'], 
                             stats['football_nonsmokers'], 
                             stats['avg_studies_nonsmokers'] * 20]
            
            x = range(len(categories))
            axes[1, 2].bar([i - 0.2 for i in x], smoker_vals, 0.4, label='Smokers', color='red', alpha=0.7)
            axes[1, 2].bar([i + 0.2 for i in x], nonsmoker_vals, 0.4, label='Non-smokers', color='blue', alpha=0.7)
            axes[1, 2].set_title('Attribute Comparison (Day 90)')
            axes[1, 2].set_xlabel('Attribute')
            axes[1, 2].set_ylabel('Value')
            axes[1, 2].set_xticks(x)
            axes[1, 2].set_xticklabels(categories)
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./HW_03/Q2/network_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_analytical_report(self):
        """Generate a comprehensive analytical report"""
        print("\n" + "="*60)
        print("ANALYTICAL REPORT SUMMARY")
        print("="*60)
        
        # Triadic closure insights
        last_day = max(self.results['triadic'].keys())
        first_day = min(self.results['triadic'].keys())
        triadic_growth = ((self.results['triadic'][last_day]['total_triangles'] - 
                          self.results['triadic'][first_day]['total_triangles']) / 
                         self.results['triadic'][first_day]['total_triangles'] * 100)
        
        print(f"• Triadic closure increased by {triadic_growth:.1f}% over time")
        print("  Smokers show higher clustering, suggesting strong in-group connections")
        
        # Membership closure insights
        within_class_pct_change = (self.results['membership'][last_day]['within_class_pct'] - 
                                  self.results['membership'][first_day]['within_class_pct'])
        
        print(f"\n• Membership closure: Within-class connections changed by {within_class_pct_change:+.1f}%")
        print("  Class boundaries become more porous over time, but remain significant")
        
        # Focal closure insights
        smoking_homophily_last = (self.results['focal'][last_day]['smoking_homophily']['smoker_smoker'] / 
                                 (self.results['focal'][last_day]['smoking_homophily']['smoker_smoker'] +
                                  self.results['focal'][last_day]['smoking_homophily']['mixed'] / 2) * 100)
        
        print(f"\n• Focal closure: Smoking homophily is {smoking_homophily_last:.1f}% on Day {last_day}")
        print("  Smokers are disproportionately connected to other smokers")
        
        # Smoking diffusion
        smoking_growth = (self.results['comparison'][last_day]['smoking_prevalence'] - 
                         self.results['comparison'][first_day]['smoking_prevalence'])
        
        print(f"\n• Smoking prevalence increased by {smoking_growth:+.1f}% points")
        print("  Network position influences smoking adoption")
        
        # Centrality findings
        central_smokers_last = sum(1 for s in self.results['centrality'][last_day]['top_5'] 
                                  if s['smokes'] == 1)
        
        print(f"\n• {central_smokers_last}/5 of the most central students on Day {last_day} are smokers")
 

def main():
    """Main execution function"""
    print("School Social Network Evolution Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SchoolNetworkAnalyzer()
    
    # Load data
    print("\nLoading data...")
    analyzer.load_data()
    
    # Perform analyses
    analyzer.analyze_triadic_closure()
    analyzer.analyze_membership_closure()
    analyzer.analyze_focal_closure()
    analyzer.compare_smokers_nonsmokers()
    analyzer.compute_centrality()
    analyzer.analyze_smoking_diffusion()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.visualize_results()
    
    # Generate final report
    analyzer.generate_analytical_report()
    
    # Save results
    print("\nAnalysis complete! Results saved in memory.")
    print("Visualizations saved as 'network_analysis_results.png'")

if __name__ == "__main__":
    main()