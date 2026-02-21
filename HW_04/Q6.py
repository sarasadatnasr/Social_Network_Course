"""
Bonus Question 6: Operation "Dormant Cell" Detection
Custom algorithm for identifying suspicious hidden communities

Output:
- ./images/Q6/full_network_dormant_cell.png
- ./images/Q6/dormant_cell_subgraph.png
- ./images/Q6/suspicion_scores_bar.png
"""

from collections import defaultdict
from typing import Dict, List
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import community as community_louvain


def calculate_internal_density(G: nx.Graph, members: set) -> float:
    """
    Calculate internal density of a community
    density = actual edges / maximum possible edges
    Maximum possible edges = n*(n-1)/2 for undirected graph
    """
    n = len(members)
    if n < 2:
        return 0.0

    internal_edges = 0
    for u in members:
        for v in members:
            if u < v and G.has_edge(u, v):  # u < v to avoid double counting
                internal_edges += 1

    max_possible = n * (n - 1) / 2
    return internal_edges / max_possible


def calculate_external_connectivity(G: nx.Graph, members: set, all_nodes: set) -> float:
    """
    Calculate external connectivity ratio
    connectivity = edges to outside / total possible external edges
    Lower is better (more isolated)
    """
    n = len(members)
    if n == 0:
        return 1.0

    external_edges = 0
    for u in members:
        for v in G.neighbors(u):
            if v not in members:
                external_edges += 1

    non_members = all_nodes - members
    max_possible = n * len(non_members)

    if max_possible == 0:
        return 0.0

    return external_edges / max_possible


def calculate_size_score(size: int, avg_size: float) -> float:
    """
    Calculate size optimality score
    size_score = 1 - |size - avg_size| / avg_size
    Closer to average size = higher score
    """
    if avg_size == 0:
        return 0.0

    deviation = abs(size - avg_size) / avg_size
    return max(0.0, 1.0 - deviation)


def calculate_peripherality(G: nx.Graph, members: set) -> float:
    """
    Calculate network peripherality
    peripherality = 1 - average closeness centrality
    Higher peripherality = more peripheral (less central)
    """
    if len(members) == 0:
        return 0.0

    closeness = nx.closeness_centrality(G)
    avg_closeness = np.mean([closeness[node] for node in members if node in closeness])
    return 1.0 - avg_closeness


def generate_reasoning(community_data: Dict) -> str:
    """Generate detailed reasoning for why this community is suspicious"""
    chars = community_data["characteristics"]
    score = community_data["suspicion_score"]
    members = community_data["members"][:5]
    member_list = ", ".join(map(str, members))

    reasoning = f"""
This community exhibits strong dormant cell characteristics with a suspicion score of {score:.3f}/1.000.

KEY INDICATORS:
• Internal Density ({chars['internal_density']:.3f}):
  {'High internal connectivity suggests tight-knit operational security' if chars['internal_density'] > 0.3 else 'Moderate internal cohesion'}

• External Isolation ({(1 - chars['external_connectivity']):.3f}):
  {'Strong compartmentalization - minimal contact with outsiders' if (1 - chars['external_connectivity']) > 0.7 else 'Some external contacts detected'}

• Size Optimality ({chars['size_score']:.3f}):
  Size {chars['size']} members - {'optimal operational size' if chars['size_score'] > 0.7 else 'non-standard cell size'}

• Peripherality ({chars['peripherality']:.3f}):
  {'Operating at network periphery - avoiding surveillance' if chars['peripherality'] > 0.5 else 'Located in more central network positions'}

Members include: {member_list}...
    """
    return reasoning.strip()


def find_dormant_cell(G: nx.Graph, partition: Dict) -> Dict:
    """
    Identify the most suspicious community based on dormant cell characteristics.
    Returns result dict including all community scores for visualization.
    """
    communities = defaultdict(set)
    for node, comm_id in partition.items():
        communities[comm_id].add(node)

    all_nodes = set(G.nodes())

    valid_communities = [comm for comm in communities.values() if len(comm) >= 2]
    avg_size = np.mean([len(comm) for comm in valid_communities]) if valid_communities else 5.0

    w1, w2, w3, w4 = 0.35, 0.35, 0.15, 0.15

    community_scores = []
    for comm_id, members in communities.items():
        if len(members) < 2:
            continue

        internal_density = calculate_internal_density(G, members)
        external_connectivity = calculate_external_connectivity(G, members, all_nodes)
        size_score = calculate_size_score(len(members), avg_size)
        peripherality = calculate_peripherality(G, members)

        suspicion_score = (
            w1 * internal_density +
            w2 * (1.0 - external_connectivity) +
            w3 * size_score +
            w4 * peripherality
        )

        community_scores.append({
            "community_id": comm_id,
            "suspicion_score": float(suspicion_score),
            "members": list(members),
            "characteristics": {
                "internal_density": float(internal_density),
                "external_connectivity": float(external_connectivity),
                "size": int(len(members)),
                "size_score": float(size_score),
                "peripherality": float(peripherality),
                "avg_closeness": float(1.0 - peripherality),
            }
        })

    if not community_scores:
        return {
            "suspected_community_id": -1,
            "suspicion_score": 0.0,
            "members": [],
            "characteristics": {},
            "reasoning": "No valid communities found with at least 2 members.",
            "all_scores": []
        }

    top_community = max(community_scores, key=lambda x: x["suspicion_score"])
    reasoning = generate_reasoning(top_community)

    return {
        "suspected_community_id": top_community["community_id"],
        "suspicion_score": top_community["suspicion_score"],
        "members": top_community["members"],
        "characteristics": {
            "internal_density": top_community["characteristics"]["internal_density"],
            "external_connectivity": top_community["characteristics"]["external_connectivity"],
            "size": top_community["characteristics"]["size"],
            "peripherality": top_community["characteristics"]["peripherality"],
        },
        "reasoning": reasoning,
        "all_scores": community_scores
    }


def visualize_dormant_cell(G: nx.Graph, partition: Dict, dormant_result: Dict, out_dir: str = "./images/Q6"):
    """
    Generate THREE SEPARATE figures and save each as a PNG.

    Saves:
    1) full_network_dormant_cell.png
    2) dormant_cell_subgraph.png
    3) suspicion_scores_bar.png
    """
    os.makedirs(out_dir, exist_ok=True)

    dormant_members = set(dormant_result["members"])
    suspected_id = dormant_result["suspected_community_id"]

    # Shared layout for full network consistency
    pos = nx.spring_layout(G, seed=42)

    # -------------------------
    # Plot 1: Full network
    # -------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    node_colors = ["red" if node in dormant_members else "lightblue" for node in G.nodes()]

    nx.draw(
        G, pos,
        node_color=node_colors,
        node_size=100,
        font_size=8,
        with_labels=True,
        ax=ax1,
        edge_color="gray",
        alpha=0.7
    )
    ax1.set_title(f"Full Network - Dormant Cell (ID: {suspected_id}) in Red")
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "full_network_dormant_cell.png"), dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # -------------------------
    # Plot 2: Dormant cell subgraph
    # -------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    dormant_subgraph = G.subgraph(dormant_members)

    if dormant_subgraph.number_of_nodes() > 0:
        sub_pos = nx.spring_layout(dormant_subgraph, seed=42)
        nx.draw(
            dormant_subgraph, sub_pos,
            node_color="red",
            node_size=300,
            font_size=10,
            with_labels=True,
            ax=ax2,
            edge_color="darkred",
            width=2
        )
        ax2.set_title(f"Dormant Cell Internal Structure ({len(dormant_members)} members)")
    else:
        ax2.text(0.5, 0.5, "No internal connections", ha="center", va="center")
        ax2.set_title("Dormant Cell (Isolated Nodes)")

    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "dormant_cell_subgraph.png"), dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # -------------------------
    # Plot 3: Suspicion score comparison bar chart
    # -------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    all_scores = dormant_result.get("all_scores", [])

    if all_scores:
        community_ids = [s["community_id"] for s in all_scores]
        suspicion_scores = [s["suspicion_score"] for s in all_scores]

        colors = ["red" if cid == suspected_id else "steelblue" for cid in community_ids]

        bars = ax3.bar(range(len(community_ids)), suspicion_scores, color=colors, alpha=0.7)
        ax3.set_xlabel("Community ID")
        ax3.set_ylabel("Suspicion Score")
        ax3.set_title("Suspicion Scores Across All Communities")
        ax3.set_xticks(range(len(community_ids)))
        ax3.set_xticklabels(community_ids, rotation=45)

        for bar, score in zip(bars, suspicion_scores):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

        ax3.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Suspicion Threshold")
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No scores available", ha="center", va="center")
        ax3.set_title("Suspicion Scores Across All Communities")

    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "suspicion_scores_bar.png"), dpi=300, bbox_inches="tight")
    plt.close(fig3)


def analyze_les_miserables_characters(members: List[str]) -> str:
    """Analyze the roles of characters in Les Misérables"""
    character_roles = {
        "Valjean": "Jean Valjean - Protagonist, former convict turned philanthropist",
        "Gavroche": "Gavroche - Street urchin, revolutionary",
        "Marius": "Marius Pontmercy - Student revolutionary, loves Cosette",
        "Cosette": "Cosette - Valjean's adopted daughter",
        "Javert": "Inspector Javert - Police inspector obsessed with Valjean",
        "Thenardier": "Thénardier - Corrupt innkeeper, criminal",
        "Eponine": "Éponine - Thénardier's daughter, loves Marius",
        "Enjolras": "Enjolras - Leader of the ABC revolutionary students",
        "Courfeyrac": "Courfeyrac - ABC revolutionary",
        "Combeferre": "Combeferre - ABC revolutionary",
        "Feuilly": "Feuilly - ABC revolutionary, worker",
        "Bahorel": "Bahorel - ABC revolutionary",
        "Bossuet": "Bossuet - ABC revolutionary",
        "Joly": "Joly - ABC revolutionary",
        "Grantaire": "Grantaire - ABC revolutionary, skeptic",
        "Myriel": "Bishop Myriel - Kind bishop who inspired Valjean",
        "Fantine": "Fantine - Tragic factory worker, Cosette's mother",
        "MmeThenardier": "Madame Thénardier - Thénardier's wife",
    }

    analysis = "CHARACTER ANALYSIS:\n"
    for member in members:
        if member in character_roles:
            analysis += f"• {character_roles[member]}\n"
        else:
            analysis += f"• {member} - Character in Les Misérables\n"
    return analysis


if __name__ == "__main__":
    # Load network
    G = nx.les_miserables_graph()

    print("=" * 80)
    print('OPERATION: DORMANT CELL DETECTION')
    print("=" * 80)

    # Community detection
    print("Running Louvain community detection...")
    partition = community_louvain.best_partition(G)

    num_communities = len(set(partition.values()))
    print(f"Analyzing {num_communities} detected communities...")

    # Execute dormant cell detection
    dormant = find_dormant_cell(G, partition)

    # Display results
    print("\n" + "=" * 80)
    print("🚨 DORMANT CELL DETECTED")
    print("=" * 80)
    print(f"Community ID: {dormant['suspected_community_id']}")
    print(f"Suspicion Score: {dormant['suspicion_score']:.3f}/1.000")

    print(f"\nMembers ({len(dormant['members'])}):")
    for i, member in enumerate(sorted(dormant["members"]), 1):
        print(f"  {i:2d}. {member}")

    print("\nCharacteristics:")
    for key, value in dormant["characteristics"].items():
        if key != "size":
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print("\nAnalysis:")
    print(dormant["reasoning"])

    print("\n" + "-" * 80)
    print(analyze_les_miserables_characters(dormant["members"]))

    # Generate visualizations (separate PNGs)
    visualize_dormant_cell(G, partition, dormant, out_dir="./images/Q6")

    print("\n✓ Mission Complete")