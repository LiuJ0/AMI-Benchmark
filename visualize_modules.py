import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from typing import List, Set, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from collections import defaultdict
import os
import pandas as pd
from earth_movers import *

def visualize_similar_modules(G: nx.Graph, module1: List[int], module2: List[int], 
                              algo1: str, algo2: str, mod1: int, mod2: int,
                              name2ENSG: pd.DataFrame, save_path: str = None,
                              colors: Dict[str, str] = None, use_labels: bool = True):
    """
    Visualize two similar modules, with shared nodes half red and half blue.
    For disjoint modules, the shortest path is colored grey.
    
    :param G: The complete graph
    :param module1: List of node indices for the first module
    :param module2: List of node indices for the second module
    :param algo1: Name of the first algorithm
    :param algo2: Name of the second algorithm
    :param mod1: Module number for the first algorithm
    :param mod2: Module number for the second algorithm
    :param save_path: Path to save the figure (if None, figure is displayed)
    """
    if colors is None: 
        colors = {"algo1": "red", "algo2": "blue", "shared": "purple"}
    set1, set2 = set(module1), set(module2)
    shared_genes = set1.intersection(set2)

    id_to_name = dict(zip(name2ENSG['Gene stable ID'], name2ENSG['Gene name']))
    
    # Function to get gene name (or original ID if not found)
    def get_gene_name(gene_id):
        return id_to_name.get(gene_id, str(gene_id))
    
    # Determine the subgraph to visualize
    if not shared_genes:
        # Case 1: Disjoint modules
        path = nx.shortest_path(G, source=module1[0], target=module2[0])
        nodes_to_show = set(module1 + module2 + path)
    else:
        # Case 2: Overlapping modules
        nodes_to_show = set1.union(set2)
    
    subgraph = G.subgraph(nodes_to_show)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subgraph, k=0.5, iterations=20)

    pos_array = np.array(list(pos.values()))
    pos_min, pos_max = pos_array.min(axis=0), pos_array.max(axis=0)
    for node in pos:
        pos[node] = (pos[node] - pos_min) / (pos_max - pos_min)
    
    # Draw the subgraph
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2)
    
    # Color the nodes
    nx.draw_networkx_nodes(subgraph, pos, nodelist=set1 - set2, node_color=colors['algo1'], node_size=500)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=set2 - set1, node_color=colors['algo2'], node_size=500)
    
    # Draw shared nodes as half red, half blue
    for node in shared_genes:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=[node], node_color=colors['shared'], node_size=500)

    # Color the shortest path grey for disjoint modules
    if not shared_genes:
        path_nodes = set(path) - set1 - set2
        nx.draw_networkx_nodes(subgraph, pos, nodelist=path_nodes, node_color='grey', node_size=300)
    
    # Add labels
    if use_labels:
        labels = {node: get_gene_name(node) for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, font_weight='bold')
    
    # Add a legend
    red_patch = plt.Rectangle((0, 0), 1, 1, fc=colors['algo1'])
    blue_patch = plt.Rectangle((0, 0), 1, 1, fc=colors['algo2'])
    split_patch = plt.Rectangle((0, 0), 1, 1, fc=colors['shared'])
    legend_elements = [
        red_patch, blue_patch, split_patch,
        plt.Rectangle((0, 0), 1, 1, fc="grey") if not shared_genes else None
    ]
    legend_labels = [
        f'{algo1} (Module {mod1})', 
        f'{algo2} (Module {mod2})', 
        'Shared Genes',
        'Shortest Path' if not shared_genes else None
    ]
    legend_elements = [e for e in legend_elements if e is not None]
    legend_labels = [l for l in legend_labels if l is not None]
    plt.legend(legend_elements, legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add a title
    if not shared_genes:
        plt.title(f"Disjoint Modules: {algo1}_{mod1} vs {algo2}_{mod2}")
    else:
        plt.title(f"Overlapping Modules: {algo1}_{mod1} vs {algo2}_{mod2}")
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_conserved_modules(G: nx.Graph, conserved_modules: Dict[Tuple, List[Tuple]], modules: Dict[str, List[List[str]]], 
                                name2ENSG: pd.DataFrame, similarity_threshold: float = 0.7, output_dir: str = "conserved_modules", colors=None, use_labels=True):
    """
    Visualize conserved modules with similarity above the given threshold.
    
    :param G: The complete graph
    :param conserved_modules: Output from find_conserved_modules
    :param similarity_threshold: Minimum similarity to visualize
    :param output_dir: Directory to save the figures
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    similar_module_groups = {}
    processed_pairs = set()
    
    for (algo1, mod1), matches in conserved_modules.items():
        module1 = modules[algo1][mod1]
        if len(module1) < 3:
            continue  # Skip modules with fewer than 3 nodes
        
        for algo2, mod2, similarity in matches:
            module2 = modules[algo2][mod2]
            if len(module2) < 3:
                continue  # Skip modules with fewer than 3 nodes
            
            if True: # similarity >= similarity_threshold:
                # Create a unique identifier for this pair of modules
                pair_id = tuple(sorted([(algo1, mod1), (algo2, mod2)]))
                
                # Skip if we've already processed this pair
                if pair_id in processed_pairs:
                    continue
                processed_pairs.add(pair_id)
                
                if pair_id not in similar_module_groups:
                    group_id = len(similar_module_groups) + 1
                    similar_module_groups[pair_id] = group_id
                else:
                    group_id = similar_module_groups[pair_id]
                
                print(f"Similar Module Group {group_id}: {algo1} (Module {mod1}) vs {algo2} (Module {mod2}), Similarity: {similarity:.2f}")
                
                # round similarity to 2 decimal places
                save_path = os.path.join(output_dir, f"{group_id}_{algo1}_{mod1}_vs_{algo2}_{mod2}_{similarity:.3f}_{(1 - similarity)/(similarity):.3f}.png")
                visualize_similar_modules(G, module1, module2, algo1, algo2, mod1, mod2, name2ENSG, save_path, colors, use_labels=use_labels)

    print(f"\nTotal number of similar module groups: {len(similar_module_groups)}")
    