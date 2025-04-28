import numpy as np
from utils import *
import networkx as nx
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import os
import pandas as pd
from earth_movers import *

def visualize_similar_modules(G: nx.Graph, module1: List[int], module2: List[int], 
                              algo1: str, algo2: str, mod1: int, mod2: int,
                              name2ENSG: pd.DataFrame, save_path: str = None,
                              colors: Dict[str, str] = None, use_labels: bool = False):
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
        # Case 1: Disjoint modules - find the shortest path between any pair of nodes
        shortest_path = None
        min_path_length = float('inf')
        
        # Find shortest path between any pair of nodes from the two modules
        for node1 in module1:
            for node2 in module2:
                try:
                    path = nx.shortest_path(G, source=node1, target=node2)
                    if len(path) < min_path_length:
                        min_path_length = len(path)
                        shortest_path = path
                except nx.NetworkXNoPath:
                    continue
        if shortest_path is None:
            # If no path exists, just show both modules separately
            nodes_to_show = set(module1 + module2)
        else:
            nodes_to_show = set(module1 + module2 + shortest_path)
        print('Shortest Path', shortest_path, 'Length', min_path_length)
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
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5, width=2.5)
    
    # Color the nodes - increase node size from 500 to 800
    nx.draw_networkx_nodes(subgraph, pos, nodelist=set1 - set2, node_color=colors['algo1'], node_size=800)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=set2 - set1, node_color=colors['algo2'], node_size=800)
    
    # Draw shared nodes with increased size from 500 to 800
    for node in shared_genes:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=[node], node_color=colors['shared'], node_size=800)

    # Color the shortest path grey for disjoint modules - increase from 300 to 600
    if not shared_genes and shortest_path is not None:
        path_nodes = set(shortest_path) - set1 - set2
        nx.draw_networkx_nodes(subgraph, pos, nodelist=path_nodes, node_color='grey', node_size=800)
    
    # Add labels (now default to False)
    if use_labels:
        labels = {node: get_gene_name(node) for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, font_weight='bold')
    
    # Add a legend with larger font size
    red_patch = plt.Rectangle((0, 0), 1, 1, fc=colors['algo1'])
    blue_patch = plt.Rectangle((0, 0), 1, 1, fc=colors['algo2'])
    split_patch = plt.Rectangle((0, 0), 1, 1, fc=colors['shared'])
    legend_elements = [
        red_patch, blue_patch, split_patch if shared_genes else None,
        plt.Rectangle((0, 0), 1, 1, fc="grey") if not shared_genes and shortest_path is not None else None
    ]
    legend_labels = [
        f'{algo1} (Module {mod1})', 
        f'{algo2} (Module {mod2})', 
        'Shared Genes' if shared_genes else None,
        'Shortest Path' if not shared_genes and shortest_path is not None else None
    ]
    
    legend_elements = [e for e in legend_elements if e is not None]
    legend_labels = [l for l in legend_labels if l is not None]
    plt.legend(legend_elements, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=18)
    
    # Title removed
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_conserved_modules(G: nx.Graph, conserved_modules: Dict[Tuple, List[Tuple]], modules: Dict[str, List[List[str]]], 
                                name2ENSG: pd.DataFrame, similarity_threshold: float = 0.7, output_dir: str = "conserved_modules", colors=None, use_labels=False):
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
                save_path = os.path.join(output_dir, f"{group_id}_{algo1}_{mod1}_vs_{algo2}_{mod2}_{similarity:.3f}_{(1 - similarity)/(similarity):.3f}.svg")
                G.remove_edges_from(list(nx.selfloop_edges(G)))
                visualize_similar_modules(G, module1, module2, algo1, algo2, mod1, mod2, name2ENSG, save_path, colors, use_labels=use_labels)

    print(f"\nTotal number of similar module groups: {len(similar_module_groups)}")
    
if __name__ == "__main__":
    name2ENSG = pd.read_csv("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/original/Name2ENSG_GRCh38.csv")
    name2ENSG_fb = pd.read_csv("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/original/name2ENSG_fb.tsv", sep='\t')

    three_dbs = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/processed/network_full_ids_v2.tsv", source="node1", target="node2")
    dip = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/original/dip_original.sif", source="node1", target="node2")
    # make dip a simple graph
    dip.remove_edges_from(nx.selfloop_edges(dip))
    string_db = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/processed/STRING_v12_FBgn_geq900.sif", source="node1", target="node2")

    algos = ["PAPER", "DOMINO", "HotNet2", "FDRnet"]
    res_dir = "/lab01/Projects/Jason_Projects/aneuploidy/ppi/test/test_07-21-2024_EMP-benchmark/true_solutions"
    modules_report_dir = "/lab01/Projects/Jason_Projects/aneuploidy/ppi/test/test_07-21-2024_EMP-benchmark/report/module_cache_files"

    illumina_modules = get_all_modules(res_dir, algos, "gene_scores_illumina_v3")
    tvc_modules = get_all_modules(res_dir, algos, "gene_scores_tvc")
    tnfa_modules = get_all_modules(res_dir, algos, "tnfa")
    fly_transcriptome_modules = get_all_modules(res_dir, algos, "fly_transcriptome")
    all_modules = combine_dictionaries(illumina_modules, tvc_modules, tnfa_modules, fly_transcriptome_modules)


    illumina_conserved_modules = find_conserved_modules(three_dbs, illumina_modules, threshold=0)
    tvc_conserved_modules = find_conserved_modules(three_dbs, tvc_modules, threshold=0)
    tnfa_conserved_modules = find_conserved_modules(dip, tnfa_modules, threshold=0)
    fly_conserved_modules = find_conserved_modules(string_db, fly_transcriptome_modules, threshold=0)

    colors = {
        "algo1": "#d47264", # red
        "algo2": "#8ec1da", # blue 
        "shared": "#a559aa", # purple
        "path": "#f0f0f0" # grey
    }

    visualize_conserved_modules(three_dbs, illumina_conserved_modules, illumina_modules, name2ENSG, 
                                similarity_threshold = 0, output_dir="../figures/conserved_modules_illumina", colors=colors, use_labels=True)

    visualize_conserved_modules(three_dbs, tvc_conserved_modules, tvc_modules, name2ENSG,
                                similarity_threshold = 0, output_dir="../figures/conserved_modules_tvc", colors=colors, use_labels=True)

    visualize_conserved_modules(dip, tnfa_conserved_modules, tnfa_modules, name2ENSG,
                                similarity_threshold = 0, output_dir="../figures/conserved_modules_tnfa", colors=colors, use_labels=True)

    visualize_conserved_modules(string_db, fly_conserved_modules, fly_transcriptome_modules, name2ENSG_fb,
                                similarity_threshold = 0, output_dir="../figures/conserved_modules_fly_transcriptome", colors=colors, use_labels=True)