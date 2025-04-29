from utils import *
from typing import Dict, List, Set, Tuple
import networkx as nx
import pandas as pd
from itertools import combinations
from earth_movers import *
import itertools
from matplotlib_venn import venn3, venn3_circles

def hausdorff_distance(G: nx.Graph, module1: List[str], module2: List[str]) -> float:
    min_dist = np.inf
    for gene1 in module1:
        for gene2 in module2:
            dist = nx.shortest_path_length(G, source=gene1, target=gene2)
            if dist < min_dist:
                min_dist = dist
    return min_dist

def count_neighbors(G: nx.Graph, modules1: List[List[str]], modules2: List[List[str]]) -> int:
    # count modules that are neighbors (hausdorff distance = 1)
    count = 0
    for module1 in modules1:
        for module2 in modules2:
            if hausdorff_distance(G, module1, module2) <= 1:
                count += 1
    return count

def merge_by_conductance_pairwise(G: nx.Graph, modules1: List[List[str]], modules2: List[List[str]]) -> List[List[str]]:
    merged_modules = []
    times_merged = 0
    for module1 in modules1: 
        merged = G.subgraph(module1)
        for module2 in modules2: 
            temp = nx.compose(merged, G.subgraph(module2))
            if conductance(G, temp) <= min(conductance(G, module1), conductance(G, module2)):
                times_merged += 1
                merged = temp
        merged_modules.append(merged)
    #print("stage 1", sorted([len(module) for module in merged_modules]))
    for module2 in modules2: 
        merged = G.subgraph(module2)
        for module1 in modules1: 
            temp = nx.compose(merged, G.subgraph(module1))
            if conductance(G, temp) <= min(conductance(G, module1), conductance(G, module2)):
                times_merged += 1
                merged = temp
        merged_modules.append(merged)
    print("times merged", times_merged)
    # drop duplicates
    merged_modules = list(set([tuple(sorted(list(module.nodes))) for module in merged_modules]))
    #print("stage 2", sorted([len(module) for module in merged_modules]))
    return merged_modules

def conductance(G: nx.Graph, module: List[str]) -> float:
    return nx.conductance(G, G.subgraph(module))

def merge_by_conductance(algorithms: List[str], total_modules: List[List[Set[int]]], G: nx.Graph, verbosity=0) -> List[Set[int]]:
    n = len(algorithms)
    conductance_ratios = {}
    
    # Compute initial conductance matrices
    for i in range(n):
        for j in range(i + 1, n):
            conductance_ratios[(i, j)] = compute_conductance_ratio_matrix(total_modules[i], total_modules[j], G)
    if verbosity > 1:
        for (i,j), matrix in conductance_ratios.items():
            print(i, j)
            print(matrix)
    merged = []
    i, j, k, l = find_minimum_across_matrices(conductance_ratios)
    merges = 0
    used_modules = []
    while conductance_ratios[(i, j)][k, l] <= 1:
        if verbosity > 0:
            if i != n and j != n:
                print(f"Merging {algorithms[i]}[{k}] (size {len(total_modules[i][k])}) and {algorithms[j]}[{l}] (size {len(total_modules[j][l])})")
            elif i == n and j != n:
                print(f"Merging merged[{k}] (size {len(merged[k])}) and {algorithms[j]}[{l}] (size {len(total_modules[j][l])})")
            elif i != n and j == n:
                print(f"Merging {algorithms[i]}[{k}] (size {len(total_modules[i][k])}) and merged[{l}] (size {len(merged[l])})")
        merges += 1
        if i != n and j != n:
            merged_module = nx.compose(G.subgraph(total_modules[i][k]), G.subgraph(total_modules[j][l]))
            used_modules.append((i,k))
            used_modules.append((j,l))
            merged.append(merged_module)
        elif i == n and j != n: 
            merged_module = nx.compose(G.subgraph(merged[k]), G.subgraph(total_modules[j][l]))
            used_modules.append((j,l))
            merged[k] = merged_module

        elif i != n and j == n:
            merged_module = nx.compose(G.subgraph(total_modules[i][k]), G.subgraph(merged[l]))
            used_modules.append((i,k))
            merged[l] = merged_module
        else: 
            raise Exception("tragic error")

        # Update conductance matrices with merged set
        for m in range(n):
            if (n,m) in conductance_ratios:
                temp = conductance_ratios[(n,m)]
                # determine indices where values are infinity
                inf_indices = np.where(temp == np.inf)
                conductance_ratios[(n, m)] = compute_conductance_ratio_matrix(merged, total_modules[m], G)
                # set values to infinity where they were infinity before
                conductance_ratios[(n, m)][inf_indices] = np.inf

                # iterate through inf_indices
                for x, y in zip(*inf_indices):
                    conductance_ratios[(n,m)][:,y] = np.inf
                    conductance_ratios[(n,m)][x,:] = np.inf
            else: 
                conductance_ratios[(n, m)] = compute_conductance_ratio_matrix(merged, total_modules[m], G)
        
        conductance_ratios = remove_from_matrices(conductance_ratios, i, j, k, l, used_modules)
        if verbosity > 2:
            for (i,j), matrix in conductance_ratios.items():
                print(i, j)
                print(matrix)
        i, j, k, l = find_minimum_across_matrices(conductance_ratios)
    if verbosity > 1:
        for (i,j), matrix in conductance_ratios.items():
            print(i, j)
            print(matrix)
    # Add remaining modules
    remaining_modules = []
    for i in range(n):
        for j in range(len(total_modules[i])):
            if (i,j) not in used_modules:
                remaining_modules.append(G.subgraph(total_modules[i][j]))
    for m in remaining_modules:
        merged.append(m)

    # cast them back to list of strings
    merged = [list([node for node in module.nodes]) for module in merged]
    return merged

def compute_conductance_ratio_matrix(modules_1: List[Set[int]], modules_2: List[Set[int]], G: nx.Graph) -> np.ndarray:
    n, m = len(modules_1), len(modules_2)
    A = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            merged = nx.compose(G.subgraph(modules_1[i]), G.subgraph(modules_2[j]))
            A[i, j] = conductance(G, merged) / min(conductance(G, modules_1[i]), conductance(G, modules_2[j]))

    return A

def remove_from_matrices(conductance_ratios: Dict[Tuple[int, int], np.ndarray], i: int, j: int, k: int, l: int, used_modules):
    n = max(max(key) for key in conductance_ratios.keys())
    # number of merged modules 
    n_merged = len(conductance_ratios[(n, 0)]) - 1
    if i != n and j != n:
        conductance_ratios[(n, i)][n_merged, :] = np.inf
        conductance_ratios[(n, j)][n_merged, :] = np.inf
    for x, y in used_modules:
        conductance_ratios[(n, x)][:, y] = np.inf

    for x in range(n):
        if i != n:
            if (i, x) in conductance_ratios:
                conductance_ratios[(i, x)][k, :] = np.inf
            if (x, i) in conductance_ratios:
                conductance_ratios[(x, i)][:, k] = np.inf
        if j != n:
            if (j, x) in conductance_ratios:
                conductance_ratios[(j, x)][l, :] = np.inf
            if (x, j) in conductance_ratios:
                conductance_ratios[(x, j)][:, l] = np.inf
    return conductance_ratios

def find_minimum_across_matrices(conductance_ratios: Dict[Tuple[int, int], np.ndarray]) -> Tuple[int, int, int, int]:
    min_value = np.inf
    min_indices = None
    
    for (i, j), matrix in conductance_ratios.items():
        min_in_matrix = np.min(matrix)
        if min_in_matrix < min_value:
            min_value = min_in_matrix
            k, l = np.unravel_index(np.argmin(matrix), matrix.shape)
            min_indices = (i, j, k, l)
    
    return min_indices

def create_module_venn_diagrams(all_modules, algorithms, network, dataset_name):
    """
    Create Venn diagrams showing module overlap between algorithms for a dataset.
    
    Parameters:
    -----------
    all_modules : Dict[str, List[List[str]]]
        Dictionary mapping algorithm names to their modules.
    algorithms : List[str]
        List of algorithm names to include in the diagram.
    network : nx.Graph
        Network graph for calculating conductance.
    dataset_name : str
        Name of the dataset for the title.
    """
    output_dir = "venn_diagrams"
    os.makedirs(output_dir, exist_ok=True)
    
    # First create sets of genes from each algorithm's modules
    gene_sets = {}
    for algo in algorithms:
        all_genes = set()
        for module in all_modules[algo]:
            all_genes.update(module)
        gene_sets[algo] = all_genes
    
    # Get the original module counts
    module_counts = {algo: len(modules) for algo, modules in all_modules.items()}
    
    # Prepare for merging - convert to the format expected by merge_by_conductance
    modules_list = [all_modules[algo] for algo in algorithms]
    
    # Perform the merge
    merged_modules = merge_by_conductance(algorithms, modules_list, network, verbosity=0)
    
    # Track which algorithm each gene came from
    algo_gene_map = {}
    for algo_idx, algo in enumerate(algorithms):
        for gene in gene_sets[algo]:
            if gene not in algo_gene_map:
                algo_gene_map[gene] = set()
            algo_gene_map[gene].add(algo_idx)
    
    # Determine membership pattern for each merged module
    module_patterns = {"100": 0, "010": 0, "001": 0, "110": 0, "101": 0, "011": 0, "111": 0}
    
    for module in merged_modules:
        # Convert module to set for faster lookups
        module_genes = set(module)
        
        # Determine which algorithms contributed to this module
        contributing_algos = set()
        for gene in module_genes:
            if gene in algo_gene_map:
                contributing_algos.update(algo_gene_map[gene])
        
        # Create binary pattern and increment count
        pattern = ""
        for i in range(3):
            pattern += "1" if i in contributing_algos else "0"
        
        module_patterns[pattern] += 1
    
    # Create the Venn diagram
    plt.figure(figsize=(10, 8))
    v = venn3(
        subsets=(
            module_patterns["100"],  # PAPER only
            module_patterns["010"],  # DOMINO only
            module_patterns["110"],  # PAPER + DOMINO
            module_patterns["001"],  # FDRnet only
            module_patterns["101"],  # PAPER + FDRnet
            module_patterns["011"],  # DOMINO + FDRnet
            module_patterns["111"]   # All three
        ),
        set_labels=algorithms
    )
    
    # Add more detailed labels
    for i, algo in enumerate(algorithms):
        idx = 2**i  # 1, 2, or 4
        if v and v.get_label_by_id(str(idx)):
            v.get_label_by_id(str(idx)).set_text(f"{algo}\n{module_patterns[bin(idx)[2:].zfill(3)]}/{module_counts[algo]}")
    
    plt.title(f"Module Merging for {dataset_name}")
    plt.savefig(f"{output_dir}/{dataset_name}_merged_modules.png", dpi=300, bbox_inches='tight')
    
    # Print statistics
    print(f"\nModule merging statistics for {dataset_name}:")
    total_original = sum(module_counts.values())
    total_merged = sum(module_patterns.values())
    
    print(f"Original modules: {total_original}")
    print(f"Merged modules: {total_merged}")
    print(f"Reduction: {total_original - total_merged} modules ({(total_original - total_merged)/total_original*100:.1f}%)")
    
    # Calculate overlap percentages
    for pattern, count in sorted(module_patterns.items()):
        if count > 0:
            algo_names = [algorithms[i] for i, bit in enumerate(pattern) if bit == "1"]
            if len(algo_names) == 1:
                print(f"Modules unique to {algo_names[0]}: {count}")
            else:
                print(f"Modules shared between {' + '.join(algo_names)}: {count}")
    
    return v


if __name__ == '__main__': 
    
    algos = ["PAPER", "DOMINO", "FDRnet"]
    res_dir = "/lab01/Projects/Jason_Projects/aneuploidy/ppi/test/test_07-21-2024_EMP-benchmark/true_solutions"

    illumina_modules = get_all_modules(res_dir, algos, "gene_scores_illumina_v3")
    tvc_modules = get_all_modules(res_dir, algos, "gene_scores_tvc")
    tnfa_modules = get_all_modules(res_dir, algos, "tnfa")
    fly_transcriptome_modules = get_all_modules(res_dir, algos, "fly_transcriptome")

    three_dbs = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/processed/network_full_ids_v2.tsv", source="node1", target="node2")
    dip = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/original/dip_original.sif", source="node1", target="node2")
    string_db = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/processed/STRING_v12_FBgn_geq900.sif", source="node1", target="node2")
    

    illumina_list = [illumina_modules[algo] for algo in algos]
    res = merge_by_conductance(algos, illumina_list, three_dbs, verbosity=1)
    sum = 0
    for modules in illumina_list: 
        #print(sorted([len(module) for module in modules]))
        sum += len(modules)
    #print(sorted([len(module) for module in res]))
    print(f"clusters: {sum - len(res)}")
    print("original module count: paper", len(illumina_modules["PAPER"]), "domino", len(illumina_modules["DOMINO"]), "fdrnet", len(illumina_modules["FDRnet"]))

    tvc_list = [tvc_modules[algo] for algo in algos]
    res = merge_by_conductance(algos, tvc_list, three_dbs, verbosity=1)
    sum = 0
    for modules in tvc_list: 
        #print(sorted([len(module) for module in modules]))
        sum += len(modules)
    #print(sorted([len(module) for module in res]))
    print(f"clusters: {sum - len(res)}")
    print("original module count: paper", len(tvc_modules["PAPER"]), "domino", len(tvc_modules["DOMINO"]), "fdrnet", len(tvc_modules["FDRnet"]))
    
    tnfa_list = [tnfa_modules[algo] for algo in algos]
    res = merge_by_conductance(algos, tnfa_list, dip, verbosity=1)
    sum = 0
    for modules in tnfa_list: 
        #print(sorted([len(module) for module in modules]))
        sum += len(modules)
    #print(sorted([len(module) for module in res]))
    print(f"clusters: {sum - len(res)}")
    print("original module count: paper", len(tnfa_modules["PAPER"]), "domino", len(tnfa_modules["DOMINO"]), "fdrnet", len(tnfa_modules["FDRnet"]))
    
    fly_transcriptome_list = [fly_transcriptome_modules[algo] for algo in algos]
    res = merge_by_conductance(algos, fly_transcriptome_list, string_db, verbosity=1)
    sum = 0
    for modules in fly_transcriptome_list: 
        #print(sorted([len(module) for module in modules]))
        sum += len(modules)
    #print(sorted([len(module) for module in res]))
    print(f"clusters: {sum - len(res)}")
    print("original module count: paper", len(fly_transcriptome_modules["PAPER"]), "domino", len(fly_transcriptome_modules["DOMINO"]), "fdrnet", len(fly_transcriptome_modules["FDRnet"]))


