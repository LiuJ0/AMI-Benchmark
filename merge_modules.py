from typing import Dict, List, Set, Tuple
import networkx as nx
import pandas as pd
from itertools import combinations
from earth_movers import *
import itertools

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

if __name__ == '__main__': 
    def read_modules(algo_dir, algo_name, dataset_name):
        modules_path = os.path.join(algo_dir, dataset_name + "_" + algo_name, "report")
        # Read any files with that start with "ALGO_NAME"_module_genes_*
        modules = []
        for file in os.listdir(modules_path):
            if file.startswith(algo_name + "_module_genes_"):
                modules.append([x.strip() for x in open(os.path.join(modules_path, file)).readlines()])
        return modules

    def network_to_graph(network_file, source="node1", target="node2"): 
        network = pd.read_csv(network_file, sep="\t")
        return nx.from_pandas_edgelist(network, source=source, target=target, edge_attr=True, create_using=nx.Graph)

    def get_all_modules(res_dir, algos, dataset): 
        modules = {}
        for algo in algos: 
            modules[algo] = read_modules(res_dir, algo, dataset)
        return modules
    
    def merge_pairwise(dataset, network, algo1, algo2): 
        merged_modules = merge_by_conductance_pairwise(network, dataset[algo1], dataset[algo2])
        print(algo1, sorted([len(module) for module in dataset[algo1]]))
        print(algo2, sorted([len(module) for module in dataset[algo2]]))
        print("merged", sorted([len(module) for module in merged_modules]))
        return merged_modules
    
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
        print(sorted([len(module) for module in modules]))
        sum += len(modules)
    print(sorted([len(module) for module in res]))
    print(f"clusters: {sum - len(res)}")

    tvc_list = [tvc_modules[algo] for algo in algos]
    res = merge_by_conductance(algos, tvc_list, three_dbs, verbosity=1)
    sum = 0
    for modules in tvc_list: 
        print(sorted([len(module) for module in modules]))
        sum += len(modules)
    print(sorted([len(module) for module in res]))
    print(f"clusters: {sum - len(res)}")
    # print(res)
    
    tnfa_list = [tnfa_modules[algo] for algo in algos]
    res = merge_by_conductance(algos, tnfa_list, dip, verbosity=1)
    sum = 0
    for modules in tnfa_list: 
        print(sorted([len(module) for module in modules]))
        sum += len(modules)
    print(sorted([len(module) for module in res]))
    print(f"clusters: {sum - len(res)}")

    fly_transcriptome_list = [fly_transcriptome_modules[algo] for algo in algos]
    res = merge_by_conductance(algos, fly_transcriptome_list, string_db, verbosity=1)
    sum = 0
    for modules in fly_transcriptome_list: 
        print(sorted([len(module) for module in modules]))
        sum += len(modules)
    print(sorted([len(module) for module in res]))
    print(f"clusters: {sum - len(res)}")
    

