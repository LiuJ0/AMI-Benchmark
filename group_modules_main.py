from group_modules import *
from merge_modules import *
from typing import Dict, List
from collections import defaultdict
import pandas as pd 
import networkx as nx
import os
import pickle 


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

def combine_dictionaries(*dictionaries: Dict[str, List[List[str]]]) -> Dict[str, List[List[str]]]:
    combined = defaultdict(list)

    for dictionary in dictionaries:
        for algorithm, list_of_lists in dictionary.items():
            combined[algorithm].extend(list_of_lists)

    return dict(combined)

def perform_clustering(unlabeled, labeled, save_paths=['cooc_eigenvalues.png', 'nl_eigenvalues.png', 'heatmap.png'], sep_lines=False, filter_genes=True, recursive=False):
    if filter_genes: 
        module_count = 0
        for algo, modules in labeled.items(): 
            for module in modules: 
                module_count += 1
        filtered_modules, removed = filter_non_overlapping_modules(labeled)
        # print("Removed module", removed)
        removed_genes = []
        for algo, module in removed: 
            removed_genes += module
        filtered_unlabeled = [module for modules in filtered_modules.values() for module in modules]
        cooccurence_matrix, gene_list = generate_cooccurrence_matrix(filtered_unlabeled)

        print(f"Number of modules: {module_count - len(removed)} | Number of genes: {len(gene_list)} | Number of removed modules: {len(removed)} | Number of removed genes: {len(removed_genes)}")
    # Illumina: 657 genes, 608 removed, 49 remaining
    # Tvc: 762 genes, 703 removed, 59 remaining
    # TnfA: 100 genes, 66 removed, 34 remaining
    # Fly transcriptome: 739 genes, 519 removed, 220 remaining
    else: 
        cooccurence_matrix, gene_list = generate_cooccurrence_matrix(unlabeled)
        filtered_unlabeled = unlabeled
        print(f"Number of genes: {len(gene_list)}")
    matrix = cooccurrence_to_matrix(cooccurence_matrix, gene_list)
    normalized_matrix = matrix # normalize(matrix, norm='l1', axis=1)
    plot_eigenvalues(normalized_matrix, save_path=save_paths[0])
    # Illumina: 7 clusters, Tvc: 5, TnfA: 6, Fly transcriptome: 10
    suggested_clusters_eigen = estimate_clusters_eigenvalues(normalized_matrix, save_path=save_paths[1], max_clusters=len(filtered_unlabeled)) # number of initial modules
    print(f"\nSuggested number of clusters (eigengap heuristic): {suggested_clusters_eigen}")
    if recursive:
        spectral_cluster_labels, initial = recursive_spectral_clustering(normalized_matrix, max_cluster_size=40, max_clusters=len(filtered_unlabeled) // 2, max_iterations=5)
        # spectral_cluster_labels, initial = recursive_spectral_clustering_silhouette(normalized_matrix, min_silhouette_score=0.3, max_clusters=len(filtered_unlabeled) // 2, max_iterations=5)
    else:
        spectral_cluster_labels = perform_spectral_clustering(normalized_matrix, suggested_clusters_eigen)
        initial = spectral_cluster_labels
    # hierarchical_cluster_labels = perform_hierarchical_clustering(normalized_matrix, suggested_clusters_eigen)

    # interpretation = interpret_clustering_results(labeled, gene_list, spectral_cluster_labels)
    # print_clustering_interpretation(interpretation, gene_list, spectral_cluster_labels)

    # matching = match_clusters_to_modules(spectral_cluster_labels, gene_list, labeled)
    # print_cluster_module_matching(matching, gene_list, spectral_cluster_labels)
    create_cooccurrence_heatmaps(matrix, gene_list, spectral_cluster_labels, save_path=save_paths[2], sep_lines=sep_lines)
    return gene_list, matrix, suggested_clusters_eigen, spectral_cluster_labels, initial

algos = ["PAPER", "DOMINO", "FDRnet"]
res_dir = "/lab01/Projects/Jason_Projects/aneuploidy/ppi/test/test_07-21-2024_EMP-benchmark/true_solutions"

illumina_modules = get_all_modules(res_dir, algos, "gene_scores_illumina_v3")
illumina_modules_nonredundant = remove_redundancies_from_dict(illumina_modules)
tvc_modules = get_all_modules(res_dir, algos, "gene_scores_tvc")
tvc_modules_nonredundant = remove_redundancies_from_dict(tvc_modules)
tnfa_modules = get_all_modules(res_dir, algos, "tnfa")
tnfa_modules_nonredundant = remove_redundancies_from_dict(tnfa_modules)
fly_transcriptome_modules = get_all_modules(res_dir, algos, "fly_transcriptome")
fly_transcriptome_modules_nonredundant = remove_redundancies_from_dict(fly_transcriptome_modules)

illumina_modules_unlabeled = [module for modules in illumina_modules.values() for module in modules]
tnfa_modules_unlabeled = [module for modules in tnfa_modules.values() for module in modules]
tvc_modules_unlabeled = [module for modules in tvc_modules.values() for module in modules]
fly_transcriptome_modules_unlabeled = [module for modules in fly_transcriptome_modules.values() for module in modules]

illumina_modules_filtered = remove_redundancies(illumina_modules_unlabeled)
tnfa_modules_filtered = remove_redundancies(tnfa_modules_unlabeled)
tvc_modules_filtered = remove_redundancies(tvc_modules_unlabeled)
fly_transcriptome_modules_filtered = remove_redundancies(fly_transcriptome_modules_unlabeled)

#print(len(illumina_modules_unlabeled)) # 145
#print(len(illumina_modules_filtered)) # 144
#print(len(tvc_modules_unlabeled)) # 122
#print(len(tvc_modules_filtered)) # 120

#print(len(tnfa_modules_unlabeled)) # 21 
#print(len(tnfa_modules_filtered)) # 19
#print(len(fly_transcriptome_modules_unlabeled)) # 95
#print(len(fly_transcriptome_modules_filtered)) # 80

three_dbs = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/processed/network_full_ids_v2.tsv", source="node1", target="node2")
dip = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/original/dip_original.sif", source="node1", target="node2")
string_db = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/processed/STRING_v12_FBgn_geq900.sif", source="node1", target="node2")



illumina_ordering, illumina_cooccurence_matrix, suggested_clusters_illumina, spectral_cluster_labels_illumina, nonrecursive_labels_illumina = perform_clustering(
    illumina_modules_filtered, illumina_modules, sep_lines=False, save_paths=['../figures/illumina_cooc_eigenvalues.png', '../figures/illumina_nl_eigenvalues.png', '../figures/illumina_heatmap.png'], filter_genes=True, recursive=False)

create_algorithm_heatmaps(illumina_modules, illumina_ordering, nonrecursive_labels_illumina, illumina_cooccurence_matrix, save_path='../figures/illumina_algorithm_heatmap.png', sep_lines=False)
res = match_clusters_to_modules(spectral_cluster_labels_illumina, illumina_ordering, illumina_modules)
print_cluster_module_matching(res, illumina_ordering, spectral_cluster_labels_illumina)


tvc_ordering, tvc_cooccurence_matrix, suggested_clusters_tvc, spectral_cluster_labels_tvc, nonrecursive_labels_tvc = perform_clustering(
    tvc_modules_filtered, tvc_modules, sep_lines=False, save_paths=['../figures/tvc_cooc_eigenvalues.png', '../figures/tvc_nl_eigenvalues.png', '../figures/tvc_heatmap.png'], filter_genes=True, recursive=False)

create_algorithm_heatmaps(tvc_modules, tvc_ordering, nonrecursive_labels_tvc, tvc_cooccurence_matrix, save_path='../figures/tvc_algorithm_heatmap.png', sep_lines=False)
res = match_clusters_to_modules(spectral_cluster_labels_tvc, tvc_ordering, tvc_modules)
print_cluster_module_matching(res, tvc_ordering, spectral_cluster_labels_tvc)

tnfa_ordering, tnfa_cooccurence_matrix, suggested_clusters_tnfa, spectral_cluster_labels_tnfa, nonrecursive_labels_tnfa = perform_clustering(
    tnfa_modules_filtered, tnfa_modules, sep_lines=False, save_paths=['../figures/tnfa_cooc_eigenvalues.png', '../figures/tnfa_nl_eigenvalues.png', '../figures/tnfa_heatmap.png'], filter_genes=True, recursive=False)

create_algorithm_heatmaps(tnfa_modules, tnfa_ordering, nonrecursive_labels_tnfa, tnfa_cooccurence_matrix, save_path='../figures/tnfa_algorithm_heatmap.png', sep_lines=False)

res = match_clusters_to_modules(spectral_cluster_labels_tnfa, tnfa_ordering, tnfa_modules)
#print(res)
print_cluster_module_matching(res, tnfa_ordering, spectral_cluster_labels_tnfa)

fly_transcriptome_ordering, fly_transcriptome_cooccurence_matrix, suggested_clusters_fly_transcriptome, spectral_cluster_labels_fly_transcriptome, nonrecursive_labels_fly_transcriptome = perform_clustering(
    fly_transcriptome_modules_filtered, fly_transcriptome_modules, sep_lines=False, save_paths=['../figures/fly_transcriptome_cooc_eigenvalues.png', '../figures/fly_transcriptome_nl_eigenvalues.png', '../figures/fly_transcriptome_heatmap.png'], filter_genes=True, recursive=False)

create_algorithm_heatmaps(fly_transcriptome_modules, fly_transcriptome_ordering, nonrecursive_labels_fly_transcriptome, fly_transcriptome_cooccurence_matrix, save_path='../figures/fly_transcriptome_algorithm_heatmap.png', sep_lines=False)

res = match_clusters_to_modules(spectral_cluster_labels_fly_transcriptome, fly_transcriptome_ordering, fly_transcriptome_modules)
print_cluster_module_matching(res, fly_transcriptome_ordering, spectral_cluster_labels_fly_transcriptome)
