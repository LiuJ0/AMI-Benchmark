#!/bin/python3
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple
from earth_movers import *
import math
import functools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def dict_to_symmetric_dataframe(data):
    # Extract unique algorithm names
    algorithms = sorted(set(key for pair in data.keys() for key in pair))
    
    # Create an empty DataFrame
    df = pd.DataFrame(index=algorithms, columns=algorithms)
    
    # Fill the DataFrame
    for (algo1, algo2), value in data.items():
        df.at[algo1, algo2] = value
        df.at[algo2, algo1] = value  # Mirror the value for symmetry
    
    # Fill diagonal with zeros
    np.fill_diagonal(df.values, 0)
    
    return df

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

def compute_similarity_matrix(G: nx.Graph, dataset_modules: Dict[str, List[List[str]]], algos: List[str]) -> pd.DataFrame:
    """
    Compute the similarity matrix for all pairs of algorithms.
    :param G: The complete graph
    :param dataset_modules: Dictionary of algorithms and their modules
    :param algos: List of algorithms
    """
    similarity_df = pd.DataFrame(columns=algos, index=algos)
    
    for algo1 in algos: 
        for algo2 in algos: 
            if algo1 == algo2: 
                continue
            print(f"Computing similarity between {algo1} and {algo2}")
            similarity = compute_solution_similarity(G, dataset_modules[algo1], dataset_modules[algo2])
            similarity_df.at[algo1, algo2] = similarity
            print(f"Similarity: {similarity}")
    
    return similarity_df

def compute_solution_similarity(G: nx.Graph, algo1_modules: List[List[int]], 
                                algo2_modules: List[List[int]]) -> float:
    """
    Compute the solution similarity score between two algorithms.
    
    :param G: The complete graph
    :param algo1_modules: List of modules from algorithm 1
    :param algo2_modules: List of modules from algorithm 2
    :return: Solution similarity score
    """
    # Compute EMD between all pairs of modules
    distance_matrix = np.zeros((len(algo1_modules), len(algo2_modules)))
    for i, mod1 in enumerate(algo1_modules):
        for j, mod2 in enumerate(algo2_modules):
            distance_matrix[i, j] = emd_subgraph_similarity(G, mod1, mod2)


    # Use Hungarian algorithm to find optimal matching
    try:
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
    except ValueError: 
        print(distance_matrix)
        return np.inf
    
    # Compute total distance (sum of EMDs for matched modules)
    total_distance = distance_matrix[row_ind, col_ind].sum()
    
    # Compute similarity score (inverse of total distance)
    similarity_score = 1 / (1 + total_distance)
    
    return similarity_score

def compare_all_algorithms(G: nx.Graph, algorithm_modules: Dict[str, List[List[str]]], method, total_modules) -> Dict[Tuple[str, str], float,]:
    """
    Compare all pairs of algorithms and compute their solution similarity scores.
    
    :param G: The complete graph
    :param algorithm_modules: Dictionary of algorithms and their modules
    :return: Dictionary of pairwise solution similarity scores
    """
    algorithms = list(algorithm_modules.keys())
    similarity_scores = {}
    
    for i in range(len(algorithms)):
        for j in range(i+1, len(algorithms)):
            algo1, algo2 = algorithms[i], algorithms[j]
            score = method(G, algorithm_modules[algo1], algorithm_modules[algo2], total_modules)
            similarity_scores[(algo1, algo2)] = score
    
    return similarity_scores

def compute_similarity_non_matching(G: nx.Graph, algo1_modules: List[List[str]], 
                                algo2_modules: List[List[str]], total_modules) -> float:
    """
    Compute the solution similarity score between two algorithms without matching.

    Suitable when |algo1_modules| >> |algo2_modules|. 

    :param G: The complete graph
    :param algo1_modules: List of modules from algorithm 1
    :param algo2_modules: List of modules from algorithm 2
    :return: Solution similarity score
    """
    # Compute EMD between all pairs of modules
    distance_matrix = np.zeros((len(algo1_modules), len(algo2_modules)))
    for i, mod1 in enumerate(algo1_modules):
        for j, mod2 in enumerate(algo2_modules):
            print("Computing EMD between", mod1, mod2)
            distance_matrix[i, j] = emd_subgraph_similarity(G, mod1, mod2)
            print("EMD:", distance_matrix[i, j])
    left_sum = 0
    for i in range(len(algo1_modules)):
        left_sum += np.min(distance_matrix[i])
    right_sum = 0
    for j in range(len(algo2_modules)):
        right_sum += np.min(distance_matrix[:,j])
    distance = (left_sum + right_sum) / 2
    return 1 / (1 + distance)
    
def compute_similarity_concordance(G: nx.Graph, algo1_modules: List[List[str]], algo2_modules: List[List[str]], 
                                   total_modules: List[List[str]], distance_matrix=None) -> float: 
    """
    :param G: The complete graph
    :param algo1_modules: List of modules from algorithm 1
    :param algo2_modules: List of modules from algorithm 2
    :param total_modules: List of all modules
    :return: Solution similarity score
    """
    def belong_same_module(mod1, mod2, algo1_modules, algo2_modules):
        return any(mod1 in x and mod2 in x for x in algo1_modules) and any(mod1 in x and mod2 in x for x in algo2_modules)
    sum = 0
    total_genes = [gene for module in total_modules for gene in module]
    missing_genes = set(total_genes) - set([gene for module in algo1_modules for gene in module]) - set([gene for module in algo2_modules for gene in module])
    algo1_modules.append(list(missing_genes))
    algo2_modules.append(list(missing_genes))
    total_sum = 0
    for gene1 in total_genes:
        for gene2 in total_genes:
            if gene1 == gene2: 
                continue
            if belong_same_module(gene1, gene2, algo1_modules, algo2_modules): 
                sum += 1/(nx.shortest_path_length(G, source=gene1, target=gene2))
            total_sum += 1/(nx.shortest_path_length(G, source=gene1, target=gene2))
    return sum / total_sum


def calculate_distance_matrix(G, total_modules): 
    distance_matrix = np.zeros((len(total_modules), len(total_modules)))
    for i, mod1 in enumerate(total_modules):
        for j, mod2 in enumerate(total_modules):
            distance_matrix[i, j] = emd_subgraph_similarity(G, mod1, mod2)
    return distance_matrix


def reorder_matrix(df, new_order=['PAPER', 'DOMINO', 'HotNet2', 'FDRnet'], output_file=None):
    """
    Reorder a matrix stored in a CSV file.
    
    Parameters:
    file_path (str): Path to the input CSV file
    new_order (list): List of column names in the desired order
    output_file (str, optional): Path to save the reordered CSV file. If None, doesn't save to file.
    
    Returns:
    pd.DataFrame: Reordered pandas DataFrame
    """
    # Read the CSV file
    
    # Reorder the rows and columns
    df_reordered = df.reindex(index=new_order, columns=new_order)
    return df_reordered

def create_combined_heatmap(matrix1_path, matrix2_path, output_path='combined_heatmap.png', 
                          matrix1_name='Matrix 1', matrix2_name='Matrix 2', dataset_name='Dataset',
                          annot_size=16, axis_label_size=14, title_size=18, colorbar_size=14):
    """
    Create a heatmap combining two symmetric matrices, where the upper triangle shows matrix1
    and the lower triangle shows matrix2, with white on the diagonal.
    
    Parameters:
    matrix1_path (str): Path to first CSV file (will be shown in upper triangle)
    matrix2_path (str): Path to second CSV file (will be shown in lower triangle)
    output_path (str): Path where the output image will be saved
    annot_size (int): Font size for the numbers in the heatmap
    axis_label_size (int): Font size for axis labels
    title_size (int): Font size for the title
    colorbar_size (int): Font size for the colorbar label
    """
    # Read both matrices
    df1 = pd.read_csv(matrix1_path, index_col=0)
    df2 = pd.read_csv(matrix2_path, index_col=0)
    
    # Verify that both matrices have the same dimensions and labels
    if not (df1.index.equals(df1.columns) and 
            df2.index.equals(df2.columns) and 
            df1.index.equals(df2.index)):
        raise ValueError("Matrices must be square and have matching labels")
    
    # Convert DataFrames to numpy arrays
    matrix1 = df1.to_numpy()
    matrix2 = df2.to_numpy()
    
    # Create the combined matrix
    combined_matrix = np.zeros_like(matrix1)
    
    # Fill upper triangle (excluding diagonal) with matrix1
    mask_upper = np.triu_indices(len(matrix1), k=1)  # k=1 excludes diagonal
    combined_matrix[mask_upper] = matrix1[mask_upper]
    
    # Fill lower triangle with matrix2
    mask_lower = np.tril_indices(len(matrix2), k=-1)
    combined_matrix[mask_lower] = matrix2[mask_lower]
    
    # Fill diagonal with np.nan to make it white
    np.fill_diagonal(combined_matrix, np.nan)
    
    # Convert back to DataFrame to preserve labels
    combined_df = pd.DataFrame(combined_matrix, 
                             index=df1.index, 
                             columns=df1.columns)
    
    # Create the figure
    plt.figure(figsize=(12, 10))
    
    # Set font sizes for all text elements
    plt.rcParams.update({'font.size': axis_label_size})
    
    # Create the heatmap
    heatmap = sns.heatmap(combined_df, 
                         annot=True,
                         fmt='.3f',
                         cmap='YlGnBu',
                         vmin=0,
                         vmax=1,
                         annot_kws={'size': annot_size},
                         cbar_kws={'label': 'Similarity Score'})
    
    # Customize the plot with specific font sizes
    plt.title(f'{dataset_name} Similarity Matrices\nUpper: {matrix1_name}, Lower: {matrix2_name}',
             fontsize=title_size, pad=20)
    
    # Set font sizes for axis labels
    plt.xticks(rotation=45, ha='right', fontsize=axis_label_size)
    plt.yticks(fontsize=axis_label_size)
    
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_ylabel('Similarity Score', fontsize=colorbar_size)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def mean_similarity(matrix):
    df = pd.read_csv(matrix, index_col=0)
    return df.mean().mean()

if __name__ == '__main__': 
    name2ENSG = pd.read_csv("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/original/Name2ENSG_GRCh38.csv")
    name2ENSG_fb = pd.read_csv("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/original/name2ENSG_fb.tsv", sep='\t')

    three_dbs = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/processed/network_full_ids_v2.tsv", source="node1", target="node2")
    dip = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/original/dip_original.sif", source="node1", target="node2")
    string_db = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/processed/STRING_v12_FBgn_geq900.sif", source="node1", target="node2")

    algos = ["PAPER", "DOMINO", "HotNet2", "FDRnet"]
    res_dir = "/lab01/Projects/Jason_Projects/aneuploidy/ppi/test/test_07-21-2024_EMP-benchmark/true_solutions"
    modules_report_dir = "/lab01/Projects/Jason_Projects/aneuploidy/ppi/test/test_07-21-2024_EMP-benchmark/report/module_cache_files"

    illumina_modules = get_all_modules(res_dir, algos, "gene_scores_illumina_v3")
    tvc_modules = get_all_modules(res_dir, algos, "gene_scores_tvc")
    tnfa_modules = get_all_modules(res_dir, algos, "tnfa")
    fly_transcriptome_modules = get_all_modules(res_dir, algos, "fly_transcriptome")

    total_illumina = []; total_tvc = []; total_tnfa = []; total_fly = []

    for algo in algos: total_illumina += illumina_modules[algo]
    for algo in algos: total_tvc += tvc_modules[algo]
    for algo in algos: total_tnfa += tnfa_modules[algo]
    for algo in algos: total_fly += fly_transcriptome_modules[algo]
    
    #illumina_matrix_nonmatching = compare_all_algorithms(three_dbs, illumina_modules, compute_similarity_non_matching, total_illumina, illumina_distance_matrix)
    #dict_to_symmetric_dataframe(illumina_matrix_nonmatching).to_csv("../similarity_scores/illumina_matrix_nonmatching.csv")
    #illumina_matrix_concordance = compare_all_algorithms(three_dbs, illumina_modules, compute_similarity_concordance, total_illumina)
    #reorder_matrix(dict_to_symmetric_dataframe(illumina_matrix_concordance)).to_csv("../similarity_scores/illumina_matrix_concordance.csv")

    #tvc_matrix_nonmatching = compare_all_algorithms(three_dbs, tvc_modules, compute_similarity_non_matching, total_tvc, tvc_distance_matrix)
    #tvc_matrix_concordance = compare_all_algorithms(three_dbs, tvc_modules, compute_similarity_concordance, total_tvc)
    #dict_to_symmetric_dataframe(tvc_matrix_nonmatching).to_csv("../similarity_scores/tvc_matrix_nonmatching.csv")
    #reorder_matrix(dict_to_symmetric_dataframe(tvc_matrix_concordance)).to_csv("../similarity_scores/tvc_matrix_concordance.csv")

    #tnfa_matrix_nonmatching = compare_all_algorithms(dip, tnfa_modules, compute_similarity_non_matching, total_tnfa, tnfa_distance_matrix)
    #tnfa_matrix_concordance = compare_all_algorithms(dip, tnfa_modules, compute_similarity_concordance, total_tnfa)
    #dict_to_symmetric_dataframe(tnfa_matrix_nonmatching).to_csv("../similarity_scores/tnfa_matrix_nonmatching.csv")
    #reorder_matrix(dict_to_symmetric_dataframe(tnfa_matrix_concordance)).to_csv("../similarity_scores/tnfa_matrix_concordance.csv")

    #fly_matrix_nonmatching = compare_all_algorithms(string_db, fly_transcriptome_modules, compute_similarity_non_matching, total_fly, fly_distance_matrix)
    #fly_matrix_concordance = compare_all_algorithms(string_db, fly_transcriptome_modules, compute_similarity_concordance, total_fly)
    #dict_to_symmetric_dataframe(fly_matrix_nonmatching).to_csv("../similarity_scores/fly_matrix_nonmatching.csv")
    #reorder_matrix(dict_to_symmetric_dataframe(fly_matrix_concordance)).to_csv("../similarity_scores/fly_matrix_concordance.csv")

    #create_combined_heatmap("../similarity_scores/illumina_similarity.csv", "../similarity_scores/illumina_matrix_nonmatching_reordered.csv", output_path="../similarity_scores/illumina_combined.png", matrix1_name="matching", matrix2_name="sum of EMDs", dataset_name="Illumina")
    #create_combined_heatmap("../similarity_scores/tvc_similarity.csv", "../similarity_scores/tvc_matrix_nonmatching_reordered.csv", output_path="../similarity_scores/tvc_combined.png", matrix1_name="matching", matrix2_name="sum of EMDs", dataset_name="TVC")
    #create_combined_heatmap("../similarity_scores/tvc_similarity.csv", "../similarity_scores/tvc_matrix_nonmatching_reordered.csv", output_path="../similarity_scores/tvc_combined.svg", matrix1_name="matching", matrix2_name="sum of EMDs", dataset_name="TVC", axis_label_size=18)
    #create_combined_heatmap("../similarity_scores/tnfa_similarity.csv", "../similarity_scores/tnfa_matrix_nonmatching_reordered.csv", output_path="../similarity_scores/tnfa_combined.png", matrix1_name="matching", matrix2_name="sum of EMDs", dataset_name="TNFA")
    #create_combined_heatmap("../similarity_scores/fly_transcriptome_similarity.csv", "../similarity_scores/fly_matrix_nonmatching_reordered.csv", output_path="../similarity_scores/fly_combined.png", matrix1_name="matching", matrix2_name="sum of EMDs", dataset_name="Fly Transcriptome")
    paths1 = ["../similarity_scores/illumina_similarity.csv", "../similarity_scores/tvc_similarity.csv", "../similarity_scores/tnfa_similarity.csv", "../similarity_scores/fly_transcriptome_similarity.csv"]
    paths2 = ["../similarity_scores/illumina_matrix_nonmatching.csv", "../similarity_scores/tvc_matrix_nonmatching.csv", "../similarity_scores/tnfa_matrix_nonmatching.csv", "../similarity_scores/fly_matrix_nonmatching.csv"]
    print(sum([mean_similarity(path) for path in paths1])/len(paths1))
    print(sum([mean_similarity(path) for path in paths2])/len(paths2))
