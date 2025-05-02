import numpy as np
from utils import *
from collections import defaultdict
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from typing import List, Dict
import seaborn as sns
from collections import defaultdict

def generate_cooccurrence_matrix(module_sets):
    """
    Generate a gene-gene co-occurrence matrix based on multiple sets of modules.
    
    :param module_sets: A list of lists, where each inner list represents the modules
                        identified by one algorithm. Each module is a set of gene IDs.
    :return: A dictionary representing the co-occurrence matrix, where keys are
             tuples of gene pairs and values are their co-occurrence counts.
    """
    # Create a set of all unique genes
    all_genes = []
    for modules in module_sets:
        for module in modules:
            all_genes.append(module)
    
    # Convert the set to a sorted list for consistent indexing
    gene_list = sorted(list(set(all_genes)))
    
    # Create a dictionary to store co-occurrences
    cooccurrence = defaultdict(int)
    
    # Count co-occurrences
    for module in module_sets:
        for i, gene1 in enumerate(module):
            for gene2 in list(module)[i+1:]:
                if gene1 != gene2:
                    pair = tuple(sorted([gene1, gene2]))
                    cooccurrence[pair] += 1
    
    return cooccurrence, gene_list

def cooccurrence_to_matrix(cooccurrence, gene_list, normalize=False):
    """
    Convert the co-occurrence dictionary to a numpy matrix.
    
    :param cooccurrence: The co-occurrence dictionary
    :param gene_list: The list of all genes
    :return: A numpy matrix of co-occurrences
    """
    n = len(gene_list)
    matrix = np.zeros((n, n), dtype=int)
    
    for (gene1, gene2), count in cooccurrence.items():
        i, j = gene_list.index(gene1), gene_list.index(gene2)
        matrix[i, j] = matrix[j, i] = count
        if normalize: 
            matrix[i, j] = matrix[j, i] = min(count, 1)

    # assert symmetric matrix
    assert np.allclose(matrix, matrix.T)

    return matrix


def perform_spectral_clustering(matrix, n_clusters=2):
    """
    Perform spectral clustering on the co-occurrence matrix.
    
    :param matrix: The co-occurrence matrix
    :param n_clusters: Number of clusters to form
    :return: Cluster labels for each gene
    """
    # Normalize the matrix
    normalized_matrix = matrix # normalize(matrix, norm='l1', axis=1)
    
    # Perform spectral clustering
    clustering = SpectralClustering(n_clusters=n_clusters, 
                                    assign_labels='kmeans',
                                    random_state=0,
                                    affinity='precomputed').fit(normalized_matrix)
    
    return clustering.labels_

def plot_eigenvalues(matrix, save_path=None):
    """
    Plot the eigenvalues of the co-occurence matrix. 
    """
    eigenvalues = np.sort(np.linalg.eigvals(matrix))[::-1]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues of Co-occurrence Matrix')
    plt.show()

    if save_path:
        plt.savefig(save_path)


def estimate_clusters_eigenvalues(matrix, max_clusters, save_path=False):
    """
    Estimate the number of clusters using the eigengap heuristic.
    
    :param matrix: The normalized co-occurrence matrix
    :param max_clusters: Maximum number of clusters to consider
    :return: Suggested number of clusters
    """
    # Compute the normalized Laplacian
    D = np.diag(np.sum(matrix, axis=1) + 1e-6)  # Add small value to avoid division by zero
    # print(D)
    L = D - matrix
    L_norm = np.dot(np.dot(np.linalg.inv(np.sqrt(D)), L), np.linalg.inv(np.sqrt(D)))
    
    # Compute eigenvalues
    eigenvalues = np.sort(np.linalg.eigvals(L_norm))
    
    # Compute eigengaps
    eigengaps = np.diff(eigenvalues)
    
    # Plot eigenvalues
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
    plt.xlabel('Eigenvalue Index', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    names_to_titles = {
        "../figures/illumina_nl_eigenvalues.png": "Aneuploidy1",
        "../figures/tvc_nl_eigenvalues.png": "Aneuploidy2",
        "../figures/tnfa_nl_eigenvalues.png": "TNFa",
        "../figures/fly_transcriptome_nl_eigenvalues.png": "Fly Transcriptome",
    }

    plt.title(names_to_titles[save_path], fontsize=16)
    plt.yticks([])
    plt.xticks(fontsize=12)
    plt.tight_layout()

    plt.show()

    if save_path:
        plt.savefig(save_path)
    
    # Find the largest eigengap
    suggested_clusters = np.argmax(eigengaps[1:max_clusters]) + 1
    
    return suggested_clusters

def print_clustering_interpretation(interpretation, gene_list, cluster_labels):
    """
    Print a detailed interpretation of the clustering results.
    
    :param interpretation: The output from interpret_clustering_results
    :param gene_list: The list of all genes
    :param cluster_labels: The cluster labels for each gene
    """
    n_clusters = max(cluster_labels) + 1
    
    print("\nClustering Interpretation:")
    print("==========================")
    
    # Print cluster compositions
    for cluster in range(n_clusters):
        print(f"\nCluster {cluster}:")
        genes_in_cluster = [gene for gene, label in zip(gene_list, cluster_labels) if label == cluster]
        print(f"  Genes: {', '.join(genes_in_cluster)}")
        print(f"  Number of genes: {len(genes_in_cluster)}")
        print(f"  Purity: {interpretation['cluster_purities'][cluster]:.2f}")
        
        algorithm_counts = defaultdict(int)
        for algo, _, _ in interpretation['cluster_compositions'][cluster]:
            algorithm_counts[algo] += 1
        
        print("  Composition:")
        for algo, count in algorithm_counts.items():
            print(f"    Algorithm {algo}: {count} genes")
    
    # Print module distribution statistics
    print("\nModule Distribution Across Clusters:")
    for algo_index, distribution in enumerate(interpretation['module_distributions']):
        print(f"  Module {algo_index}: {distribution}")
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"  Average Cluster Purity: {np.mean(interpretation['cluster_purities']):.2f}")
    print(f"  Average Module Homogeneity: {np.mean(interpretation['module_homogeneities']):.2f}")


def remove_redundancies(modules: List[List[str]]) -> List[List[str]]: 
    module_sets = [set(module) for module in modules]
    non_redundant_modules = []
    for i, module in enumerate(module_sets):
        is_redundant = False
        for j, other_module in enumerate(module_sets):
            if i != j and module.issubset(other_module):
                is_redundant = True
                break
        if not is_redundant:
            non_redundant_modules.append(list(module))
    return non_redundant_modules

def match_clusters_to_modules(cluster_labels, gene_list, algorithm_modules):
    """
    Match clusters to original modules and present the results.
    
    :param cluster_labels: The cluster labels for each gene
    :param gene_list: The list of all genes
    :param algorithm_modules: A dictionary where keys are algorithm names and values are lists of modules
    :return: A dictionary containing the matching results
    """
    n_clusters = max(cluster_labels) + 1
    gene_to_cluster = dict(zip(gene_list, cluster_labels))
    
    # Initialize results dictionary
    results = {
        'cluster_compositions': defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
        'module_distributions': defaultdict(lambda: defaultdict(int)),
        'unmatched_genes': set(gene_list),
        'unmerged_modules': defaultdict(list),
        'gene_to_modules': defaultdict(set)  # Track which modules each gene belongs to
    }
    
    # Match genes to modules and clusters
    for algo_name, modules in algorithm_modules.items():
        for module_index, module in enumerate(modules):
            for gene in module:
                if gene in gene_to_cluster:
                    cluster = gene_to_cluster[gene]
                    results['cluster_compositions'][cluster][algo_name][module_index] += 1
                    results['module_distributions'][algo_name][cluster] += 1
                    results['unmatched_genes'].discard(gene)
                    # Track which modules the gene belongs to
                    results['gene_to_modules'][gene].add(f"{algo_name}[{module_index}]")
    
    # Identify unmerged modules (clusters with only one module)
    for cluster, algos in results['cluster_compositions'].items():
        if len(algos) == 1:  # Only one algorithm in this cluster
            algo_name = list(algos.keys())[0]
            if len(algos[algo_name]) == 1:  # Only one module from this algorithm
                module_index = list(algos[algo_name].keys())[0]
                results['unmerged_modules'][algo_name].append(module_index)
    
    return results

def print_cluster_module_matching(results, gene_list, cluster_labels):
    """
    Print the results of cluster-module matching in a readable format.
    
    :param results: The output from match_clusters_to_modules
    :param gene_list: The list of all genes
    :param cluster_labels: The cluster labels for each gene
    """
    n_clusters = max(cluster_labels) + 1
    gene_to_cluster = dict(zip(gene_list, cluster_labels))
    
    print("\nGene to Algorithm-Module Mappings by Cluster:")
    print("===========================================")

    n_pure_modules = 0
    
    for cluster in range(n_clusters):
        print(f"\nCluster {cluster}: ", end="")
        
        # Get genes in this cluster
        cluster_genes = [gene for gene, label in gene_to_cluster.items() if label == cluster]

        used_modules = set()
        
        # Sort genes alphabetically for consistent output
        for gene in sorted(cluster_genes):
            # Get the algorithms and modules this gene belongs to
            algo_modules = results['gene_to_modules'].get(gene, set())
            if algo_modules:
                # Sort the algorithm-module combinations for consistent output
                for algo_module_str in sorted(algo_modules):
                    used_modules.add(algo_module_str)
                # print(f"{gene},{algo_modules_str}")
        for algo_module_str in sorted(used_modules):
            print(algo_module_str, end=", ")
        if len(used_modules) == 1:
            n_pure_modules += 1
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("==================")
    print(f"Total number of clusters: {n_clusters}")
    print(f"Total number of genes: {len(gene_list)}")
    print(f"Number of pure modules: {n_pure_modules}")
    
    if results['unmatched_genes']:
        print("\nUnmatched Genes:")
        print(', '.join(sorted(results['unmatched_genes'])))
    else:
        print("\nAll genes were matched to modules.")
        
def remove_redundancies_from_dict(algorithm_outputs: Dict[str, List[List[str]]]) -> Dict[str, List[List[str]]]:
    # Convert all modules to sets for easier comparison
    algorithm_sets = {
        algo: [set(module) for module in modules]
        for algo, modules in algorithm_outputs.items()
    }
    
    # Dictionary to store non-redundant modules
    non_redundant = {algo: [] for algo in algorithm_outputs}
    
    # Flatten all modules into a single list with their algorithm names
    all_modules = [
        (algo, module_set) 
        for algo, module_sets in algorithm_sets.items() 
        for module_set in module_sets
    ]
    
    # Check each module against all others
    for i, (algo, module) in enumerate(all_modules):
        is_redundant = False
        for j, (other_algo, other_module) in enumerate(all_modules):
            if i != j and module.issubset(other_module):
                is_redundant = True
                break
        if not is_redundant:
            # Convert set back to sorted list for consistency
            non_redundant[algo].append(sorted(module))
    
    return non_redundant

def create_cooccurrence_heatmaps(matrix, gene_list, cluster_labels, save_path=None, show_labels=False, sep_lines=True):
    """
    Create heatmaps of the co-occurrence matrix before and after clustering.
    
    :param matrix: The co-occurrence matrix
    :param gene_list: List of gene names
    :param cluster_labels: Cluster labels for each gene
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    if not show_labels:
        xticklabels = yticklabels = False
    else:
        xticklabels = yticklabels = gene_list
    
    # Heatmap before clustering
    sns.heatmap(matrix, ax=ax1, cmap="YlGnBu", xticklabels=xticklabels, yticklabels=yticklabels)
    ax1.set_title("Co-occurrence Matrix Before Clustering")
    ax1.set_xlabel("Genes")
    ax1.set_ylabel("Genes")
    
    # Sort genes by cluster labels
    sorted_indices = np.argsort(cluster_labels)
    sorted_matrix = matrix[sorted_indices][:, sorted_indices]
    sorted_gene_list = [gene_list[i] for i in sorted_indices]
    
    # Heatmap after clustering
    sns.heatmap(sorted_matrix, ax=ax2, cmap="YlGnBu", xticklabels=xticklabels, yticklabels=yticklabels)
    ax2.set_title("Co-occurrence Matrix After Clustering")
    ax2.set_xlabel("Genes")
    ax2.set_ylabel("Genes")
    
    # Add cluster separation lines
    if sep_lines:
        cluster_sizes = [sum(cluster_labels == i) for i in range(max(cluster_labels) + 1)]
        cumulative_sizes = np.cumsum(cluster_sizes)
        for size in cumulative_sizes[:-1]:
            ax2.axhline(y=size, color='red', linestyle='--')
            ax2.axvline(x=size, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.show()

    if save_path:
        plt.savefig(save_path)


def filter_consensus_genes(algorithm_modules):
    """
    Filter out genes that are present in only one algorithm's modules.
    
    :param algorithm_modules: A list of lists of gene IDs for each algorithm
    :return: A tuple containing the filtered algorithm_modules and a list of removed genes
    """
    # Count occurrences of each gene across all algorithms
    gene_counts = defaultdict(set)
    for algo, modules in algorithm_modules.items():
        for module in modules:
            for gene in module:
                gene_counts[gene].add(algo)
    
    # Identify genes to keep (present in more than one algorithm)
    genes_to_keep = {gene for gene, algos in gene_counts.items() if len(algos) > 1}
    
    # Filter modules
    filtered_modules = {}
    for algo, modules in algorithm_modules.items():
        filtered_modules[algo] = [set(gene for gene in module if gene in genes_to_keep) for module in modules]
        # Remove any empty modules
        filtered_modules[algo] = [module for module in filtered_modules[algo] if module]
    
    # Identify removed genes
    removed_genes = set(gene_counts.keys()) - genes_to_keep
    
    return filtered_modules, list(removed_genes)

def filter_non_overlapping_modules(algorithm_modules):
    """
    Filter out modules that have no overlap with any other modules across all algorithms.
    
    :param algorithm_modules: A dictionary where keys are algorithm names and values are lists of modules (sets of gene IDs)
    :return: A tuple containing the filtered algorithm_modules and a list of removed modules
    """
    # Create a dictionary to store all genes and the modules they appear in
    gene_to_modules = defaultdict(list)
    
    # Populate gene_to_modules
    for algo, modules in algorithm_modules.items():
        for i, module in enumerate(modules):
            for gene in module:
                gene_to_modules[gene].append((algo, i))
    
    # Identify modules to keep (those with at least one overlapping gene)
    modules_to_keep = set()
    for gene, appearances in gene_to_modules.items():
        if len(appearances) > 1:  # If a gene appears in more than one module
            for algo, module_index in appearances:
                modules_to_keep.add((algo, module_index))
    
    # Filter modules
    filtered_modules = {}
    removed_modules = []
    for algo, modules in algorithm_modules.items():
        filtered_modules[algo] = []
        for i, module in enumerate(modules):
            if (algo, i) in modules_to_keep:
                filtered_modules[algo].append(module)
            else:
                removed_modules.append((algo, module))
    
    return filtered_modules, removed_modules

def create_algorithm_heatmaps(algorithm_modules, joint_gene_list, cluster_labels, joint_cooccurrence_matrix, y_axis=None, save_path=None, sep_lines=True):
    """
    Create heatmaps for each algorithm's co-occurrence matrix, aligned with the joint clustering.
    
    :param algorithm_modules: A dictionary where keys are algorithm names and values are lists of modules
    :param joint_gene_list: The list of all genes from the joint analysis
    :param cluster_labels: Cluster labels for each gene in joint_gene_list
    :param save_path: Path to save the figure (optional)
    """
    # Sort genes by cluster labels
    sorted_indices = np.argsort(cluster_labels)
    sorted_gene_list = [joint_gene_list[i] for i in sorted_indices]
    
    # Create a figure with subplots for each algorithm plus the joint matrix
    n_algorithms = len(algorithm_modules)
    fig, axes = plt.subplots(1, n_algorithms+1, figsize=(6*(n_algorithms+1), 5))
    
    if n_algorithms == 1:
        axes = [axes]
    for ax, (algo_name, modules) in zip(axes[:-1], algorithm_modules.items()):
        # Generate co-occurrence matrix for this algorithm
        algo_cooccurrence, algo_genes = generate_cooccurrence_matrix(modules)
        algo_matrix = cooccurrence_to_matrix(algo_cooccurrence, algo_genes, normalize=True)
        
        # Create a full-size matrix with all genes
        full_matrix = np.zeros((len(joint_gene_list), len(joint_gene_list)))
        
        # Create a mapping from joint_gene_list to their indices
        joint_gene_indices = {gene: i for i, gene in enumerate(joint_gene_list)}
        
        # Fill in the values from the algorithm's matrix
        for i, gene1 in enumerate(algo_genes):
            for j, gene2 in enumerate(algo_genes):
                if gene1 in joint_gene_indices and gene2 in joint_gene_indices:
                    ii, jj = joint_gene_indices[gene1], joint_gene_indices[gene2]
                    full_matrix[ii, jj] = algo_matrix[i, j]
        
        # Reorder the matrix based on the clustering
        sorted_matrix = full_matrix[sorted_indices][:, sorted_indices]
        
        # Create heatmap
        cbar = sns.heatmap(sorted_matrix, ax=ax, cmap="YlGnBu", xticklabels=False, yticklabels=False)
        cbar.collections[0].colorbar.ax.tick_params(labelsize=16)
        ax.set_title(f"{algo_name}", fontsize=18)
        if sep_lines:
            cluster_sizes = [sum(cluster_labels == i) for i in range(max(cluster_labels) + 1)]
            cumulative_sizes = np.cumsum(cluster_sizes)
            for size in cumulative_sizes[:-1]:
                ax.axhline(y=size, color='red', linestyle='--')
                ax.axvline(x=size, color='red', linestyle='--')

    # Add joint matrix
    sorted_joint_matrix = joint_cooccurrence_matrix[sorted_indices][:, sorted_indices]
    ax = axes[-1]
    cbar = sns.heatmap(sorted_joint_matrix, ax=ax, cmap="YlGnBu", xticklabels=False, yticklabels=False)
    cbar.collections[0].colorbar.ax.tick_params(labelsize=18)
    ax.set_title("Joint", fontsize=16)
    if sep_lines:
        cluster_sizes = [sum(cluster_labels == i) for i in range(max(cluster_labels) + 1)]
        cumulative_sizes = np.cumsum(cluster_sizes)
        for size in cumulative_sizes[:-1]:
            ax.axhline(y=size, color='red', linestyle='--')
            ax.axvline(x=size, color='red', linestyle='--')
    # add y-axis label
    axes[0].set_ylabel(y_axis, fontsize=16)
    plt.tight_layout()

    plt.show()

    if save_path: 
        plt.savefig(save_path, dpi=500)

def recursive_spectral_clustering(matrix, max_cluster_size, max_clusters, max_iterations=5):
    """
    Perform recursive spectral clustering on the co-occurrence matrix.
    
    :param matrix: The co-occurrence matrix
    :param max_cluster_size: Maximum allowed size for a cluster
    :param max_clusters: Maximum number of clusters to form in each iteration
    :param max_iterations: Maximum number of recursive iterations
    :return: Tuple containing (final cluster labels, original cluster labels)
    """
    def cluster_recursively(submatrix, current_indices, depth=0):
        print(f"Depth: {depth} | Cluster size: {len(submatrix)}")
        if depth >= max_iterations or len(submatrix) <= max_cluster_size:
            return [current_indices]
        
        # Perform spectral clustering
        n_clusters = estimate_clusters_eigenvalues(submatrix, max_clusters)
        clustering = SpectralClustering(n_clusters=n_clusters, 
                                        assign_labels='kmeans',
                                        random_state=0,
                                        affinity='precomputed').fit(submatrix)
        
        labels = clustering.labels_
        
        # If this is the first level, save the original clustering
        if depth == 0:
            nonlocal original_labels
            original_labels = labels.copy()
        
        # Sort clusters by their mean position in the original matrix
        cluster_order = []
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            mean_position = np.mean(current_indices[cluster_indices])
            cluster_order.append((i, mean_position))
        
        cluster_order.sort(key=lambda x: x[1])
        
        # Recursively cluster large clusters
        final_clusters = []
        for i, _ in cluster_order:
            cluster_indices = np.where(labels == i)[0]
            cluster_size = len(cluster_indices)
            
            if cluster_size > max_cluster_size:
                subcluster_indices = cluster_recursively(
                    submatrix[np.ix_(cluster_indices, cluster_indices)],
                    current_indices[cluster_indices],
                    depth + 1
                )
                final_clusters.extend(subcluster_indices)
            else:
                final_clusters.append(current_indices[cluster_indices])
        
        return final_clusters

    # Initialize original_labels
    original_labels = None

    # Start the recursive clustering process
    all_indices = np.arange(len(matrix))
    final_clusters = cluster_recursively(matrix, all_indices)
    
    # Create final labels
    final_labels = np.zeros(len(matrix), dtype=int)
    for i, cluster in enumerate(final_clusters):
        final_labels[cluster] = i
    
    return final_labels, original_labels

def recursive_spectral_clustering_silhouette(matrix, min_silhouette_score, max_clusters, max_iterations=5):
    """
    Perform recursive spectral clustering on the co-occurrence matrix.
    
    :param matrix: The co-occurrence matrix
    :param min_silhouette_score: Minimum acceptable silhouette score for a cluster
    :param max_clusters: Maximum number of clusters to form in each iteration
    :param max_iterations: Maximum number of recursive iterations
    :return: Tuple containing (final cluster labels, original cluster labels)
    """
    def cluster_recursively(submatrix, current_indices, depth=0):
        print(f"Depth: {depth} | Cluster size: {len(submatrix)}")
        if depth >= max_iterations or len(submatrix) < 2:  # Check if there's only one data point
            return [current_indices]
        
        # Perform spectral clustering
        n_clusters = min(estimate_clusters_eigenvalues(submatrix, max_clusters), len(submatrix))
        clustering = SpectralClustering(n_clusters=n_clusters, 
                                        assign_labels='kmeans',
                                        random_state=0,
                                        affinity='precomputed').fit(submatrix)
        
        labels = clustering.labels_
        
        # If this is the first level, save the original clustering
        if depth == 0:
            nonlocal original_labels
            original_labels = labels.copy()
        
        # Calculate silhouette score
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            sil_score = silhouette_score(submatrix, labels, metric='precomputed')
            print(f"Depth: {depth} | Silhouette Score: {sil_score}")
        else:
            sil_score = 0  # If there's only one cluster, set silhouette score to 0
            print(f"Depth: {depth} | Only one cluster formed, silhouette score set to 0")
        
        # If silhouette score is good enough or only one cluster, return current clustering
        if sil_score >= min_silhouette_score or len(unique_labels) == 1:
            return [current_indices]
        
        # Sort clusters by their mean position in the original matrix
        cluster_order = []
        for i in unique_labels:
            cluster_indices = np.where(labels == i)[0]
            mean_position = np.mean(current_indices[cluster_indices])
            cluster_order.append((i, mean_position))
        
        cluster_order.sort(key=lambda x: x[1])
        
        # Recursively cluster poor quality clusters
        final_clusters = []
        for i, _ in cluster_order:
            cluster_indices = np.where(labels == i)[0]
            
            # Calculate silhouette score for this specific cluster
            print(cluster_indices)
            if len(cluster_indices) > 1:
                cluster_submatrix = submatrix[np.ix_(cluster_indices, cluster_indices)]
                if len(np.unique(cluster_submatrix)) > 1:  # Check if submatrix has more than one unique value
                    cluster_sil_score = silhouette_score(cluster_submatrix,
                                                         np.zeros(len(cluster_indices)),
                                                         metric='precomputed')
                else:
                    cluster_sil_score = 0
            else:
                cluster_sil_score = 0
            
            if cluster_sil_score < min_silhouette_score and len(cluster_indices) > 1:
                subcluster_indices = cluster_recursively(
                    submatrix[np.ix_(cluster_indices, cluster_indices)],
                    current_indices[cluster_indices],
                    depth + 1
                )
                final_clusters.extend(subcluster_indices)
            else:
                final_clusters.append(current_indices[cluster_indices])
        
        return final_clusters

    # Initialize original_labels
    original_labels = None

    # Start the recursive clustering process
    all_indices = np.arange(len(matrix))
    final_clusters = cluster_recursively(matrix, all_indices)
    
    # Create final labels
    final_labels = np.zeros(len(matrix), dtype=int)
    for i, cluster in enumerate(final_clusters):
        final_labels[cluster] = i
    
    return final_labels, original_labels

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
    normalized_matrix = matrix 
    plot_eigenvalues(normalized_matrix, save_path=save_paths[0])
    # Illumina: 7 clusters, Tvc: 5, TnfA: 6, Fly transcriptome: 10
    suggested_clusters_eigen = estimate_clusters_eigenvalues(normalized_matrix, save_path=save_paths[1], max_clusters=len(filtered_unlabeled)) # number of initial modules
    print(f"\nSuggested number of clusters (eigengap heuristic): {suggested_clusters_eigen}")
    if recursive:
        spectral_cluster_labels, initial = recursive_spectral_clustering(normalized_matrix, max_cluster_size=40, max_clusters=len(filtered_unlabeled) // 2, max_iterations=5)
    else:
        spectral_cluster_labels = perform_spectral_clustering(normalized_matrix, suggested_clusters_eigen)
        initial = spectral_cluster_labels
    create_cooccurrence_heatmaps(matrix, gene_list, spectral_cluster_labels, save_path=save_paths[2], sep_lines=sep_lines)
    return gene_list, matrix, suggested_clusters_eigen, spectral_cluster_labels, initial


if __name__ == "__main__":
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

    three_dbs = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/processed/network_full_ids_v2.tsv", source="node1", target="node2")
    dip = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/original/dip_original.sif", source="node1", target="node2")
    string_db = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/processed/STRING_v12_FBgn_geq900.sif", source="node1", target="node2")


    illumina_ordering, illumina_cooccurence_matrix, suggested_clusters_illumina, spectral_cluster_labels_illumina, nonrecursive_labels_illumina = perform_clustering(
        illumina_modules_filtered, illumina_modules, sep_lines=False, save_paths=['../figures/illumina_cooc_eigenvalues.png', '../figures/illumina_nl_eigenvalues.png', '../figures/illumina_heatmap.png'], filter_genes=True, recursive=False)
    create_algorithm_heatmaps(illumina_modules, illumina_ordering, nonrecursive_labels_illumina, illumina_cooccurence_matrix, save_path='../figures/illumina_algorithm_heatmap.png', sep_lines=False, y_axis='Aneuploidy1')
    res = match_clusters_to_modules(spectral_cluster_labels_illumina, illumina_ordering, illumina_modules)
    print_cluster_module_matching(res, illumina_ordering, spectral_cluster_labels_illumina)


    tvc_ordering, tvc_cooccurence_matrix, suggested_clusters_tvc, spectral_cluster_labels_tvc, nonrecursive_labels_tvc = perform_clustering(
        tvc_modules_filtered, tvc_modules, sep_lines=False, save_paths=['../figures/tvc_cooc_eigenvalues.png', '../figures/tvc_nl_eigenvalues.png', '../figures/tvc_heatmap.png'], filter_genes=True, recursive=False)
    create_algorithm_heatmaps(tvc_modules, tvc_ordering, nonrecursive_labels_tvc, tvc_cooccurence_matrix, save_path='../figures/tvc_algorithm_heatmap.png', sep_lines=False, y_axis='Aneuploidy2')
    res = match_clusters_to_modules(spectral_cluster_labels_tvc, tvc_ordering, tvc_modules)
    print_cluster_module_matching(res, tvc_ordering, spectral_cluster_labels_tvc)

    tnfa_ordering, tnfa_cooccurence_matrix, suggested_clusters_tnfa, spectral_cluster_labels_tnfa, nonrecursive_labels_tnfa = perform_clustering(
        tnfa_modules_filtered, tnfa_modules, sep_lines=False, save_paths=['../figures/tnfa_cooc_eigenvalues.png', '../figures/tnfa_nl_eigenvalues.png', '../figures/tnfa_heatmap.png'], filter_genes=True, recursive=False)
    create_algorithm_heatmaps(tnfa_modules, tnfa_ordering, nonrecursive_labels_tnfa, tnfa_cooccurence_matrix, save_path='../figures/tnfa_algorithm_heatmap.png', sep_lines=False, y_axis='TNFa')
    res = match_clusters_to_modules(spectral_cluster_labels_tnfa, tnfa_ordering, tnfa_modules)
    print_cluster_module_matching(res, tnfa_ordering, spectral_cluster_labels_tnfa)

    fly_transcriptome_ordering, fly_transcriptome_cooccurence_matrix, suggested_clusters_fly_transcriptome, spectral_cluster_labels_fly_transcriptome, nonrecursive_labels_fly_transcriptome = perform_clustering(
        fly_transcriptome_modules_filtered, fly_transcriptome_modules, sep_lines=False, save_paths=['../figures/fly_transcriptome_cooc_eigenvalues.png', '../figures/fly_transcriptome_nl_eigenvalues.png', '../figures/fly_transcriptome_heatmap.png'], filter_genes=True, recursive=False)
    create_algorithm_heatmaps(fly_transcriptome_modules, fly_transcriptome_ordering, nonrecursive_labels_fly_transcriptome, fly_transcriptome_cooccurence_matrix, save_path='../figures/fly_transcriptome_algorithm_heatmap.png', sep_lines=False, y_axis='Fly Transcriptome')
    res = match_clusters_to_modules(spectral_cluster_labels_fly_transcriptome, fly_transcriptome_ordering, fly_transcriptome_modules)
    print_cluster_module_matching(res, fly_transcriptome_ordering, spectral_cluster_labels_fly_transcriptome)
