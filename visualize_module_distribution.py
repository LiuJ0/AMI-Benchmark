import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
from collections import defaultdict
import os

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

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

def process_full_modules_reports(directory):
    # List to store individual dataframes
    dfs = []
    
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        if filename.startswith("full_modules_report") and filename.endswith(".tsv"):
            file_path = os.path.join(directory, filename)
            
            # Extract ALGO from the filename
            # Assuming format: full_modules_report_{ALGO}_{DATASET}.tsv
            parts = filename.split('_')
            if len(parts) >= 4:
                algo = parts[3]
            else:
                algo = "Unknown"
            
            # Read the TSV file into a dataframe
            df = pd.read_csv(file_path, sep='\t')
            
            # Add filename and ALGO as columns
            df['filename'] = filename
            df['ALGO'] = algo
            
            # Append the dataframe to our list
            dfs.append(df)
    
    # Concatenate all dataframes in the list
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    else:
        print("No matching files found.")
        return None
def create_scatter_plot(df, x_col, y_col, title, save_path=None):
    # Set the style for the plot
    plt.style.use('seaborn-v0_8-colorblind')
    
    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the scatter plot, using ALGO for color
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='ALGO', ax=ax)
    
    # Set the title and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust legend
    ax.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Tight layout to prevent clipping of labels
    plt.tight_layout()
    
    if save_path:
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        # Show the plot if not saving
        plt.show()
    
    # Close the figure to free up memory
    plt.close(fig)

def plot_stacked_bar(csv_file, dataset=None, save_path=None):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter the dataframe for the given dataset
    if dataset:
        df_filtered = df[df['Dataset'] == dataset]
    else:
        df_filtered = df
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the stacked bar plot
    df_filtered.plot(x='Algorithm', y=['EV Terms', 'HG Terms'], 
                     kind='bar', ax=ax, logy=True)
    
    # Customize the plot
    plt.title(f'EHR for {dataset} Dataset')
    plt.xlabel('Algorithm')
    plt.ylabel('Count')
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        # Save the plot to the specified path
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Show the plot
    plt.show()
def visualize_module_sizes(modules: Dict[str, List[List[str]]], save_path: str = None):
    # Prepare data for plotting
    data = []
    module_counts = {}
    for algorithm, module_list in modules.items():
        sizes = [len(module) for module in module_list]
        data.extend([(algorithm, size) for size in sizes])
        module_counts[algorithm] = len(module_list)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Algorithm', 'Module Size'])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Create box plot
    sns.boxplot(x='Algorithm', y='Module Size', data=df, whis=[0, 100])
    
    # Add swarm plot for individual data points
    sns.swarmplot(x='Algorithm', y='Module Size', data=df, color=".25", size=4, alpha=0.6)
    
    plt.title('Distribution of Module Sizes by Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Module Size')
    
    # Add count annotations above each box
    unique_algorithms = df['Algorithm'].unique()
    for idx, algorithm in enumerate(unique_algorithms):
        count = module_counts[algorithm]
        plt.text(idx, plt.ylim()[1], f'n={count}', 
                horizontalalignment='center',
                verticalalignment='bottom',
                fontweight='bold')
    
    # Adjust y-axis limits to make room for count annotations
    current_ymin, current_ymax = plt.ylim()
    plt.ylim(current_ymin, current_ymax * 1.1)  # Add 10% padding at the top
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
def create_module_distribution(modules: Dict[str, List[List[str]]], save_path: str = None, log_scale: bool = False):
    # Count module sizes for each algorithm
    size_counts = defaultdict(lambda: defaultdict(int))
    for algorithm, module_list in modules.items():
        for module in module_list:
            size = len(module)
            size_counts[algorithm][size] += 1
    
    # Get unique module sizes and algorithms
    all_sizes = sorted(set(size for counts in size_counts.values() for size in counts))
    algorithms = list(modules.keys())
    
    # Prepare data for stacking
    data = []
    for algorithm in algorithms:
        data.append([size_counts[algorithm][size] for size in all_sizes])
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(all_sizes))
    
    for i, d in enumerate(data):
        ax.bar(all_sizes, d, bottom=bottom, label=algorithms[i])
        bottom += d
    
    ax.set_xlabel('Module Size')
    ax.set_ylabel('Count')
    ax.set_title('Module Size Distribution by Algorithm')
    ax.legend()
    
    # Set y-axis to log scale if requested
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Count (Log Scale)')
    
    # Improve x-axis labeling
    max_ticks = 20  # Maximum number of ticks to show
    step = max(1, len(all_sizes) // max_ticks)
    plt.xticks(all_sizes[::step], all_sizes[::step])
    plt.xticks(rotation=45)  # Rotate labels for better readability
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # visualize module sizes
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
    all_modules = combine_dictionaries(illumina_modules, tvc_modules, tnfa_modules, fly_transcriptome_modules)

    visualize_module_sizes(all_modules, save_path="module_sizes.png")