import pandas as pd 
import networkx as nx
import os
import pickle 

from earth_movers import *
from visualize_modules import *
from solution_similarity import *
from visualize_module_distribution import *
from merge_modules import *

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

def count_terms(x): 
    if type(x) != type(0.01):
        return x.count('\n') + 1
    else: 
        return 0
    
def modules_to_dataframe(res_dir, algos, dataset):
    all_data = []
    
    for algo in algos:
        modules = read_modules(res_dir, algo, dataset)
        for index, module in enumerate(modules):
            all_data.append({
                "algo": algo,
                "module_id": index,
                "genes": module,})

    return pd.DataFrame(all_data)

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

#df = process_full_modules_reports(modules_report_dir)
#df['go_terms'] = df['fp'].apply(count_terms) + df['tp'].apply(count_terms)
#create_scatter_plot(df, "module_size", "EHR", title="Module Size vs. EHR", save_path="figures/module_size_vs_score.png")
#create_scatter_plot(df, "module_size", "go_terms", title="Module Size vs. GO Terms", save_path="figures/module_size_vs_go_terms.png")

illumina_modules = get_all_modules(res_dir, algos, "gene_scores_illumina_v3")
tvc_modules = get_all_modules(res_dir, algos, "gene_scores_tvc")
tnfa_modules = get_all_modules(res_dir, algos, "tnfa")
fly_transcriptome_modules = get_all_modules(res_dir, algos, "fly_transcriptome")

all_modules = combine_dictionaries(illumina_modules, tvc_modules, tnfa_modules, fly_transcriptome_modules)

"""
#illumina_conserved_modules = find_conserved_modules(three_dbs, illumina_modules, threshold=2)
tnfa_conserved_modules = find_conserved_modules(dip, tnfa_modules, threshold=2)

colors = {
    "algo1": "#d47264", # red
    "algo2": "#8ec1da", # blue 
    "shared": "#a559aa", # purple
    "path": "#f0f0f0" # grey
}

#visualize_conserved_modules(three_dbs, illumina_conserved_modules, illumina_modules, name2ENSG, 
#                            similarity_threshold = 0, output_dir="conserved_modules_illumina", colors=colors)

visualize_conserved_modules(dip, tnfa_conserved_modules, tnfa_modules, name2ENSG,
                            similarity_threshold = 0, output_dir="conserved_modules_tnfa", colors=colors)


tnfa_similarity_df = compute_similarity_matrix(dip, tnfa_modules, algos)
tnfa_similarity_df.to_csv('../similarity_scores/tnfa_similarity.csv')

fly_transcriptome_similarity_df = compute_similarity_matrix(string_db, fly_transcriptome_modules, algos)
fly_transcriptome_similarity_df.to_csv('../similarity_scores/fly_transcriptome_similarity.csv')
"""
from collections import Counter

def count_connected_component_sizes(G):
    # Get the connected components
    connected_components = list(nx.connected_components(G))
    
    # Count the size of each component
    component_sizes = [len(component) for component in connected_components]
    
    # Use Counter to count the frequency of each size
    size_counts = Counter(component_sizes)
    
    # Sort the results by component size
    sorted_counts = sorted(size_counts.items(), key=lambda x: x[0])
    
    return sorted_counts

# [(1, 144), (2, 129), (3, 39), (4, 9), (5, 12), (6, 2), (7, 3), (9, 1), (11, 1), (12, 1), (23, 1), (33, 1), (2318, 1)]
#result = count_connected_component_sizes(dip)
#print(result)
"""
create_module_distribution(illumina_modules, save_path="../figures/module_distribution_illumina.png")
visualize_module_sizes(illumina_modules, save_path="../figures/module_sizes_illumina.png")
create_module_distribution(all_modules, save_path="../figures/module_distribution_all.png")
visualize_module_sizes(all_modules, save_path="../figures/module_sizes_all.svg")

plot_stacked_bar("ehr_counts.csv", "Illumina", save_path="figures/ehr_counts_illumina.png")
plot_stacked_bar("ehr_counts.csv", "Tvc", save_path="figures/ehr_counts_tvc.png")
plot_stacked_bar("ehr_counts.csv", "TNFA", save_path="figures/ehr_counts_tnfa.png")
plot_stacked_bar("ehr_counts.csv", "Fly Transcriptome", save_path="figures/ehr_counts_fly_transcriptome.png")
plot_stacked_bar("ehr_counts.csv", save_path="figures/ehr_counts_all.png")

# Compute solution similarity for Illumina dataset

illumina_similarity_df = compute_similarity_matrix(three_dbs, illumina_modules, algos)
illumina_similarity_df.to_csv('illumina_similarity.csv')
tvc_similarity_df = compute_similarity_matrix(three_dbs, tvc_modules, algos)
tvc_similarity_df.to_csv('tvc_similarity.csv')
tnfa_similarity_df = compute_similarity_matrix(dip, tnfa_modules, algos)
tnfa_similarity_df.to_csv('tnfa_similarity.csv')
fly_transcriptome_similarity_df = compute_similarity_matrix(string_db, fly_transcriptome_modules, algos)
fly_transcriptome_similarity_df.to_csv('fly_transcriptome_similarity.csv')
"""

#illumina_conserved_modules = find_conserved_modules(three_dbs, illumina_modules)
#tvc_conserved_modules = find_conserved_modules(three_dbs, tvc_modules)
#tnfa_conserved_modules = find_conserved_modules(dip, tnfa_modules, threshold=0.1)
#print(tnfa_conserved_modules)
#print(tnfa_conserved_modules)
fly_transcriptome_conserved_modules = find_conserved_modules(string_db, fly_transcriptome_modules)

colors = {
    "algo1": "#d47264", # red
    "algo2": "#8ec1da", # blue 
    "shared": "#a559aa", # purple
    "path": "#f0f0f0" # grey
}

#visualize_conserved_modules(three_dbs, illumina_conserved_modules, illumina_modules, name2ENSG, similarity_threshold = 0.3, output_dir="../conserved_modules/conserved_modules_illumina", colors=colors)
#visualize_conserved_modules(three_dbs, tvc_conserved_modules, tvc_modules, name2ENSG, similarity_threshold = 0.3, output_dir="../conserved_modules/conserved_modules_tvc", colors=colors)
#visualize_conserved_modules(dip, tnfa_conserved_modules, tnfa_modules, name2ENSG, similarity_threshold = 0.4, output_dir="../conserved_modules/conserved_modules_tnfa", colors=colors)
visualize_conserved_modules(string_db, fly_transcriptome_conserved_modules, fly_transcriptome_modules, name2ENSG_fb, similarity_threshold = 0.3, output_dir="../conserved_modules/conserved_modules_fly_transcriptome", colors=colors, use_labels=False)

"""
res = merge_modules( fly_transcriptome_modules, string_db, "merge-intersect")
print(res)
all_modules = [(set(module), algo, i) for algo, modules in fly_transcriptome_modules.items() for i, module in enumerate(modules)]
visualize_merge_intersect(string_db, all_modules, res, save_path="../figures/merge_intersect.png")


combined_datasets = {"Illumina": illumina_modules, "Tvc": tvc_modules, "TNFA": tnfa_modules, "Fly Transcriptome": fly_transcriptome_modules}
networks = {"Illumina": three_dbs, "Tvc": three_dbs, "TNFA": dip, "Fly Transcriptome": string_db}
results = merge_across_algos(combined_datasets, networks)
results.to_csv("../merged_modules.csv")

fly_transcriptome_merged = merge_modules(fly_transcriptome_modules, string_db, "merge-intersect")
analyze_module_overlap(fly_transcriptome_merged).to_csv('../fly_transcriptome_merged_overlap.csv')
"""
