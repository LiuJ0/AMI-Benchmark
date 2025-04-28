"""
This file contains some utility functions for loading in moduels and preprocessing. 
"""

import os
from typing import Dict, List
import networkx as nx
from collections import defaultdict
import pandas as pd

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

def modules_to_dataframe(res_dir, algos, dataset):
    all_data = []
    
    for algo in algos:
        modules = read_modules(res_dir, algo, dataset)
        for index, module in enumerate(modules):
            all_data.append({
                "dataset": dataset,
                "algo": algo,
                "module_id": index,
                "genes": module,})

    return pd.DataFrame(all_data)