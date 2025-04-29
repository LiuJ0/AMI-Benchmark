from utils import *
import pandas as pd
import os
from collections import defaultdict, Counter
import networkx as nx
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple
from earth_movers import *
import math
import functools

def sort_by_count(input_dict):
    return dict(sorted(input_dict.items(), key=lambda x: x[1]['count'], reverse=True))

def find_overlapping_genes(modules):
    gene_info = defaultdict(lambda: {"algorithms": defaultdict(set)})
    
    for algo_name, algo_modules in modules.items():
        for i, module in enumerate(algo_modules):
            for gene in module:
                gene_info[gene]["algorithms"][algo_name].add(i)
    print(gene_info)
    # Convert defaultdict to regular dict and calculate unique algorithm count
    overlapping_genes = {}
    for gene, info in gene_info.items():
        unique_algo_count = len(info["algorithms"])
        overlapping_genes[gene] = {
            "count": unique_algo_count,
            "algorithms": {algo: list(modules) for algo, modules in info["algorithms"].items()}
        }

    return overlapping_genes

if __name__ == "__main__":
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

    # change defaultdict object to a regular dictionary
    illumina_genes = sort_by_count(find_overlapping_genes(illumina_modules))
    tvc_genes = sort_by_count(find_overlapping_genes(tvc_modules))
    tnfa_genes = sort_by_count(find_overlapping_genes(tnfa_modules))
    fly_transcriptome_genes = sort_by_count(find_overlapping_genes(fly_transcriptome_modules))

    def count_distribution(genes):
        count_dist = Counter()
        for gene, info in genes.items():
            count_dist[info["count"]] += 1
        return count_dist

    print("Illumina", count_distribution(illumina_genes))
    print("Tvc", count_distribution(tvc_genes))
    print("TnfA", count_distribution(tnfa_genes))
    print("Fly transcriptome", count_distribution(fly_transcriptome_genes))

    for gene, info in illumina_genes.items():
        if info["count"] == 3:
            print(gene, info)

    print("tvc")
    for gene, info in tvc_genes.items():
        if info["count"] == 3:
            print(gene, info)

    df = pd.DataFrame(illumina_genes).T
    df.to_csv('illumina_overlap.csv')
    df = pd.DataFrame(tvc_genes).T
    df.to_csv('tvc_overlap.csv')