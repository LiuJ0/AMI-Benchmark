"""
CUTOFFS TO TEST (NODE_CUTOFF, EDGE_CUTOFF):
(0.1, 0.04)
(0.1, 0.1)
(0.2, 0.1)
(0.3, 0.1)
"""
import sys
import subprocess
import math
import os
import pandas as pd
import igraph
import pickle
import time

sys.path.insert(0, '../')
sys.path.insert(0, '../..')
from src import constants
from src.utils.ensembl2entrez import ensembl2entrez_convertor
from src.utils.network import get_network_genes

from src.runners.abstract_runner import AbstractRunner

sys.path.insert(0, '/lab01/Projects/Jason_Projects/aneuploidy/ppi/code/PAPER')
from PAPER.tree_tools import *
from PAPER.gibbsSampling import *
from PAPER.grafting import *

from src.runners import run_pcst


def generate_pval_network(dataset_file_name, network_file_name):
    network = pd.read_csv(network_file_name, sep='\t')
    gene_scores = pd.read_csv(dataset_file_name, sep='\t')
    gene_scores = gene_scores[['id', 'qval']]

    merge1 = network.merge(gene_scores, left_on='node1', right_on='id')
    merge1.drop_duplicates(subset=['node1', 'node2'], inplace=True)

    merge2 = merge1.merge(gene_scores, left_on='node2', right_on='id')
    merge2.drop_duplicates(subset=['node1', 'node2'], inplace=True)

    merge2.drop(columns=['id_x', 'id_y'], inplace=True)
    res = merge2.rename(columns={'pval_x': 'node1_pval', 'pval_y': 'node2_pval'})

    res['edge_weight'] = res['node1_pval'] * res['node2_pval']
    res['edge_weight_sqrt'] = (res['node1_pval'] * res['node2_pval']) ** 0.5
    return res


def filter_network(res, node_cutoff, edge_cutoff, output_folder):
    res1 = res.query(f'node1_pval < {node_cutoff} and node2_pval < {node_cutoff}')
    filtered = res1[res1['edge_weight_sqrt'] < edge_cutoff]
    edges = filtered[['node1', 'node2']].values.tolist()
    vertices = list(set(list(filtered['node1']) + list(filtered['node2'])))
    with open(os.path.join(output_folder, "subnetwork.tsv"), "w") as f:
        for edge in edges:
            f.write(edge[0] + '\t' + edge[1] + '\n')
    g = igraph.Graph()
    g.add_vertices(vertices)
    g.add_edges(edges)
    g.to_undirected()
    g.simplify()
    g.vs["gene_id"] = range(len(g.vs))
    largest = g.clusters().giant()
    return g, largest


class PAPERRunner(AbstractRunner):
    def __init__(self):
        super().__init__("PAPER")

    def create_graph(self, res, output_folder):
        edges = []; vertices = []
        f = open(os.path.join(output_folder, "subnetwork.tsv"), "w")
        for edge in res['elements']['edges']:
            f.write(edge['data']['source'] + '\t' + edge['data']['target'] + '\n')
            edges.append((edge['data']['source'], edge['data']['target']))
            vertices.append(edge['data']['source'])
            vertices.append(edge['data']['target'])
        f.close()
        g = igraph.Graph()
        g.add_vertices(vertices)
        g.add_edges(edges)
        g.to_undirected()
        g.simplify()
        g.vs["gene_id"] = range(len(g.vs))
        bar = g.clusters().giant()
        return bar

    def generate_network_subset(self, dataset_file_name, network_file_name, output_folder, anno_dir="/lab01/Projects/Jason_Projects/aneuploidy/ppi/code/NetworkPPI/anno/"):
        df = pd.read_csv(dataset_file_name, sep='\t')
        genes_list = df[df['qval'] < 0.05]['id']
        active_genes_path = os.path.join(output_folder, "active_genes_list.txt")
        genes_list.to_csv(active_genes_path, index=False, header=False)
        start = time.time()
        res = network_subset(active_genes_path, anno_dir, network_file_name)
        print("NetworkPPI Runtime: ", time.time() - start)
        return self.create_graph(res, output_folder)

    def generate_alternate_network(self, dataset_file_name, network_file_name, output_folder):
        res = generate_pval_network(dataset_file_name, network_file_name)
        g, largest = filter_network(res, NODE_CUTOFF, EDGE_CUTOFF, output_folder)
        return largest

    def generate_pcst_network(self, dataset_file_name, network_file_name, output_folder, node_cutoff=0.1, edge_cutoff=0.1, n_steps=20, theta=0.5):
        print('Running PAPER with parameters node_cutoff:', node_cutoff, 'edge_cutoff:', edge_cutoff,
              'n_steps:', n_steps, 'theta:', theta)
        res, pcst_vertices, pcst_edges = run_pcst.get_pcst_network(dataset_file_name, network_file_name, output_folder,
                                                              node_cutoff, edge_cutoff, n_steps, theta)
        edges = list(res[['node1', 'node2']].itertuples(index=False, name=None))
        g = igraph.Graph()
        g.add_vertices(pcst_vertices)
        g.add_edges(edges)
        g.to_undirected()
        g.simplify()
        largest = g.clusters().giant()
        return largest

    def extract_modules(self, res, graf, method="coo", SIZE_LOWER_LIM=6, COO_THRESH=0.4, MAX_OVERLAP=0.6, MIN_UNIQUE=0.1, totalM=50):
        if method == "coo":
            node_tree_coo = res["node_tree_coo"]
            group_map = {}
            for ii in range(len(node_tree_coo)):
                myname = graf.vs[ii]["name"]

                myco = node_tree_coo[ii, :]
                myco = myco / sum(myco)

                sig_groups = np.where(myco > COO_THRESH)[0]
                if (len(sig_groups) == 0):
                    sig_groups = [np.argmax(myco)]

                for g in sig_groups:
                    group_map[g] = group_map.get(g, []) + [ii]
            valid_groups = np.array(sorted(group_map.keys()))
            for g in valid_groups:
                print(str(g) + ": " + str(len(group_map[g])))
            return [group_map[g] for g in group_map if g in valid_groups]
        elif method == "prp":
            tree_count = res["tree_count"]
            tree_ord = np.argsort(-np.array(tree_count))
            # compute number of trees with significant proportion of total MCMC samples
            J = len([j for j in range(len(tree_count)) if tree_count[tree_ord[j]] > 0.1 * totalM])

            tree_dict = {}
            tree_prob_dict = {}
            valid_trees = list(range(J))
            for j in range(J):
                mytree = tree_ord[j]

                my_prob = res["freq"][mytree] / sum(res["freq"][mytree])
                my_prob_s = -np.sort(-my_prob)
                try:
                    topn_ix = int(np.max(np.where(np.cumsum(my_prob_s) <= 0.95)))
                except ValueError:
                    valid_trees.remove(j)
                    continue
                my_prob_args = np.argsort(-my_prob)

                top_nodes = my_prob_args[0:topn_ix]
                tree_dict[j] = top_nodes
                tree_prob_dict[j] = my_prob

            for j in reversed(valid_trees):
                # intersection between current tree and other trees
                my_nodes = set(tree_dict[j])
                my_unique_nodes = set(tree_dict[j])
                overlap = np.zeros(J)
                for k in valid_trees:
                    if (k == j):
                        continue
                    my_unique_nodes = my_unique_nodes.difference(tree_dict[k])

                    overlap[k] = len(my_nodes.intersection(tree_dict[k]))
                try:
                    if (max(overlap) / len(tree_dict[j]) > MAX_OVERLAP):
                        valid_trees.remove(j)
                        continue
                except:
                    valid_trees.remove(j)
                    continue
                try:
                    if (len(my_unique_nodes) / len(tree_dict[j]) < MIN_UNIQUE):
                        valid_trees.remove(j)
                except:
                    valid_trees.remove(j)
                    continue
            for j in valid_trees:
                print("Tree " + str(j) + " has " + str(len(tree_dict[j])) + " nodes")
            ## compute the number of nodes not in any of the trees
            all_nodes = set(range(n))
            for j in valid_trees:
                all_nodes = all_nodes.difference(tree_dict[j])
            print(len(all_nodes) / n)
            return [tree_dict[g] for g in tree_dict]\

    def label_ids(self, graf, groups):
        labeled = []
        for group in groups:
            labeled.append([graf.vs[i]["name"] for i in group])
        return labeled

    def run_PAPER_full(self, network, method="collapsed", M=400, size_thresh=0.02, birth_thresh=0.6, shatter_thresh=1):
        start = time.time()
        res = gibbsToConv(network, DP=True, method='collapsed', M=M, size_thresh=size_thresh, birth_thresh=birth_thresh)
        print("PAPER Runtime: ", time.time() - start)
        module_ids = self.extract_modules(res[1], network)
        modules = self.label_ids(network, module_ids)
        return modules

    def run(self, dataset_file_name, network_file_name, output_folder, **kwargs):
        defaults = {
            'iterations': 400,
            'node_cutoff': 0.1,
            'edge_cutoff': 0.1,
            'n_steps': 20,
            'theta': 0.5,
            'shatter': 1
        }
        M, node_cutoff, edge_cutoff, n_steps, theta, shatter = [
            kwargs.get(key, default) for key, default in defaults.items()
        ]
        print("Begin generation of subnetwork")
        subnet = self.generate_pcst_network(dataset_file_name, network_file_name, output_folder,
                                            node_cutoff=node_cutoff, edge_cutoff=edge_cutoff, n_steps=n_steps, theta=theta)
        print("Begin running PAPER for", M, "iterations")
        modules = self.run_PAPER_full(subnet, M=M, shatter_thresh=shatter)
        modules = [x for x in modules if len(x) > 3]
        bg_genes = get_network_genes(network_file_name)
        all_bg_genes = [bg_genes for x in modules]
        return modules, all_bg_genes
