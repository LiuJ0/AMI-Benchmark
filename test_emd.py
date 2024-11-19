import pandas as pd 
import networkx as nx
import os
import pickle 

from earth_movers import *
from visualize_modules import *
from solution_similarity import *
from visualize_module_distribution import *
from merge_modules import *

G = nx.Graph()

edge_list = [(1, 2), (2, 3), (4, 5)]
G.add_edges_from(edge_list)

# Create subnetworks
subnetwork1 = G.subgraph([1,2,3])
subnetwork2 = G.subgraph([4, 5])

# compute similarity
similarity = emd_subgraph_similarity(G, subnetwork1.nodes, subnetwork2.nodes)
print(similarity)