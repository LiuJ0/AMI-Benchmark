import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from typing import List, Set, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from collections import defaultdict
import os
import pandas as pd
from pulp import *

def find_conserved_modules(G: nx.Graph, algorithm_modules: Dict[str, List[List[int]]], 
                           threshold: float = 0.3, min_algorithms: int = 2) -> Dict[Tuple, List[Tuple]]:
    """
    Find conserved modules across multiple algorithms using EMD.
    
    :param G: The complete graph
    :param algorithm_modules: Dictionary of algorithms and their modules
    :param threshold: Similarity threshold for considering a match
    :param min_algorithms: Minimum number of algorithms a module should be conserved in
    :return: Dictionary of conserved modules
    """
    all_matches = defaultdict(list)
    algorithms = list(algorithm_modules.keys())

    # Compare modules between each pair of algorithms
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            algo1, algo2 = algorithms[i], algorithms[j]
            print(algo1, algo2)
            matches = find_module_matches(G, algorithm_modules[algo1], algorithm_modules[algo2], threshold)
            for module1, module2, similarity in matches:
                all_matches[(algo1, module1)].append((algo2, module2, similarity))
                all_matches[(algo2, module2)].append((algo1, module1, similarity))

    # Find conserved modules
    conserved_modules = {}
    for (algo, module), matches in all_matches.items():
        if len(set(match[0] for match in matches)) >= min_algorithms - 1:
            conserved_modules[(algo, module)] = matches

    return conserved_modules

def find_module_matches(G: nx.Graph, modules1: List[List[int]], modules2: List[List[int]], 
                        threshold: float = 0.5) -> List[Tuple[int, int, float]]:
    """
    Find matches between two lists of modules based on their EMD similarity.
    """
    similarities = []
    for i, module1 in enumerate(modules1):
        for j, module2 in enumerate(modules2):
            emd = emd_subgraph_similarity(G, module1, module2)
            if emd == np.inf:
                similarity = 0
            else: 
                similarity = 1 / (1 + emd)  # Convert EMD to a similarity measure
            similarities.append((i, j, similarity))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    matches = []
    used1, used2 = set(), set()
    for i, j, sim in similarities:
        if sim < threshold:
            break
        if i not in used1 and j not in used2:
            matches.append((i, j, sim))
            used1.add(i)
            used2.add(j)
    
    return matches

def emd_subgraph_similarity(G: nx.Graph, subgraph1, subgraph2):

    # check if modules are connected
    if nx.has_path(G, subgraph1[0], subgraph2[0]) == False:
        return np.inf

    # represent subgraphs as distributions
    nodes1 = set(subgraph1)
    nodes2 = set(subgraph2)
    all_nodes = list(nodes1.union(nodes2))
    
    distribution1 = create_distribution(all_nodes, nodes1)
    distribution2 = create_distribution(all_nodes, nodes2)
    
    # calculate distance matrix only for relevant nodes
    distance_matrix = calculate_distance_matrix(G, all_nodes)
    
    # solve the transportation problem
    flow = solve_transportation_problem(distribution1, distribution2, distance_matrix)
    
    # calculate EMD
    emd = calculate_emd(flow, distance_matrix)
    
    return emd

def create_distribution(all_nodes, subgraph_nodes):
    distribution = np.zeros(len(all_nodes))
    for i, node in enumerate(all_nodes):
        if node in subgraph_nodes:
            distribution[i] = 1
    return distribution / np.sum(distribution)  # Normalize

def calculate_distance_matrix(G, nodes):
    n = len(nodes)
    distance_matrix = np.zeros((n, n))
    disconnected_modules = 0
    for i in range(n):
        for j in range(i+1, n):
            if nodes[i] in G and nodes[j] in G:
                try:
                    dist = nx.shortest_path_length(G, source=nodes[i], target=nodes[j])
                except nx.NetworkXNoPath:
                    disconnected_modules += 1
                    dist = np.inf
            else: 
                disconnected_modules += 1
                dist = np.inf
            distance_matrix[i, j] = distance_matrix[j, i] = dist
    if disconnected_modules > 0: 
        print(f"Warning: {disconnected_modules} modules are disconnected from the main graph")
    return distance_matrix

def solve_transportation_problem(dist1, dist2, distance_matrix):
    n, m = len(dist1), len(dist2)
    
    # Create the LP problem
    prob = LpProblem("Earth Mover's Distance", LpMinimize)
    
    # Create variables
    flow_vars = [[LpVariable(f'flow_{i}_{j}', lowBound=0) for j in range(m)] for i in range(n)]
    
    # Objective function
    prob += lpSum(distance_matrix[i][j] * flow_vars[i][j] for i in range(n) for j in range(m))
    
    # Constraints
    for i in range(n):
        prob += lpSum(flow_vars[i][j] for j in range(m)) == dist1[i]
    
    for j in range(m):
        prob += lpSum(flow_vars[i][j] for i in range(n)) == dist2[j]
    
    # Solve the problem
    prob.solve()
    
    # Extract the flow
    flow = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            flow[i][j] = flow_vars[i][j].varValue
    
    return flow

def solve_transportation_problem_network_simplex(dist1, dist2, distance_matrix):
    # Ensure the distributions sum to the same total
    total1, total2 = sum(dist1), sum(dist2)
    if not np.isclose(total1, total2):
        raise ValueError("The two distributions must sum to the same total")

    # Convert distributions to integer values (required by linear_sum_assignment)
    scale = 1e6  # Scale factor to convert floats to integers
    dist1_int = np.round(np.array(dist1) * scale).astype(int)
    dist2_int = np.round(np.array(dist2) * scale).astype(int)

    # Create supply and demand arrays
    supply = np.concatenate([dist1_int, np.zeros(len(dist2_int), dtype=int)])
    demand = np.concatenate([np.zeros(len(dist1_int), dtype=int), dist2_int])

    # Create the cost matrix
    n, m = len(dist1), len(dist2)
    cost_matrix = np.pad(distance_matrix, ((0, m), (0, n)), mode='constant', constant_values=0)

    # Solve the transportation problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Extract the flow
    flow = np.zeros((n, m))
    for r, c in zip(row_ind, col_ind):
        if r < n and c < m:
            flow[r, c] = min(supply[r], demand[c]) / scale

    return flow

def calculate_emd(flow, distance_matrix):
    return np.sum(flow * distance_matrix)