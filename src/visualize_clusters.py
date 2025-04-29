import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Wedge
from collections import defaultdict
from utils import *

def read_cluster_data(csv_file):
    """
    Read cluster data from CSV file and parse algorithm lists.
    
    :param csv_file: Path to CSV file with 'gene,algos' format
    :return: Dictionary mapping genes to their algorithms
    """
    df = pd.read_csv(csv_file)
    gene_to_algos = {}
    
    for _, row in df.iterrows():
        # Remove quotes and split algorithms
        algos = row['algos'].strip('"').split(',')
        gene_to_algos[row['gene']] = algos
        
    return gene_to_algos

def create_pie_chart_node(pos, algorithms, radius=0.1):
    """
    Create a pie chart node showing algorithm membership.
    
    :param pos: (x, y) position of the node
    :param algorithms: List of algorithms this gene belongs to
    :param radius: Radius of the pie chart
    :return: List of matplotlib patches for the pie chart
    """
    # Define colors for each algorithm
    color_map = {
        'DOMINO': '#FF9999',    # Light red
        'FDRnet': '#99FF99',    # Light green
        'PAPER': '#9999FF'      # Light blue
    }
    
    # Count number of algorithms
    n_algos = len(algorithms)
    if n_algos == 0:
        return []
    
    # Calculate angle for each section
    angle = 360.0 / n_algos
    
    # Create wedges
    patches = []
    for i, algo in enumerate(algorithms):
        start_angle = i * angle
        wedge = Wedge(pos, 0.05, start_angle, start_angle + angle,
                     facecolor=color_map[algo], edgecolor='black', linewidth=0.5)
        patches.append(wedge)
    
    return patches

def filter_graph_by_genes(G, genes):
    """
    Filter the graph to only include specified genes and their connections.
    
    :param G: Original NetworkX graph
    :param genes: Set of genes to keep
    :return: Filtered NetworkX graph
    """
    # Create a subgraph with only the genes from the CSV
    filtered_nodes = set(G.nodes()) & genes
    subgraph = G.subgraph(filtered_nodes).copy()
    
    return subgraph

def visualize_cluster(G, gene_to_algos, output_file=None):
    """
    Visualize the cluster with pie chart nodes representing algorithm membership.
    
    :param G: NetworkX graph representing the filtered PPI network
    :param gene_to_algos: Dictionary mapping genes to their algorithms
    :param output_file: Optional path to save the visualization
    """
    # Create figure and axis
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    
    # Get layout positions
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=2)
    
    # Create pie chart nodes for each gene
    for gene in G.nodes():
        algorithms = gene_to_algos.get(gene, [])
        patches = create_pie_chart_node(pos[gene], algorithms)
        for patch in patches:
            ax.add_patch(patch)
    
    # Add gene labelsx  
    
    # Create legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#FF9999', label='DOMINO'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#99FF99', label='FDRnet'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#9999FF', label='PAPER')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=18)
    
    # Set axis properties
    plt.axis('equal')
    plt.axis('off')
    
    # Add title
    # plt.title('Cluster Visualization\nNodes colored by algorithm membership', pad=20)
    
    # Add statistics
    #stats_text = f"Number of nodes: {G.number_of_nodes()}\n"
    #stats_text += f"Number of edges: {G.number_of_edges()}"
    #plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             #verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.show()

def main(csv_file, ppi_graph):
    """
    Main function to create the visualization.
    
    :param csv_file: Path to CSV file with gene-algorithm mappings
    :param ppi_graph: NetworkX graph representing the PPI network
    """
    # Read cluster data
    gene_to_algos = read_cluster_data(csv_file)
    
    # Filter graph to only include genes from the CSV
    filtered_graph = filter_graph_by_genes(ppi_graph, set(gene_to_algos.keys()))
    
    # Create visualization
    visualize_cluster(filtered_graph, gene_to_algos, output_file="cluster0.png")

if __name__ == "__main__":
    dip = network_to_graph("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/original/dip_original.sif", source="node1", target="node2")
    dip.remove_edges_from(nx.selfloop_edges(dip))
    main("cluster0.txt", dip)