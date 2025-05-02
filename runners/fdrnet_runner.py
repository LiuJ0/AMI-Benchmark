import sys
import subprocess
import math
sys.path.insert(0, '../')
sys.path.insert(0, '../..')
import os
import pandas as pd

from src import constants
from src.implementations.domino import main as domino_main
from src.utils.ensembl2entrez import ensembl2entrez_convertor
from src.utils.network import get_network_genes

from src.runners.abstract_runner import AbstractRunner

fdrnet_dir = "/lab01/Projects/Jason_Projects/aneuploidy/ppi/code/FDRnet"


def prepare_data(dataset, network, output_folder):
    data_df = pd.read_csv(dataset, sep='\t')
    network_df = pd.read_csv(network, sep='\t')
    # Create the index-to-gene file
    index2gene = data_df['id']
    index2gene.index = index2gene.index + 1
    index2gene.to_csv(os.path.join(output_folder, 'index2gene'), sep='\t', header=False)
    # Create the gene-to-score file
    gene2score = data_df[['id', 'pval']]
    gene2score.to_csv(os.path.join(output_folder, 'gene2score.txt'), sep='\t', header=False, index=False)
    # Create the edge list
    index2gene_dict = dict(zip(data_df['id'], range(1, len(data_df) + 1)))
    network_df['node1'] = network_df['node1'].map(index2gene_dict)
    network_df['node2'] = network_df['node2'].map(index2gene_dict)
    network_df = network_df[['node1', 'node2']]
    network_df.dropna(inplace=True)
    network_df = network_df.astype(int)
    network_df.to_csv(os.path.join(output_folder, 'edge_list'), sep='\t', header=False, index=False)


class FDRnetRunner(AbstractRunner):
    def __init__(self):
        super().__init__("FDRnet")

    def write_bash(self, dataset_file_name, network_file_name, output_folder, bd=0.1, sb=0.1, sz=400, al=0.05, rg=0.01):
        prepare_data(dataset_file_name, network_file_name, output_folder)
        with open(os.path.join(output_folder, "runFDRnet.sh"), "w") as f:
            # previous line without f-strings
            f.write("Rscript /lab01/Projects/Jason_Projects/aneuploidy/ppi/code/FDRnet/fdr_correction.r " + output_folder + "/gene2score.txt " + output_folder + "/gene2score_fdr.txt \n" +
                    "python " + fdrnet_dir + "/src/FDRnet_main.py -igi " + os.path.join(output_folder, "index2gene") + " -iel " + os.path.join(output_folder, "edge_list") + " -igl " + 
                    os.path.join(output_folder, "gene2score_fdr.txt") + " -ofn " + output_folder + "/fdrnet_output.csv -se all" + " -bd " + str(bd) + " -sz " + str(sz) + " -al " + str(al) + " -rg " + str(rg) + " -sb " + str(sb) + "\n")
        return os.path.join(output_folder, "runFDRnet.sh")

    def get_modules(self, output_folder):
        df = pd.read_csv(os.path.join(output_folder, "fdrnet_output.csv"))
        # Contains > 1 gene
        df_filtered = df[df['Subnetwork'].str.contains(' ')]
        modules = df_filtered['Subnetwork'].str.split(' ').tolist()
        print("Number of modules: ", len(modules))
        return modules

    def run(self, dataset_file_name, network_file_name, output_folder, **kwargs):
        defaults = {
            'bd': 0.1,
            'sb': 0.1,
            'sz': 400,
            'al': 0.05,
            'rg': 0.01
        }
        bd, sb, sz, al, rg = [kwargs.get(key, default) for key, default in defaults.items()]
        bash_file = self.write_bash(dataset_file_name, network_file_name, output_folder, bd=bd, sb=sb, sz=sz, al=al, rg=rg)
        subprocess.run(args=["bash", bash_file])
        modules = self.get_modules(output_folder)
        bg_genes = get_network_genes(network_file_name)
        all_bg_genes = [bg_genes for x in modules]
        return modules, all_bg_genes
