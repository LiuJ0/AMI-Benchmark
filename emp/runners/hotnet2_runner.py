import sys
import subprocess
import math
sys.path.insert(0, '../')
sys.path.insert(0, '../..')
import os
import pandas as pd
import numpy as np
import re

from src import constants
from src.implementations.domino import main as domino_main
from src.utils.ensembl2entrez import ensembl2entrez_convertor
from src.utils.network import get_network_genes

from src.runners.abstract_runner import AbstractRunner


NETWORK_CACHE="/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/hotnet2_permutations_3dbs/"

def check_file_exists(dataset_file_name, beta, CACHE_DIR):
    base_name = os.path.basename(dataset_file_name).replace('.tsv', '')
    pattern = f'{base_name}_ppr_{beta}_(\d+)\.h5'
    
    for file in os.listdir(CACHE_DIR):
        if re.match(pattern, file):
            return True
    
    return False

class HotNet2Runner(AbstractRunner):
    def __init__(self):
        super().__init__("HotNet2")
    def make_gene_index(self, dataset_file_name, output_folder): 
        df = pd.read_csv(dataset_file_name, sep='\t')
        df['id'].to_csv(os.path.join(output_folder, "gene_index.txt"), sep='\t', index=True, header=False)
        return os.path.join(output_folder, "gene_index.txt")
    
    def make_edge_list(self, network_file_name, output_folder): 
        network = pd.read_csv(network_file_name, sep='\t')
        gene_index = pd.read_csv(os.path.join(output_folder, "gene_index.txt"), sep='\t', header=None, names=['index', 'gene'])
        gene_to_index = dict(zip(gene_index['gene'], gene_index['index']))
        # map each gene in the network to its index
        network['node1'] = network['node1'].apply(lambda x: gene_to_index.get(x, np.nan))
        network['node2'] = network['node2'].apply(lambda x: gene_to_index.get(x, np.nan))
        network_filtered = network.dropna(subset=['node1', 'node2'])
        network_filtered = network_filtered[['node1', 'node2']].astype({'node1': int, 'node2': int})
        network_filtered.to_csv(os.path.join(output_folder, "edge_list.txt"), sep='\t', index=False, header=False)
        return os.path.join(output_folder, "edge_list.txt")
    
    def make_heat_file(self, dataset_file_name, output_folder):
        df = pd.read_csv(dataset_file_name, sep='\t')
        df['hotnet2'] = df['qval'].apply(lambda x: -math.log(x, 10))
        df[['id', 'hotnet2']].to_csv(os.path.join(output_folder, "heat_file.txt"), sep='\t',  index=False, header=False)
        print("Heat file made")
        return os.path.join(output_folder, "heat_file.txt")

    def make_network_files(self, dataset_file_name, network_file_name, output_folder, gene_index, edge_list, cores=1, beta=0.4, network_permutations=100, heat_permutations=100, q=115):
        with open(os.path.join(output_folder, "make_network_files.sh"), "w") as f:
            f.write("""hotnet2="/lab01/Projects/Jason_Projects/aneuploidy/ppi/code/hotnet2"
heat_input={heat_input}
heat_file={heat_file}
output={output}
gene_index={gene_index}
edge_list={edge_list}
output_dir={output_dir}
dataset_name={dataset_name}
permutation_names={permutation_names}
network_permutations={network_permutations}
heat_permutations={heat_permutations}
cores={cores}
beta={beta}
network_file={network_file}

cd $hotnet2
source /lab01/Projects/Jason_Projects/aneuploidy/ppi/code/hotnet2-env/bin/activate
                    
echo -e "\nMAKING NETWORK FILE\n"
                    
python $hotnet2/makeNetworkFiles.py \
        -e $edge_list \
        -i $gene_index \
        -is 0 \
        -o $output_dir \
        -nn $dataset_name \
        -p $dataset_name \
        -b $beta \
        -np $network_permutations \
        -q {q} \
        -c $cores
        """.format(
        heat_input=os.path.join(output_folder,"heat_file.txt"),
        heat_file=os.path.join(output_folder, "heat_file.json"),
        output=os.path.join(output_folder, "results"),
        gene_index=gene_index, 
        edge_list=edge_list,
        output_dir=output_folder,
        dataset_name=os.path.basename(dataset_file_name).replace('.tsv', ''),
        permutation_names=os.path.join(output_folder, 'permuted', os.path.basename(dataset_file_name).replace('.tsv', '') + '_ppr_' + str(beta) + '_##NUM##.h5'),
        network_file=os.path.join(output_folder, os.path.basename(dataset_file_name).replace('.tsv', '') + '_ppr_' + str(beta) + '.h5'),
        network_permutations=network_permutations,
        heat_permutations=heat_permutations,
        cores=cores,
        q=q,
        beta=beta))
        print("Making network files")
        return os.path.join(output_folder, "make_network_files.sh")
    
    def write_bash(self, dataset_file_name, network_file_name, output_folder, gene_index, edge_list, network_file, permutation_names, cores=1, beta=0.4, network_permutations=100, heat_permutations=100, q=115):
        with open(os.path.join(output_folder, "run_hotnet2.sh"), 'w') as f:
            f.write("""hotnet2="/lab01/Projects/Jason_Projects/aneuploidy/ppi/code/hotnet2" 
heat_input={heat_input}
heat_file={heat_file}
output={output}
gene_index={gene_index}
edge_list={edge_list}
output_dir={output_dir}
dataset_name={dataset_name}
permutation_names={permutation_names}
network_permutations={network_permutations}
heat_permutations={heat_permutations}
cores={cores}
beta={beta}
network_file={network_file}

cd $hotnet2
source /lab01/Projects/Jason_Projects/aneuploidy/ppi/code/hotnet2-env/bin/activate

echo -e "\nMAKING HEAT FILE\n"

python $hotnet2/makeHeatFile.py scores -hf $heat_input -o $heat_file -n test

echo -e "\nRUNNING HOTNET2\n"

python $hotnet2/HotNet2.py \
    -nf $network_file \
    -pnp $permutation_names \
    -hf $heat_file \
    -o $output \
    -c $cores \
    --verbose 4 \
    -np $network_permutations \
    -hp $heat_permutations \
""".format(
        heat_input=os.path.join(output_folder,"heat_file.txt"),
        heat_file=os.path.join(output_folder, "heat_file.json"),
        output=os.path.join(output_folder, "results"),
        gene_index=gene_index, 
        edge_list=edge_list,
        output_dir=output_folder,
        dataset_name=os.path.basename(dataset_file_name).replace('.tsv', ''),
        permutation_names=permutation_names,
        network_file=network_file,
        network_permutations=network_permutations,
        heat_permutations=heat_permutations,
        cores=cores,
        q=q,
        beta=beta))
        print("Bash file written")
        return os.path.join(output_folder, "run_hotnet2.sh")
    def get_modules(self, output_folder):
        # figure out if we need to write module directory or if it's implemented
        report_path = os.path.join(output_folder, "results", "consensus")
        modules = []
        for line in open(os.path.join(report_path, "subnetworks.tsv")).readlines():
            if not line[0] == "#":
                trim = lambda x: x.replace('\n', '').replace('\t', '')
                modules.append([trim(x) for x in line.split(' ')])
        print("Number of modules: ", len(modules))
        return modules
    def run(self, dataset_file_name, network_file_name, output_folder, **kwargs):
        defaults = {
            'network_permutations': 10,
            'heat_permutations': 10,
            'cores': 5,
            'beta': 0.5,
            'q': 3,
            'cache_network': False
        }
        network_permutations, heat_permutations, cores, beta, q, cache_network = [
            kwargs.get(key, default) for key, default in defaults.items()
        ]
        gene_index = self.make_gene_index(dataset_file_name, output_folder)
        edge_list = self.make_edge_list(network_file_name, output_folder)
        heat_file = self.make_heat_file(dataset_file_name, output_folder)
        print("cache_network is ", cache_network==True)
        print("network directory is ", check_file_exists(dataset_file_name, beta, NETWORK_CACHE))
        print(dataset_file_name,  os.path.basename(dataset_file_name).replace('.tsv', ''))
        if not (check_file_exists(dataset_file_name, beta, os.path.join(NETWORK_CACHE, 'permuted')) and cache_network): 
            script_path = self.make_network_files(dataset_file_name, network_file_name, output_folder, gene_index, edge_list, cores=cores, beta=beta, network_permutations=network_permutations, heat_permutations=heat_permutations, q=q)
            subprocess.run(['bash', script_path])
            permutation_names=os.path.join(output_folder, 'permuted', os.path.basename(dataset_file_name).replace('.tsv', '') + '_ppr_' + str(beta) + '_##NUM##.h5')
            network_file=os.path.join(output_folder, os.path.basename(dataset_file_name).replace('.tsv', '') + '_ppr_' + str(beta) + '.h5')
            script_path = self.write_bash(dataset_file_name, network_file_name, output_folder, gene_index, edge_list, network_file, permutation_names, cores=cores, beta=beta, network_permutations=network_permutations, heat_permutations=heat_permutations, q=q)
        else: 
            print("using cached network permutations")
            permutation_names=os.path.join(NETWORK_CACHE, 'permuted',  os.path.basename(dataset_file_name).replace('.tsv', '') + '_ppr_' + str(beta) + '_##NUM##.h5')
            network_file=os.path.join(NETWORK_CACHE, os.path.basename(dataset_file_name).replace('.tsv', '') + '_ppr_' + str(beta) + '.h5')
            script_path = self.write_bash(dataset_file_name, network_file_name, output_folder, gene_index, edge_list,network_file, permutation_names, cores=cores, beta=beta, network_permutations=network_permutations, heat_permutations=heat_permutations, q=q)
        subprocess.run(['bash', script_path])
        modules = self.get_modules(output_folder)
        bg_genes = get_network_genes(network_file_name)
        all_bg_genes = [bg_genes for x in modules]
        return modules, all_bg_genes
