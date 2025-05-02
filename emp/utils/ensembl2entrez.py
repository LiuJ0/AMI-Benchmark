import os
from src import constants
import pandas as pd

g2e_dict = None
e2g_dict = None
ensembl2entrez_dict = None
entrez2ensembl_dict = None

def load_gene_dictionary(gene_list_file_name, gene_list_path=None):
    if gene_list_path == None:
        gene_list_path = os.path.join(constants.dir_path,"data",gene_list_file_name)
    f = open(gene_list_path,'r')
    lines = [l.strip() for l in f]
    f.close()
    return lines


def get_entrez2ensembl_dictionary():
    lines_dict = load_gene_dictionary(constants.ENSEMBL_TO_ENTREZ)

    entrez2ensembl = {}
    for cur in lines_dict:
        splited_line = cur.split()
        if len(splited_line) != 2: continue
        if splited_line[0].find('.') > 0:
            limit = splited_line[0].find('.')
        else:
            limit = len(splited_line[0])
            entrez2ensembl[splited_line[1]] = splited_line[0][:limit]
    return entrez2ensembl

def get_ensembl2entrez_dictionary():

    lines_dict = load_gene_dictionary(constants.ENSEMBL_TO_ENTREZ)
    ensembl2gene_symbols = {}
    for cur in lines_dict:
        splited_line = cur.split()
        if len(splited_line) !=2: continue
        if splited_line[0].find('.') > 0:
            limit = splited_line[0].find('.')
        else:
            limit = len(splited_line[0])
        ensembl2gene_symbols[splited_line[0][:limit]] = splited_line[1]
    return ensembl2gene_symbols


def ensembl2entrez_convertor(e_ids):
    if e_ids[0].startswith("FBgn"):
        print("FB ids are being used")
        # Load the dataframe for FBgn ID conversion
        df = pd.read_csv("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/original/fly/fb2entrez.tsv", sep="\t")
        
        # Create a dictionary mapping FlyBase gene ID to NCBI gene ID
        fb2entrez_dict = dict(zip(df["FlyBase gene ID"], df["NCBI gene (formerly Entrezgene) ID"]))
        
        # Convert the input FBgn IDs to a pandas Series
        fb_series = pd.Series(e_ids)
        
        # Map the FBgn IDs to NCBI gene IDs using the dictionary
        results = fb_series.map(fb2entrez_dict).tolist()
        
        return results
    else:
        global ensembl2entrez_dict
        if ensembl2entrez_dict is None:
            ensembl2entrez_dict = get_ensembl2entrez_dictionary()
        results = []
        for cur in e_ids:
            if cur.split(".")[0] in ensembl2entrez_dict:
                results.append(ensembl2entrez_dict[cur.split(".")[0]])
        return results


def entrez2ensembl_convertor(entrez_ids):
    if str(entrez_ids[0]).startswith("FBgn"):
        # Load the dataframe for FBgn ID conversion
        df = pd.read_csv("/lab01/Projects/Jason_Projects/aneuploidy/ppi/data/original/fly/fb2entrez.tsv", sep="\t")
        
        # Create a dictionary mapping NCBI gene ID to FlyBase gene ID
        entrez2fb_dict = dict(zip(df["NCBI gene (formerly Entrezgene) ID"], df["FlyBase gene ID"]))
        
        # Convert the input entrez IDs to a pandas Series
        entrez_series = pd.Series(entrez_ids)
        
        # Map the entrez IDs to FlyBase gene IDs using the dictionary
        results = entrez_series.astype(str).map(entrez2fb_dict).tolist()
        
        return results
    else:
        global entrez2ensembl_dict
        if entrez2ensembl_dict is None:
            entrez2ensembl_dict = get_entrez2ensembl_dictionary()
        results = []
        for cur in entrez_ids:
            cur = str(cur)
            if cur in entrez2ensembl_dict:
                results.append(entrez2ensembl_dict[cur])
        return results
