# parse a directory (with some conditions) and return a compiled list of modules with the EHR, overlap, etc. 

import os
import pandas as pd

def read_modules(algo_dir, algo_name, dataset_name):
    modules_path = os.path.join(algo_dir, dataset_name + "_" + algo_name, "report")
    modules = []
    for file in os.listdir(modules_path):
        if file.startswith(algo_name + "_module_genes_"):
            modules.append([x.strip() for x in open(os.path.join(modules_path, file)).readlines()])
    return modules

def modules_to_dataframe(res_dir, algos, dataset):
    all_data = []
    
    for algo in algos:
        modules = read_modules(res_dir, algo, dataset)
        for module_index, genes in enumerate(modules):
            all_data.append({
                "algo": algo,
                "module_id": module_index,
                "genes": genes,
            })
    
    return pd.DataFrame(all_data)

def annotate_mEHR(df, mEHR_dir, algo_name, dataset_name):
    # Read the mEHR file
    mEHR_file = f"full_modules_report_{algo_name}_{dataset_name}.tsv"
    mEHR_path = os.path.join(mEHR_dir, mEHR_file)
    mEHR_df = pd.read_csv(mEHR_path, sep='\t')
    
    # Create a dictionary to map module identifiers to their annotations
    module_annotations = {}
    for _, row in mEHR_df.iterrows():
        module_id = row['Unnamed: 0']
        annotations = row.drop('Unnamed: 0').to_dict()
        module_annotations[module_id] = annotations
    
    # Function to get annotations for a specific module
    def get_annotations(row):
        module_id = f"{dataset_name}_{row['algo']}_module_{row['module_id']}"
        return module_annotations.get(module_id, {})
    
    # Apply annotations to the original dataframe
    annotation_df = df.apply(get_annotations, axis=1, result_type='expand')
    
    # Concatenate the original dataframe with the annotations
    return pd.concat([df, annotation_df], axis=1)

def annotate_mEHR_full(df, mEHR_dir, algo_names, dataset_name):
    dfs = []
    for algo_name in algo_names:
        dfs.append(annotate_mEHR(df[df['algo'] == algo_name], mEHR_dir, algo_name, dataset_name))
    return pd.concat(dfs)


def enhance_dataframe(df):
    # Convert genes column to sets if not already
    df['genes'] = df['genes'].apply(set)
    
    # 1. Add 'overlapping_genes' column
    def count_overlapping_genes(genes, all_genes):
        return sum(1 for other_genes in all_genes if genes != other_genes and genes.intersection(other_genes))
    
    all_genes = df['genes'].tolist()
    df['overlapping_genes'] = df['genes'].apply(lambda x: min(count_overlapping_genes(x, all_genes), len(x)))
    
    # 2. Add 'overlapping_modules' column
    df['module_id'] = df['algo'] + '_' + df['module_id'].astype(str)
    
    def find_overlapping_modules(index, genes, df):
        overlapping = []
        for idx, row in df.iterrows():
            if idx != index and genes.intersection(row['genes']):
                overlapping.append(row['module_id'])
        return ','.join(overlapping)
    
    df['overlapping_modules'] = df.apply(lambda row: find_overlapping_modules(row.name, row['genes'], df), axis=1)
    
    return df


algos = ["PAPER", "DOMINO", "FDRnet"]
res_dir = "/lab01/Projects/Jason_Projects/aneuploidy/ppi/test/test_07-21-2024_EMP-benchmark/true_solutions"
modules_report_dir = "/lab01/Projects/Jason_Projects/aneuploidy/ppi/test/test_07-21-2024_EMP-benchmark/report/module_cache_files"

illumina_df = modules_to_dataframe(res_dir, algos, 'gene_scores_illumina_v3')
print(illumina_df)
illumina_df = annotate_mEHR_full(illumina_df, modules_report_dir, algos, "gene_scores_illumina_v3")
illumina_df.to_csv("illumina_modules.csv", index=False)
enhance_dataframe(illumina_df)[['algo', 'module_id', 'EHR', 'module_size', 'overlapping_genes', 'overlapping_modules']].to_csv("illumina_modules_enhanced.csv", index=False)

tnfa_df = modules_to_dataframe(res_dir, algos, 'tnfa')
tnfa_df = annotate_mEHR_full(tnfa_df, modules_report_dir, algos, "tnfa")
tnfa_df.to_csv("tnfa_modules.csv", index=False)
enhance_dataframe(tnfa_df)[['algo', 'module_id', 'EHR', 'module_size', 'overlapping_genes', 'overlapping_modules']].to_csv("tnfa_modules_enhanced.csv", index=False)