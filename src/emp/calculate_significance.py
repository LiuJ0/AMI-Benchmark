import sys
sys.path.insert(0, '../..')

import src.constants as constants

from functools import reduce
import argparse
import os
import pandas as pd
import numpy as np
import random
from statsmodels.sandbox.stats.multicomp import fdrcorrection0
from scipy.stats import norm
from scipy.optimize import minimize
import scipy.special as sp
from scipy.stats import genpareto
import ast

def estimate_gpd_pvalue(observed, data, threshold_percentile=90):
    """
    Estimate p-value using Generalized Pareto Distribution for tail estimation.
    
    Uses Peaks Over Threshold (POT) method:
    1. Set threshold u at the specified percentile
    2. Fit GPD to excesses over u
    3. Estimate tail probability P(X > z) for observed value z
    
    Parameters:
    observed: observed value to test
    data: sample data
    threshold_percentile: percentile for threshold u (default 90)
    
    Returns:
    p-value estimate
    """
    data = np.array(data, dtype=np.float64)
    z = float(observed)
    
    # 1. Determine threshold u (e.g., 90th percentile)
    u = np.percentile(data, threshold_percentile)
    
    # 2. Extract excesses over threshold
    excesses = data[data > u] - u
    
    if len(excesses) < 10:
        return 1.0
    
    # 3. Fit Generalized Pareto Distribution to excesses
    # genpareto in scipy uses shape parameter c (our xi) and scale
    try:
        # Fit GPD using MLE
        shape, loc, scale = genpareto.fit(excesses, floc=0)  # loc=0 for excesses
        
        # 4. Estimate tail probability
        if z <= u:
            return 1.0
        else:
            # Above threshold - use GPD extrapolation
            # P(X > z) = P(X > u) * P(X > z | X > u)
            # P(X > u) ≈ n_u / n where n_u is number of exceedances
            p_exceed_u = len(excesses) / len(data)
            
            # P(X > z | X > u) = P(Y > z - u) where Y ~ GPD
            # This is the survival function of GPD at (z - u)
            excess_z = z - u
            p_exceed_given_u = genpareto.sf(excess_z, shape, loc=0, scale=scale)
            
            p_val = p_exceed_u * p_exceed_given_u
            
    except:
        print('fitting failed')
        return 1.0
    
    return min(1.0, max(0.0, p_val))

def calc_new_pval(cur_rv, cur_dist):
    """
    Calculate p-value using different statistical approaches.
    
    Parameters:
    cur_rv: real enrichment value(s) (z score, typically -log10(HG pval))
    cur_dist: distribution from permutations
    method: one of 'chebyshev', 'chernoff', 'exponential', 'normal'
    compute_mgf: if True, compute and print empirical MGF values
    
    Returns:
    String representation of p-value(s)
    """
    cur_dist = np.array(cur_dist, np.float32)
    
    # Compute mean and variance/std from permutation distribution
    mu = np.mean(cur_dist)
    sigma_sq = np.var(cur_dist, ddof=1)  # Sample variance
    sigma = np.sqrt(sigma_sq)  # Standard deviation
    
    # Handle the real value(s)
    if type(cur_rv) == str:
        z_scores = np.array(cur_rv[1:-1].split(", "), dtype=np.float32)
    else:
        z_scores = np.array([cur_rv], dtype=np.float32)
    
    new_pvals = []
    
    for z in z_scores:
        diff = z - mu
        abs_diff = np.abs(diff)
        data = cur_dist
        u = np.percentile(data, 90)
        if z > u:
            excesses = data[data > u] - u
            try:
                shape, loc, scale = genpareto.fit(excesses, floc=0)
                # Calculate p-value
                z_val = float(z)
                p_exceed_u = len(excesses) / len(data)
                excess_z = z_val - u
                p_exceed_given_u = max(genpareto.sf(excess_z, shape, loc=0, scale=scale), 1e-10)
                p_val = p_exceed_u * p_exceed_given_u
                new_pval = min(1.0, max(0.0, p_val))
            except:
                new_pval = 1.0
        else:
            new_pval = len(data[data >= z]) / len(data)
        new_pvals.append(new_pval)
    
    return str(new_pvals)

def calculate_sig_new(algo_sample=None, dataset_sample=None, n_dist_samples=300, 
                      n_total_samples=None, limit=10000, 
                      md_path=None, dist_path=None, filtered_go_ids_file="", 
                      hg_th=0.0001, alpha=0.05):
    
    filtered_go_ids = open(filtered_go_ids_file, 'r').read().split() + [constants.ROOT_GO_ID]
    
    try:
        output_md = pd.read_csv(md_path.format(dataset_sample, algo_sample), 
                                sep='\t', index_col=0).reindex(filtered_go_ids).dropna()
    except Exception:
        return None
    

    # Step 1: Identify HG-enriched terms
    max_genes_pvals = np.power(10, -output_md.loc[:, "hg_pval_max"])
    print(f"total n_genes with pval less than one: {np.size(max_genes_pvals)}/{len(filtered_go_ids)}")
    max_genes_pvals = np.append(max_genes_pvals, 
                                 np.ones(len(filtered_go_ids) - np.size(max_genes_pvals)))
    
    fdr_results = fdrcorrection0(max_genes_pvals, alpha=hg_th, method='indep', is_sorted=False)
    n_hg_true = len([cur for cur in fdr_results[0] if cur == True])
    HG_CUTOFF = (np.sort(max_genes_pvals)[n_hg_true-1] if n_hg_true > 0 else -1)
    print(f"HG cutoff: {HG_CUTOFF}, (ES={-np.log10(HG_CUTOFF)}, n={n_hg_true})")
    
    hg_enriched_mask = output_md.loc[:, "hg_pval_max"].values >= -np.log10(HG_CUTOFF)
    
    # Step 2: Load permutation distributions
    print(dist_path.format(dataset_sample, algo_sample))
    output = pd.read_csv(dist_path.format(dataset_sample, algo_sample),
                        sep='\t', index_col=0).dropna()
    
    output = output.loc[output_md.index.values, :]
    
    # Sample from permutations
    n_total_samples = n_total_samples if n_total_samples is not None else \
                      len(output.iloc[0].loc["dist_n_samples"][1:-1].split(", "))
    np.random.seed(int(random.random()*1000))
    i_choice = np.random.choice(n_total_samples, n_dist_samples, replace=False)
    i_dist = i_choice[:n_dist_samples]
    
    # Step 3: Calculate new p-values
    counter = 0
    new_pvals_list = []

    for index, cur in output.iterrows():
        if counter == limit:
            break
        pval_dist = np.array([float(x) for x in cur["dist_n_samples"][1:-1].split(", ")])[i_dist]
        new_pval_str = calc_new_pval(cur["hg_pval"], pval_dist)
        new_pvals_list.append(new_pval_str)
        output_md.loc[index, 'new_pval'] = new_pval_str
        counter += 1
    
    mask_ids = output.index.values
    
    # Convert to numeric array for statistical testing
    new_pvals_mat = [np.array([x]) if type(x) != str else 
                     np.array(x[1:-1].split(", ")).astype(np.float32)
                     for x in new_pvals_list]
    
    n_modules = 0
    if len(new_pvals_mat) != 0:
        n_modules = new_pvals_mat[0].shape[0]
    
    # Get min (best) p-value per term across modules
    min_new_pvals = np.array([np.min(x) for x in new_pvals_mat])
    
    # Step 4: Apply correction
    benj_hoch_results = fdrcorrection0(min_new_pvals, alpha=alpha, method='indep', is_sorted=False)[0]

    n_benj_true = np.sum(benj_hoch_results)
    
    print(f"# terms passing multiple correction: {n_benj_true}")
    
    # Step 5: Intersect with HG-enriched terms
    final_mask = benj_hoch_results & hg_enriched_mask[:len(benj_hoch_results)]
    n_final = np.sum(final_mask)
    
    # Extract module-level results
    mask_terms_modules = []
    for idx, is_sig in enumerate(final_mask):
        if is_sig:
            term_pvals = new_pvals_mat[idx]
            module_pass = term_pvals <= alpha
            mask_terms_modules.append(module_pass)
        else:
            mask_terms_modules.append(np.array([False] * n_modules))
    
    go_ids_result = output.index.values[final_mask]
    go_names_result = output["GO name"].values[final_mask] if "GO name" in output.columns else np.array([])
    
    print(f"# terms passing multiple correction and HG cutoff: {n_final}")
    print(f"EHR: {n_final/float(n_hg_true) if n_hg_true > 0 else 0}")

    # Convert to numeric array for statistical testing
    new_pvals_mat = [np.array([x]) if type(x) != str else 
                     np.array(x[1:-1].split(", ")).astype(np.float32)
                     for x in new_pvals_list]
    
    n_modules = 0
    if len(new_pvals_mat) != 0:
        n_modules = new_pvals_mat[0].shape[0]
    
    # Get min (best) p-value per term across modules
    min_new_pvals = np.array([np.min(x) for x in new_pvals_mat])
    
    
    n_benj_true = np.sum(benj_hoch_results)
    

    # Step 5: Intersect with HG-enriched terms
    final_mask = benj_hoch_results & hg_enriched_mask[:len(benj_hoch_results)]
    n_final = np.sum(final_mask)
    
    # Extract module-level results
    mask_terms_modules = []
    for idx, is_sig in enumerate(final_mask):
        if is_sig:
            term_pvals = new_pvals_mat[idx]
            module_pass = term_pvals <= alpha
            mask_terms_modules.append(module_pass)
        else:
            mask_terms_modules.append(np.array([False] * n_modules))
    
    go_ids_result = output.index.values[final_mask]
    go_names_result = output["GO name"].values[final_mask] if "GO name" in output.columns else np.array([])
    
    # Store results
    output_md.loc[mask_ids, "new_pval_max"] = min_new_pvals
    output_md.loc[mask_ids, "passed_bonferroni_holm"] = benj_hoch_results
    output_md.loc[mask_ids, "passed_final_filter"] = final_mask
    
    return (n_benj_true, n_final, HG_CUTOFF, n_hg_true, 
            go_ids_result, go_names_result, mask_ids, mask_terms_modules, 
            new_pvals_mat, min_new_pvals, output.index.values)





def main():
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--dataset_file', dest='dataset_file', default=constants.config_json["dataset_file"])
    parser.add_argument('--algo', dest='algo', default=constants.config_json["algo"])
    parser.add_argument('--report_folder', dest='report_folder', default=constants.config_json["report_folder"])
    parser.add_argument('--n_total_samples', help="n_total_samples", dest='n_total_samples', default=constants.config_json["n_total_samples"])
    parser.add_argument('--n_dist_samples', help="n_dist_samples", dest='n_dist_samples', default=constants.config_json["n_dist_samples"])
    parser.add_argument('--filtered_go_ids_file', dest='filtered_go_ids_file', default=constants.config_json["filtered_go_ids_file"])
    parser.add_argument('--hg_th', dest='hg_th', default=constants.config_json["hg_th"])
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.05,
                        help="Significance level for multiple testing correction")

    args = parser.parse_args()
    dataset_file = args.dataset_file
    algo = args.algo
    n_total_samples = int(args.n_total_samples)
    n_dist_samples = int(args.n_dist_samples)
    report_folder = args.report_folder
    filtered_go_ids_file = args.filtered_go_ids_file
    hg_th = float(args.hg_th)

    dataset_name = os.path.splitext(os.path.split(dataset_file)[1])[0]
    print(f"about to calculate pval for {dataset_name}_{algo}")

    md_path = os.path.join(report_folder, "emp_diff_modules_{}_{}_md.tsv")
    dist_path = os.path.join(report_folder, "emp_diff_modules_{}_{}.tsv")
    
    # Store results for each method
    method_results = {}
    print_str = ""
    res = calculate_sig_new(algo_sample=algo, dataset_sample=dataset_name, 
                                n_dist_samples=n_dist_samples, n_total_samples=n_total_samples, 
                                md_path=md_path, dist_path=dist_path, 
                                filtered_go_ids_file=filtered_go_ids_file, 
                                hg_th=hg_th, alpha=args.alpha)
    
    # Save final method results IN ORIGINAL FORMAT
    if res is not None:
        (n_benj_true, n_final, HG_CUTOFF, n_hg_true, 
         go_ids_result, go_names_result, mask_ids, mask_terms_modules, 
         new_pvals_mat, _, _) = res

        output_md = pd.read_csv(
            os.path.join(report_folder, f"emp_diff_modules_{dataset_name}_{algo}_md.tsv"),
            sep='\t', index_col=0)

        output_md.loc[mask_ids, "passed_new_permutation_test"] = "[False]"
        output_md.loc[mask_ids, "passed_new_permutation_test"] = [str(list(a)) for a in mask_terms_modules]

        output_md.loc[mask_ids, 'new_pval'] = [str(a) for a in new_pvals_mat]
        output_md.loc[mask_ids, 'new_pval_max'] = [np.max(a) for a in new_pvals_mat]

        output_md.to_csv(
            os.path.join(report_folder, f"emp_diff_modules_{dataset_name}_{algo}_new_method.tsv"),
            sep='\t')

if __name__ == "__main__":
    main()