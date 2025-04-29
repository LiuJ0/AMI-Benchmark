# Evaluation and Aggregation of Active Module Identification Algorithms
This repository contains the files used in the study "Evaluation and Aggregation of Active Module Identification Algorithms."
## Set up
To run this code, you should run EMP or your algorithms and organize them according to the instructions in [EMP Benchmark](https://github.com/Shamir-Lab/EMP-benchmark?tab=readme-ov-file#set-your-directory-structure). The `i`-th module of algorithm `algo` and dataset `dataset` should be placed into:
```
{dataset}_{algo}/report/{algo}_module_genes_{i}.txt
```
(If you decide to run EMP, your modules will automatically be formatted in this way.)

This code was tested with Python 3.10.12. We recommend using a virtual environment and installing the dependencies with 
```
pip install -r requirements.txt
```

## Overview
- `earth_movers.py`: code for computing Earth Mover's Distance (EMD). Use `emd_subgraph_similarity()`. 
- `group_modules.py`: code for spectral aggregation, used in section 2.4.2
- `merge_modules.py`: an implementation of the merging algorithm (Algorithm 1), described in section 2.4.3
- `solution_similarity.py`: code to generate similarity heatmaps (Figure 6).
- `visualize_clusters.py`: code to generate Figure 9. 
- `visualize_module_distribution.py`: generates a distribution of module sizes summed across datasets (Figure 3).
- `visualize_modules.py`: code to generate pairs of similar modules (Figure 5)