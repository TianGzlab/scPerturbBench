import pandas as pd
import scanpy as sc
import numpy as np 
import warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def calculate_top20_de_genes(adata, output_csv, condition_column="condition", control_label="ctrl", top_n=20):
    """
    Calculate the top DE genes between each perturbation and control.

    Parameters:
    - adata: AnnData object containing the Perturb-seq data.
    - condition_column: str, the column in `adata.obs` with perturbation conditions.
    - control_label: str, label for the control condition.
    - top_n: int, number of top DE genes to return for each perturbation.

    Returns:
    - de_genes_dict: A dictionary where keys are perturbations and values are DataFrames of top DE genes.
    """
    # Check if the condition column exists
    if condition_column not in adata.obs:
        raise ValueError(f"Column '{condition_column}' not found in adata.obs.")
    
    # Ensure control_label exists in the condition column
    if control_label not in adata.obs[condition_column].unique():
        raise ValueError(f"Control label '{control_label}' not found in '{condition_column}'.")

    # Perform differential expression analysis
    sc.tl.rank_genes_groups(adata, groupby=condition_column, reference=control_label, method='wilcoxon')
    
    # Extract results for each perturbation
    results = []
    unique_conditions = [c for c in adata.obs[condition_column].unique() if c != control_label]
    
    trim = 0
    for perturbation in unique_conditions:
        # Get the top genes for this perturbation
        gene_names = adata.uns['rank_genes_groups']['names'][perturbation][:top_n]
        logfoldchanges = adata.uns['rank_genes_groups']['logfoldchanges'][perturbation][:top_n]
        pvals_adj = adata.uns['rank_genes_groups']['pvals_adj'][perturbation][:top_n]
        # Append to results
        for gene , logfc, pvals_adj in zip(gene_names, logfoldchanges, pvals_adj):
            results.append({
                "condition": perturbation,
                "gene": gene,
                "logFoldChange": logfc,
                "pvals_adj":pvals_adj
            })
        trim += 1
        if trim > 100:
            break
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Top 20 DE genes for all perturbations saved to {output_csv}")
    
    return results

"""
data_path ='/data2/lanxiang/data/Task2_data/DixitRegev2016.h5ad'
dat_name = "DixitRegev2016"
adata = sc.read_h5ad(data_path)
adata.X = adata.layers['counts']
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
calculate_top20_de_genes(adata, f"top20_de_genes_{dat_name}.csv", condition_column="condition", control_label="ctrl", top_n=20)


data_path ='/data2/lanxiang/data/Task2_data/NormanWeissman2019_filtered.h5ad'
dat_name = "NormanWeissman2019_filtered"
adata = sc.read_h5ad(data_path)
adata.X = adata.layers['counts']
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
calculate_top20_de_genes(adata, f"top20_de_genes_{dat_name}.csv", condition_column="condition", control_label="ctrl", top_n=20)


data_path ='/data2/lanxiang/data/Task2_data/Sunshine2023_CRISPRi_sarscov2.h5ad'
dat_name = "Sunshine2023_CRISPRi_sarscov2"
adata = sc.read_h5ad(data_path)
adata.X = adata.layers['counts']
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
calculate_top20_de_genes(adata, f"top20_de_genes_{dat_name}.csv", condition_column="condition", control_label="ctrl", top_n=20)
"""


# read all the h5ad files in this folder: /data2/lanxiang/data/Task1_data/, and then use for loop to get the top20 DE genes for each file
import os
data_path ='/data2/lanxiang/data/Task1_data/'
for file in os.listdir(data_path):
    if file.endswith('.h5ad'):
        dat_name = file.split('.')[0]
        print(f"Processing {dat_name}...")
        adata = sc.read_h5ad(data_path+file)
        adata.X = adata.layers['counts']
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        calculate_top20_de_genes(adata, f"task1/top20_de_genes_{dat_name}.csv", condition_column="condition", control_label="ctrl", top_n=20)
        print(f"top20_de_genes_{dat_name}.csv has been saved")