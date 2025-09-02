import os
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, wasserstein_distance
from scipy.spatial.distance import cdist
from warnings import catch_warnings, simplefilter

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def compute_mmd(pred_array, truth_array, kernel='rbf', gamma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between predicted and true distributions
    
    Args:
        pred_array (numpy.ndarray): Predicted values
        truth_array (numpy.ndarray): Ground truth values  
        kernel (str): Kernel type, currently supports 'rbf'
        gamma (float): RBF kernel parameter
    
    Returns:
        float: MMD value
    """
    if kernel == 'rbf':
        dist_pred = cdist(pred_array, pred_array, metric='sqeuclidean')
        dist_truth = cdist(truth_array, truth_array, metric='sqeuclidean')
        dist_cross = cdist(pred_array, truth_array, metric='sqeuclidean')
        Kxx = np.exp(-gamma * dist_pred)
        Kyy = np.exp(-gamma * dist_truth)
        Kxy = np.exp(-gamma * dist_cross)
        return np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)
    else:
        raise ValueError("Unsupported kernel type")


def compute_metrics_single_cell(pred_mean, true_mean, ctrl_mean):
    """
    Compute metrics for single-cell (mean) data
    
    Args:
        pred_mean (numpy.ndarray): Predicted mean expression (1D array)
        true_mean (numpy.ndarray): True mean expression (1D array)
        ctrl_mean (numpy.ndarray): Control mean expression (1D array)
    
    Returns:
        dict: Dictionary containing all computed metrics
    """
    metrics = {}
    
    # Compute correlation and regression metrics with error handling
    with catch_warnings():
        simplefilter("ignore")
        try:
            metrics['R_squared'] = r2_score(true_mean, pred_mean)
            metrics['R_squared_delta'] = r2_score(true_mean - ctrl_mean, pred_mean - ctrl_mean)
        except:
            metrics['R_squared'] = np.nan
            metrics['R_squared_delta'] = np.nan
        try:
            metrics['Pearson_Correlation'] = pearsonr(true_mean, pred_mean)[0]
            metrics['Pearson_Correlation_delta'] = pearsonr(true_mean - ctrl_mean, pred_mean - ctrl_mean)[0]
        except:
            metrics['Pearson_Correlation'] = np.nan
            metrics['Pearson_Correlation_delta'] = np.nan

    # Compute distance and similarity metrics
    metrics.update({
        'MSE': mean_squared_error(true_mean, pred_mean),
        'RMSE': np.sqrt(mean_squared_error(true_mean, pred_mean)),
        'MAE': mean_absolute_error(true_mean, pred_mean),
        'Cosine_Similarity': cosine_similarity([true_mean], [pred_mean])[0, 0],
        'L2': np.linalg.norm(true_mean - pred_mean),
        'MSE_delta': mean_squared_error(true_mean - ctrl_mean, pred_mean - ctrl_mean),
        'RMSE_delta': np.sqrt(mean_squared_error(true_mean - ctrl_mean, pred_mean - ctrl_mean)),
        'MAE_delta': mean_absolute_error(true_mean - ctrl_mean, pred_mean - ctrl_mean),
        'Cosine_Similarity_delta': cosine_similarity([(true_mean - ctrl_mean)], [(pred_mean - ctrl_mean)])[0, 0],
        'L2_delta': np.linalg.norm((true_mean - ctrl_mean) - (pred_mean - ctrl_mean)),
    })

    # For single-cell data, MMD and Wasserstein distance cannot be computed, set to NaN
    metrics['MMD'] = np.nan
    metrics['Wasserstein'] = np.nan

    return metrics


def compute_metrics(pred_data, true_data, ctrl_data):
    """
    Compute metrics for multi-cell data (maintained for compatibility)
    
    Args:
        pred_data (numpy.ndarray or scipy.sparse): Predicted expression data
        true_data (numpy.ndarray or scipy.sparse): True expression data
        ctrl_data (numpy.ndarray or scipy.sparse): Control expression data
    
    Returns:
        dict: Dictionary containing all computed metrics
    """
    # Convert sparse matrices to dense if needed
    pred_data = pred_data.toarray() if not isinstance(pred_data, np.ndarray) else pred_data
    true_data = true_data.toarray() if not isinstance(true_data, np.ndarray) else true_data
    ctrl_data = ctrl_data.toarray() if not isinstance(ctrl_data, np.ndarray) else ctrl_data

    # Calculate mean expression across cells
    mean_true = np.mean(true_data, axis=0)
    mean_pred = np.mean(pred_data, axis=0)
    mean_ctrl = np.mean(ctrl_data, axis=0)

    metrics = {}
    
    # Compute correlation and regression metrics with error handling
    with catch_warnings():
        simplefilter("ignore")
        try:
            metrics['R_squared'] = r2_score(mean_true, mean_pred)
            metrics['R_squared_delta'] = r2_score(mean_true - mean_ctrl, mean_pred - mean_ctrl)
        except:
            metrics['R_squared'] = np.nan
            metrics['R_squared_delta'] = np.nan
        try:
            metrics['Pearson_Correlation'] = pearsonr(mean_true, mean_pred)[0]
            metrics['Pearson_Correlation_delta'] = pearsonr(mean_true - mean_ctrl, mean_pred - mean_ctrl)[0]
        except:
            metrics['Pearson_Correlation'] = np.nan
            metrics['Pearson_Correlation_delta'] = np.nan

    # Compute distance and similarity metrics
    metrics.update({
        'MSE': mean_squared_error(mean_true, mean_pred),
        'RMSE': np.sqrt(mean_squared_error(mean_true, mean_pred)),
        'MAE': mean_absolute_error(mean_true, mean_pred),
        'Cosine_Similarity': cosine_similarity([mean_true], [mean_pred])[0, 0],
        'L2': np.linalg.norm(mean_true - mean_pred),
        'MSE_delta': mean_squared_error(mean_true - mean_ctrl, mean_pred - mean_ctrl),
        'RMSE_delta': np.sqrt(mean_squared_error(mean_true - mean_ctrl, mean_pred - mean_ctrl)),
        'MAE_delta': mean_absolute_error(mean_true - mean_ctrl, mean_pred - mean_ctrl),
        'Cosine_Similarity_delta': cosine_similarity([(mean_true - mean_ctrl)], [(mean_pred - mean_ctrl)])[0, 0],
        'L2_delta': np.linalg.norm((mean_true - mean_ctrl) - (mean_pred - mean_ctrl)),
    })

    # Compute MMD and Wasserstein distance for each gene
    n_genes = pred_data.shape[1]
    mmd_vals = []
    ws_vals = []
    for g in range(n_genes):
        pred_gene = pred_data[:, g].reshape(-1, 1)
        true_gene = true_data[:, g].reshape(-1, 1)
        mmd_vals.append(compute_mmd(pred_gene, true_gene))
        ws_vals.append(wasserstein_distance(pred_gene.flatten(), true_gene.flatten()))

    metrics['MMD'] = np.mean(mmd_vals)
    metrics['Wasserstein'] = np.mean(ws_vals)

    return metrics


def main():
    """Main evaluation pipeline for BioLord predictions"""
    
    # Configuration paths
    Biolord_dir = "/data2/lanxiang/perturb_benchmark_v2/model/Biolord"
    raw_data_dir = "/data2/lanxiang/data/Task1_data"
    output_root = "/data2/lanxiang/perturb_benchmark_v2/Metrics/task1"

    # Define output paths for different DEG subsets
    output_paths = {
        "allgene": os.path.join(output_root, "Allgene", "Biolord_all_metrics.csv"),
        "top20": os.path.join(output_root, "Top20", "Biolord_top20_metrics.csv"),
        "top50": os.path.join(output_root, "Top50", "Biolord_top50_metrics.csv"),
        "top100": os.path.join(output_root, "Top100", "Biolord_top100_metrics.csv")
    }

    # Create output directories if they don't exist
    for path in output_paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Get DEG file directory
    deg_dir = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'

    # Initialize results storage
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }

    # Process each dataset and compute metrics
    for dataset in os.listdir(Biolord_dir):
        dataset_path = os.path.join(Biolord_dir, dataset)
        raw_file = os.path.join(raw_data_dir, f"{dataset}.h5ad")
        
        if not os.path.exists(raw_file):
            print(f"❌ Raw data missing: {raw_file}")
            continue

        adata = sc.read_h5ad(raw_file)
        print(f"▶ Processing {dataset}")

        # Process each condition file in the dataset
        for file in os.listdir(dataset_path):
            if not file.endswith("_pred.csv"):
                continue
                
            condition = file.replace("_pred.csv", "")
            pred_file = os.path.join(dataset_path, f"{condition}_pred.csv")
            real_file = os.path.join(dataset_path, f"{condition}_real.csv")

            if not os.path.exists(real_file):
                print(f"❌ Missing real data for {condition} in {dataset}")
                continue

            try:
                # Read prediction and real data (new format: first row as row name, first column as column name)
                pred_df = pd.read_csv(pred_file, index_col=0)
                true_df = pd.read_csv(real_file, index_col=0)
                
                # Get gene expression values (should only have one row of data)
                if pred_df.shape[0] != 1 or true_df.shape[0] != 1:
                    print(f"⚠️  Warning: Expected 1 row but got pred: {pred_df.shape[0]}, true: {true_df.shape[0]} for {dataset}/{condition}")
                
                pred_mean = pred_df.iloc[0].values  # Get first row as mean values
                true_mean = true_df.iloc[0].values

                # Get control group data mean
                ctrl_mask = adata.obs["condition"] == "ctrl"
                ctrl_data = adata[ctrl_mask].X
                ctrl_data = ctrl_data.toarray() if not isinstance(ctrl_data, np.ndarray) else ctrl_data
                ctrl_mean = np.mean(ctrl_data, axis=0)

                # Ensure gene order consistency
                pred_genes = pred_df.columns.tolist()
                true_genes = true_df.columns.tolist()
                adata_genes = adata.var_names.tolist()
                
                # Find common genes
                common_genes = list(set(pred_genes) & set(true_genes) & set(adata_genes))
                if len(common_genes) == 0:
                    print(f"❌ No common genes found for {dataset}/{condition}")
                    continue
                
                # Reorder to ensure gene order consistency
                pred_indices = [pred_genes.index(g) for g in common_genes]
                true_indices = [true_genes.index(g) for g in common_genes]
                ctrl_indices = [adata_genes.index(g) for g in common_genes]
                
                pred_mean_aligned = pred_mean[pred_indices]
                true_mean_aligned = true_mean[true_indices]
                ctrl_mean_aligned = ctrl_mean[ctrl_indices]

                # Compute all-gene metrics
                pert_metrics = compute_metrics_single_cell(pred_mean_aligned, true_mean_aligned, ctrl_mean_aligned)
                all_results["allgene"].append({
                    "dataset": dataset,
                    "condition": condition,
                    "model": "Biolord",
                    **pert_metrics
                })

                # Compute Top20, Top50, Top100 DEG metrics
                for top_n in [20, 50, 100]:
                    deg_file = os.path.join(deg_dir, dataset, f"DEG_{condition}.csv")
                    if not os.path.exists(deg_file):
                        print(f"❌ Missing DEG file: {deg_file}")
                        continue

                    # Load top differential expression genes
                    deg_df = pd.read_csv(deg_file)
                    top_degs = deg_df.head(top_n)["names"].tolist()
                    
                    # Find top DEGs that are in common_genes
                    top_degs_available = [g for g in top_degs if g in common_genes]
                    if len(top_degs_available) == 0:
                        print(f"❌ No top DEGs available for {dataset}/{condition} top{top_n}")
                        continue
                    
                    # Get indices of top DEGs
                    top_pred_indices = [common_genes.index(g) for g in top_degs_available]
                    
                    # Extract data for top DEGs
                    top_pred_data = pred_mean_aligned[top_pred_indices]
                    top_true_data = true_mean_aligned[top_pred_indices]
                    top_ctrl_data = ctrl_mean_aligned[top_pred_indices]

                    # Compute metrics for top DEGs
                    top_metrics = compute_metrics_single_cell(top_pred_data, top_true_data, top_ctrl_data)
                    all_results[f"top{top_n}"].append({
                        "dataset": dataset,
                        "condition": condition,
                        "model": "Biolord",
                        **top_metrics
                    })

            except Exception as e:
                print(f"❌ Failed on {dataset} / {condition}: {e}")
                import traceback
                traceback.print_exc()

    # Save all results
    for key, result in all_results.items():
        df = pd.DataFrame(result)
        df.to_csv(output_paths[key], index=False)
        print(f"✅ {key} metrics saved to: {output_paths[key]} (Total: {len(result)} records)")


if __name__ == "__main__":
    main()
