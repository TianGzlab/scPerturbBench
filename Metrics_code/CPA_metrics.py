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


def compute_metrics(pred_data, true_data, ctrl_data):
    """
    Compute comprehensive evaluation metrics comparing predictions to ground truth
    
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
    """Main evaluation pipeline for CPA"""
    
    # Configuration paths
    seen_base = "/data2/lanxiang/perturb_benchmark_v2/model/CPA/Task2/Seen"
    raw_base = "/data2/lanxiang/perturb_benchmark_v2/model/CPA"
    output_root = "/data2/lanxiang/perturb_benchmark_v2/Metrics/task2"
    deg_dir = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'

    # Define output paths for different DEG subsets
    output_paths = {
        "allgene": os.path.join(output_root, "Allgene", "CPA_Task2_all_metrics.csv"),
        "top20": os.path.join(output_root, "Top20", "CPA_Task2_top20_metrics.csv"),
        "top50": os.path.join(output_root, "Top50", "CPA_Task2_top50_metrics.csv"),
        "top100": os.path.join(output_root, "Top100", "CPA_Task2_top100_metrics.csv")
    }

    # Create output directories if they don't exist
    for path in output_paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Initialize results storage
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }

    # Process each seen split and dataset
    for seen_split in ["seen0", "seen1", "seen2"]:
        seen_path = os.path.join(seen_base, seen_split)
        
        for dataset in os.listdir(seen_path):
            csv_dir = os.path.join(seen_path, dataset)
            h5ad_path = os.path.join(raw_base, dataset, f"{dataset}_pred.h5ad")
            
            if not os.path.exists(h5ad_path):
                print(f"❌ H5AD file missing: {h5ad_path}")
                continue
                
            # Load and preprocess reference data
            adata = sc.read_h5ad(h5ad_path)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            # Process each condition CSV file
            for csv_file in os.listdir(csv_dir):
                if not csv_file.endswith(".csv"): 
                    continue
                    
                condition = csv_file.replace(".csv", "")
                
                try:
                    # Load and preprocess prediction data
                    df_pred = pd.read_csv(os.path.join(csv_dir, csv_file), index_col=0)
                    pred_data = df_pred.values
                    
                    # Apply same normalization as reference data
                    pred_data = pred_data / pred_data.sum(axis=1, keepdims=True) * 1e4
                    pred_data = np.log1p(pred_data)

                    # Extract true and control data from reference
                    true_mask = (adata.obs["condition"] == condition)
                    ctrl_mask = (adata.obs["condition"] == "ctrl")

                    true_data = adata[true_mask].X
                    ctrl_data = adata[ctrl_mask].X

                    # Align sample sizes for fair comparison
                    min_samples = min(pred_data.shape[0], true_data.shape[0], ctrl_data.shape[0])
                    sample_idx = np.random.choice(min_samples, min(500, min_samples), replace=False)
                    
                    pred_data = pred_data[sample_idx]
                    true_data = true_data[sample_idx]
                    ctrl_data = ctrl_data[sample_idx]

                    # Compute all-gene metrics
                    metrics = compute_metrics(pred_data, true_data, ctrl_data)
                    all_results["allgene"].append({
                        "seen_split": seen_split,
                        "dataset": dataset,
                        "condition": condition,
                        "model": "CPA",
                        **metrics
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

                        # Extract data for top DEGs only
                        top_pred_data = pred_data[:, [adata.var_names.get_loc(g) for g in top_degs]]
                        top_true_data = true_data[:, [adata.var_names.get_loc(g) for g in top_degs]]
                        top_ctrl_data = ctrl_data[:, [adata.var_names.get_loc(g) for g in top_degs]]

                        # Compute metrics for top DEGs
                        top_metrics = compute_metrics(top_pred_data, top_true_data, top_ctrl_data)
                        all_results[f"top{top_n}"].append({
                            "seen_split": seen_split,
                            "dataset": dataset,
                            "condition": condition,
                            "model": "CPA",
                            **top_metrics
                        })
                        
                except Exception as e:
                    print(f"❌ Failed: {seen_split}/{dataset}/{condition} | {e}")

    # Save all results
    for key, result in all_results.items():
        df = pd.DataFrame(result)
        df.to_csv(output_paths[key], index=False)
        print(f"✅ {key} metrics saved to: {output_paths[key]}")


if __name__ == "__main__":
    main()