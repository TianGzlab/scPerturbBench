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
    mmd_vals = []
    ws_vals = []
    for g in range(pred_data.shape[1]):
        pred_gene = pred_data[:, g].reshape(-1, 1)
        true_gene = true_data[:, g].reshape(-1, 1)
        mmd_vals.append(compute_mmd(pred_gene, true_gene))
        ws_vals.append(wasserstein_distance(pred_gene.flatten(), true_gene.flatten()))

    metrics['MMD'] = np.mean(mmd_vals)
    metrics['Wasserstein'] = np.mean(ws_vals)

    return metrics


def main():
    """Main evaluation pipeline for Task 3 baseline methods"""
    
    # Task 3 specific datasets with cell-type information
    task3_datasets = [
        'Burkhardt_sub10', 'Haber', 'Hagai', 'Kang', 'Weinreb_time',
        'Perturb_cmo_V1_sub10', 'Perturb_KHP_sub10',
        'Srivatsan_sciplex3_sub10', 'Tahoe100_sub10', "Parse_10M_PBMC_sub10"
    ]

    # Configuration paths
    h5ad_root = "/data2/lanxiang/data/Task3_data"
    pred_root = "/data2/lanxiang/perturb_benchmark_v2/model/Baseline"
    deg_root = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    output_root = "/data2/lanxiang/perturb_benchmark_v2/Metrics/task3"

    # Define output paths for different DEG subsets
    output_paths = {
        "allgene": os.path.join(output_root, "Allgene", "Baseline3_all_metrics.csv"),
        "top20": os.path.join(output_root, "Top20", "Baseline3_top20_metrics.csv"),
        "top50": os.path.join(output_root, "Top50", "Baseline3_top50_metrics.csv"),
        "top100": os.path.join(output_root, "Top100", "Baseline3_top100_metrics.csv")
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

    # Process each Task 3 dataset
    for dataset in task3_datasets:
        h5ad_path = os.path.join(h5ad_root, f"{dataset}.h5ad")
        if not os.path.exists(h5ad_path):
            print(f"❌ Data file not found: {h5ad_path}")
            continue

        adata = sc.read_h5ad(h5ad_path)
        print(f"🔍 Processing dataset: {dataset}")
        all_cell_types = adata.obs['cell_type'].unique()

        # Process each cell type within the dataset
        for cell_type in all_cell_types:
            # Extract control data for this cell type
            ctrl_mask = (adata.obs['condition'] == 'ctrl') & (adata.obs['cell_type'] == cell_type)
            ctrl_data = adata[ctrl_mask].X
            
            if ctrl_data.shape[0] == 0:
                print(f"⚠️ No control data found, skipping: {dataset}, cell_type={cell_type}")
                continue

            # Find prediction folder for this cell type
            pred_folder = os.path.join(pred_root, dataset, cell_type)
            if not os.path.isdir(pred_folder):
                continue

            # Process each condition in this cell type
            for fname in os.listdir(pred_folder):
                if not fname.endswith("_predicted_values.npy"):
                    continue

                base_name = fname.replace("_predicted_values.npy", "")
                pred_path = os.path.join(pred_folder, fname)
                true_path = os.path.join(pred_folder, f"{base_name}_true_values.npy")

                if not os.path.exists(true_path):
                    continue

                # Load prediction and true data
                pred = np.load(pred_path, allow_pickle=True)
                true = np.load(true_path, allow_pickle=True)

                # Handle sparse matrix formats and check for NaN values
                if not isinstance(pred, np.ndarray):
                    pred = pred.toarray()
                if not isinstance(true, np.ndarray):
                    true = true.toarray()

                min_n = min(pred.shape[0], true.shape[0], ctrl_data.shape[0])
                if min_n == 0:
                    print(f"⚠️ Empty data, skipping: {dataset}, {cell_type}, {base_name}")
                    continue

                # Sample data for computational efficiency (max 500 cells)
                n_sample = min(500, min_n)
                idx = np.random.choice(min_n, n_sample, replace=False)

                pred = pred[idx]
                true = true[idx]
                ctrl_sampled = ctrl_data[idx]
                
                if not isinstance(ctrl_sampled, np.ndarray):
                    ctrl_sampled = ctrl_sampled.toarray()

                # Skip if data contains NaN values
                if (np.isnan(pred).any() or np.isnan(true).any() or np.isnan(ctrl_sampled).any()):
                    print(f"❌ Contains NaN values, skipping: {dataset}, {cell_type}, {base_name}")
                    continue

                # Compute all-gene metrics
                all_metrics = compute_metrics(pred, true, ctrl_sampled)
                all_results["allgene"].append({
                    "dataset": dataset,
                    "cell_type": cell_type,
                    "condition": base_name,
                    "model": "Baseline3",
                    **all_metrics
                })

                # Compute no-perturbation baseline metrics
                nop_all_metrics = compute_metrics(ctrl_sampled, true, ctrl_sampled)
                all_results["allgene"].append({
                    "dataset": dataset,
                    "cell_type": cell_type,
                    "condition": base_name,
                    "model": "Noperturb",
                    **nop_all_metrics
                })

                # Compute Top20, Top50, Top100 DEG metrics
                for top_n in [20, 50, 100]:
                    deg_file = os.path.join(deg_root, dataset, cell_type, f"{base_name}.csv")
                    if not os.path.exists(deg_file):
                        print(f"❌ Missing DEG file: {deg_file}")
                        continue

                    # Load top differential expression genes
                    deg_df = pd.read_csv(deg_file)
                    top_degs = deg_df.head(top_n)["names"].tolist()

                    # Extract data for top DEGs only
                    top_pred_data = pred[:, [adata.var_names.get_loc(g) for g in top_degs]]
                    top_true_data = true[:, [adata.var_names.get_loc(g) for g in top_degs]]
                    top_ctrl_data = ctrl_sampled[:, [adata.var_names.get_loc(g) for g in top_degs]]

                    # Compute metrics for top DEGs
                    top_metrics = compute_metrics(top_pred_data, top_true_data, top_ctrl_data)
                    all_results[f"top{top_n}"].append({
                        "dataset": dataset,
                        "cell_type": cell_type,
                        "condition": base_name,
                        "model": "Baseline3",
                        **top_metrics
                    })

                    # No-perturbation baseline for top DEGs
                    nop_top_metrics = compute_metrics(top_ctrl_data, top_true_data, top_ctrl_data)
                    all_results[f"top{top_n}"].append({
                        "dataset": dataset,
                        "cell_type": cell_type,
                        "condition": base_name,
                        "model": "Noperturb",
                        **nop_top_metrics
                    })

    # Save all results
    for key, result in all_results.items():
        df = pd.DataFrame(result)
        df.to_csv(output_paths[key], index=False)
        print(f"✅ {key} metrics saved to: {output_paths[key]}")


if __name__ == "__main__":
    main()
