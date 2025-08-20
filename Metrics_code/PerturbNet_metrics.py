import os
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, wasserstein_distance
from scipy.spatial.distance import cdist
from warnings import catch_warnings, simplefilter

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def compute_mmd(pred_array, truth_array, kernel='rbf', gamma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between predicted and true arrays.
    
    MMD measures the distance between two probability distributions by comparing
    their mean embeddings in a reproducing kernel Hilbert space.
    
    Args:
        pred_array (np.ndarray): Predicted values
        truth_array (np.ndarray): True values  
        kernel (str): Kernel type, currently only 'rbf' supported
        gamma (float): RBF kernel parameter
        
    Returns:
        float: MMD value
    """
    if kernel == 'rbf':
        # Compute pairwise distances
        dist_pred = cdist(pred_array, pred_array, metric='sqeuclidean')
        dist_truth = cdist(truth_array, truth_array, metric='sqeuclidean')
        dist_cross = cdist(pred_array, truth_array, metric='sqeuclidean')
        
        # Apply RBF kernel
        Kxx = np.exp(-gamma * dist_pred)
        Kyy = np.exp(-gamma * dist_truth)
        Kxy = np.exp(-gamma * dist_cross)
        
        # MMD formula: E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
        return np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)
    else:
        raise ValueError("Unsupported kernel type")


def compute_metrics(pred_data, true_data, ctrl_data):
    """
    Compute comprehensive evaluation metrics comparing predictions to ground truth.
    
    Computes both absolute metrics (comparing predicted vs true expressions directly)
    and delta metrics (comparing perturbation effects: (treated - control)).
    
    Args:
        pred_data (np.ndarray): Predicted gene expression data [cells x genes]
        true_data (np.ndarray): True gene expression data [cells x genes] 
        ctrl_data (np.ndarray): Control gene expression data [cells x genes]
        
    Returns:
        dict: Dictionary containing all computed metrics
    """
    # Convert sparse matrices to dense arrays if needed
    pred_data = pred_data.toarray() if not isinstance(pred_data, np.ndarray) else pred_data
    true_data = true_data.toarray() if not isinstance(true_data, np.ndarray) else true_data
    ctrl_data = ctrl_data.toarray() if not isinstance(ctrl_data, np.ndarray) else ctrl_data

    # Compute mean expression profiles across cells
    mean_true = np.mean(true_data, axis=0)
    mean_pred = np.mean(pred_data, axis=0)
    mean_ctrl = np.mean(ctrl_data, axis=0)

    metrics = {}
    
    # Compute regression metrics with error handling
    with catch_warnings():
        simplefilter("ignore")
        try:
            # Absolute expression metrics
            metrics['R_squared'] = r2_score(mean_true, mean_pred)
            # Delta expression metrics (perturbation effect)
            metrics['R_squared_delta'] = r2_score(mean_true - mean_ctrl, mean_pred - mean_ctrl)
        except:
            metrics['R_squared'] = np.nan
            metrics['R_squared_delta'] = np.nan
            
        try:
            # Correlation metrics
            metrics['Pearson_Correlation'] = pearsonr(mean_true, mean_pred)[0]
            metrics['Pearson_Correlation_delta'] = pearsonr(mean_true - mean_ctrl, mean_pred - mean_ctrl)[0]
        except:
            metrics['Pearson_Correlation'] = np.nan
            metrics['Pearson_Correlation_delta'] = np.nan

    # Compute additional distance and similarity metrics
    metrics.update({
        # Absolute expression metrics
        'MSE': mean_squared_error(mean_true, mean_pred),
        'RMSE': np.sqrt(mean_squared_error(mean_true, mean_pred)),
        'MAE': mean_absolute_error(mean_true, mean_pred),
        'Cosine_Similarity': cosine_similarity([mean_true], [mean_pred])[0, 0],
        'L2': np.linalg.norm(mean_true - mean_pred),
        
        # Delta expression metrics (perturbation effect)
        'MSE_delta': mean_squared_error(mean_true - mean_ctrl, mean_pred - mean_ctrl),
        'RMSE_delta': np.sqrt(mean_squared_error(mean_true - mean_ctrl, mean_pred - mean_ctrl)),
        'MAE_delta': mean_absolute_error(mean_true - mean_ctrl, mean_pred - mean_ctrl),
        'Cosine_Similarity_delta': cosine_similarity([(mean_true - mean_ctrl)], [(mean_pred - mean_ctrl)])[0, 0],
        'L2_delta': np.linalg.norm((mean_true - mean_ctrl) - (mean_pred - mean_ctrl)),
    })

    # Compute per-gene distribution comparison metrics
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


def normalize_expression_data(data):
    """
    Normalize expression data using CPM (counts per million) followed by log1p transformation.
    
    Args:
        data (np.ndarray): Raw count data [cells x genes]
        
    Returns:
        np.ndarray: Normalized expression data
    """
    # CPM normalization (scale to 10,000 counts per cell)
    cpm_data = data / data.sum(axis=1, keepdims=True) * 1e4
    
    # Log1p transformation
    return np.log1p(cpm_data)


def validate_prediction_files(dataset_path, dataset_name):
    """
    Validate that required prediction files exist for a dataset.
    
    Args:
        dataset_path (str): Path to dataset prediction directory
        dataset_name (str): Name of the dataset
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(dataset_path):
        print(f"Prediction directory not found: {dataset_path}")
        return False
        
    # Check for prediction files
    pred_files = [f for f in os.listdir(dataset_path) if f.endswith("_predict.npy")]
    if len(pred_files) == 0:
        print(f"No prediction files found in: {dataset_path}")
        return False
        
    return True


def load_control_data(adata, dataset_name):
    """
    Load control data from AnnData object, handling different layer structures.
    
    Args:
        adata (AnnData): Annotated data object
        dataset_name (str): Name of the dataset
        
    Returns:
        np.ndarray: Control expression data
    """
    ctrl_mask = adata.obs["condition"] == "ctrl"
    
    if "counts" in adata.layers:
        ctrl_data = adata[ctrl_mask].layers["counts"]
    else:
        print(f"Warning: 'counts' layer not found in {dataset_name}, using .X layer")
        ctrl_data = adata[ctrl_mask].X
        
    return ctrl_data


def extract_top_genes(deg_df, top_n, adata, dataset_name, condition):
    """
    Extract top N differentially expressed genes that exist in the dataset.
    
    Args:
        deg_df (pd.DataFrame): DEG results dataframe
        top_n (int): Number of top genes to extract
        adata (AnnData): Annotated data object
        dataset_name (str): Name of the dataset
        condition (str): Name of the condition
        
    Returns:
        list: List of valid gene indices
    """
    if len(deg_df) < top_n:
        print(f"Warning: DEG file has only {len(deg_df)} genes, less than top{top_n}")
        
    top_degs = deg_df.head(top_n)["names"].tolist()
    
    # Get indices of genes that exist in the dataset
    valid_indices = []
    for gene in top_degs:
        if gene in adata.var_names:
            valid_indices.append(adata.var_names.get_loc(gene))
    
    if len(valid_indices) == 0:
        print(f"Warning: No top{top_n} DEGs found in dataset {dataset_name} for condition {condition}")
        
    return valid_indices


def main():
    """Main evaluation pipeline for PerturbNet model predictions."""
    
    # Path configuration
    perturbnet_dir = "/data2/lanxiang/perturb_benchmark_v2/model/PerturbNet"
    raw_data_dir = "/data2/lanxiang/data/Task1_data"
    output_root = "/data2/lanxiang/perturb_benchmark_v2/Metrics/task1"
    deg_dir = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'

    # Define output paths for different gene sets
    output_paths = {
        "allgene": os.path.join(output_root, "Allgene", "PerturbNet_all_metrics.csv"),
        "top20": os.path.join(output_root, "Top20", "PerturbNet_top20_metrics.csv"),
        "top50": os.path.join(output_root, "Top50", "PerturbNet_top50_metrics.csv"),
        "top100": os.path.join(output_root, "Top100", "PerturbNet_top100_metrics.csv")
    }

    # Create output directories
    for path in output_paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Initialize results storage
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }

    # Process each dataset
    for dataset in os.listdir(perturbnet_dir):
        dataset_path = os.path.join(perturbnet_dir, dataset, "perturbnet_predictions")
        raw_file = os.path.join(raw_data_dir, f"{dataset}.h5ad")
        
        # Validate dataset files
        if not os.path.exists(raw_file):
            print(f"Raw data file missing: {raw_file}")
            continue
            
        if not validate_prediction_files(dataset_path, dataset):
            continue

        # Load dataset
        try:
            adata = sc.read_h5ad(raw_file)
            print(f"Processing dataset: {dataset}")
        except Exception as e:
            print(f"Failed to load dataset {dataset}: {e}")
            continue

        # Load control data
        ctrl_data = load_control_data(adata, dataset)

        # Process each prediction file
        for file in os.listdir(dataset_path):
            if not file.endswith("_predict.npy"):
                continue
                
            condition = file.replace("_predict.npy", "")
            pred_file = os.path.join(dataset_path, f"{condition}_predict.npy")
            real_file = os.path.join(dataset_path, f"{condition}_real.npy")

            if not os.path.exists(real_file):
                print(f"Missing real data file for condition {condition} in dataset {dataset}")
                continue

            try:
                # Load prediction and real data
                pred_data = np.load(pred_file)
                true_data = np.load(real_file)

                # Sample data to standardize sizes and reduce computation
                min_samples = min(pred_data.shape[0], true_data.shape[0], ctrl_data.shape[0])
                sample_size = min(500, min_samples)
                sample_idx = np.random.choice(min_samples, sample_size, replace=False)

                pred_data = pred_data[sample_idx]
                true_data = true_data[sample_idx]
                ctrl_data_sampled = ctrl_data[sample_idx]

                # Convert to dense arrays if needed
                pred_data = pred_data.toarray() if not isinstance(pred_data, np.ndarray) else pred_data
                true_data = true_data.toarray() if not isinstance(true_data, np.ndarray) else true_data
                ctrl_data_sampled = ctrl_data_sampled.toarray() if not isinstance(ctrl_data_sampled, np.ndarray) else ctrl_data_sampled

                # Normalize expression data (CPM + log1p)
                pred_data = normalize_expression_data(pred_data)
                true_data = normalize_expression_data(true_data)
                ctrl_data_sampled = normalize_expression_data(ctrl_data_sampled)

                # Compute all-gene metrics
                pert_metrics = compute_metrics(pred_data, true_data, ctrl_data_sampled)
                all_results["allgene"].append({
                    "dataset": dataset,
                    "condition": condition,
                    "model": "PerturbNet",
                    **pert_metrics
                })

                # Compute top-N DEG metrics
                for top_n in [20, 50, 100]:
                    deg_file = os.path.join(deg_dir, dataset, f"DEG_{condition}.csv")
                    if not os.path.exists(deg_file):
                        print(f"Missing DEG file: {deg_file}")
                        continue

                    try:
                        deg_df = pd.read_csv(deg_file)
                        gene_indices = extract_top_genes(deg_df, top_n, adata, dataset, condition)
                        
                        if len(gene_indices) == 0:
                            continue

                        top_pred_data = pred_data[:, gene_indices]
                        top_true_data = true_data[:, gene_indices]
                        top_ctrl_data = ctrl_data_sampled[:, gene_indices]

                        top_metrics = compute_metrics(top_pred_data, top_true_data, top_ctrl_data)
                        all_results[f"top{top_n}"].append({
                            "dataset": dataset,
                            "condition": condition,
                            "model": "PerturbNet",
                            **top_metrics
                        })
                        
                    except Exception as e:
                        print(f"Error processing top{top_n} DEGs for {dataset} - {condition}: {e}")

            except Exception as e:
                print(f"Failed processing condition {condition} in dataset {dataset}: {e}")

    # Save results
    for key, result in all_results.items():
        if len(result) > 0:
            df = pd.DataFrame(result)
            df.to_csv(output_paths[key], index=False)
            print(f"Saved {key} metrics to: {output_paths[key]} ({len(result)} records)")
        else:
            print(f"No results to save for {key}")


if __name__ == "__main__":
    main()