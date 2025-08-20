#!/usr/bin/env python3
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
np.random.seed(42)


def compute_mmd(pred_array, truth_array, kernel='rbf', gamma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between predicted and true arrays.
    
    MMD measures the distance between two probability distributions by comparing
    their mean embeddings in a reproducing kernel Hilbert space. This is particularly
    important for evaluating VAE-based models as it captures distribution differences.
    
    Args:
        pred_array (np.ndarray): Predicted values
        truth_array (np.ndarray): True values  
        kernel (str): Kernel type, currently only 'rbf' supported
        gamma (float): RBF kernel parameter
        
    Returns:
        float: MMD value
        
    Raises:
        ValueError: If unsupported kernel type is specified
    """
    if kernel == 'rbf':
        # Compute pairwise squared Euclidean distances
        dist_pred = cdist(pred_array, pred_array, metric='sqeuclidean')
        dist_truth = cdist(truth_array, truth_array, metric='sqeuclidean')
        dist_cross = cdist(pred_array, truth_array, metric='sqeuclidean')

        # Apply RBF kernel transformation
        Kxx = np.exp(-gamma * dist_pred)
        Kyy = np.exp(-gamma * dist_truth)
        Kxy = np.exp(-gamma * dist_cross)

        # MMD formula: E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
        return np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)
    else:
        raise ValueError("Unsupported kernel type. Use 'rbf'.")


def compute_wasserstein(pred_array, truth_array):
    """
    Compute Wasserstein distance between predicted and true arrays.
    
    The Wasserstein distance is particularly useful for evaluating generative models
    as it provides a meaningful metric for comparing distributions.
    
    Args:
        pred_array (np.ndarray): Predicted values
        truth_array (np.ndarray): True values
        
    Returns:
        float: Wasserstein distance
    """
    return wasserstein_distance(pred_array.flatten(), truth_array.flatten())


def calculate_metrics_all(pred_data, true_data, ctrl_data):
    """
    Compute comprehensive evaluation metrics comparing predictions to ground truth.
    
    This function is specifically designed for evaluating VAE-based perturbation
    prediction models, computing both reconstruction quality metrics and 
    distribution-based metrics that are crucial for generative model evaluation.
    
    Args:
        pred_data (array-like): Predicted gene expression data [cells x genes]
        true_data (array-like): True gene expression data [cells x genes] 
        ctrl_data (array-like): Control gene expression data [cells x genes]
        
    Returns:
        dict: Dictionary containing all computed metrics
    """
    # Convert inputs to consistent numpy array format
    if isinstance(true_data, pd.DataFrame):
        true_data = true_data.values
    if isinstance(pred_data, pd.DataFrame):
        pred_data = pred_data.values
    if isinstance(ctrl_data, pd.DataFrame):
        ctrl_data = ctrl_data.values

    # Handle sparse matrices
    pred_data = pred_data.toarray() if not isinstance(pred_data, np.ndarray) else pred_data
    true_data = true_data.toarray() if not isinstance(true_data, np.ndarray) else true_data
    ctrl_data = ctrl_data.toarray() if not isinstance(ctrl_data, np.ndarray) else ctrl_data

    metrics = {}
    
    # Compute mean expression profiles across cells
    mean_true = np.mean(true_data, axis=0)
    mean_pred = np.mean(pred_data, axis=0)
    mean_ctrl = np.mean(ctrl_data, axis=0)

    # Compute regression and correlation metrics with error handling
    with catch_warnings():
        simplefilter("ignore")
        try:
            # Absolute expression metrics
            metrics['R_squared'] = r2_score(mean_true, mean_pred)
            # Delta expression metrics (perturbation effect) - crucial for transfer learning evaluation
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

    # Compute distance and similarity metrics
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
    # These are particularly important for VAE evaluation
    mmd_list, ws_list = [], []
    for g in range(pred_data.shape[1]):
        pred_gene = pred_data[:, g].reshape(-1, 1)
        true_gene = true_data[:, g].reshape(-1, 1)
        mmd_list.append(compute_mmd(pred_gene, true_gene))
        ws_list.append(compute_wasserstein(pred_gene.flatten(), true_gene.flatten()))
        
    metrics['MMD'] = np.mean(mmd_list)
    metrics['Wasserstein'] = np.mean(ws_list)

    return metrics


def parse_trvae_condition(cond_folder):
    """
    Parse condition name from trVAE folder structure.
    
    trVAE uses a specific naming convention where folders are named as
    "ctrl_to_{condition}", and we extract the target condition.
    
    Args:
        cond_folder (str): Condition folder name (e.g., "ctrl_to_stimulated")
        
    Returns:
        str: Target condition name (e.g., "stimulated")
    """
    return cond_folder.split("_to_")[-1]


def validate_cell_type_condition_data(adata, condition, celltype, dataset_name):
    """
    Validate that required cell data exists for the given condition and cell type.
    
    Args:
        adata (AnnData): Annotated data object
        condition (str): Condition name
        celltype (str): Cell type name
        dataset_name (str): Dataset name for logging
        
    Returns:
        tuple: (has_condition_cells, has_control_cells)
    """
    cond_mask = (adata.obs['condition'] == condition) & (adata.obs['cell_type'] == celltype)
    ctrl_mask = (adata.obs['condition'] == 'ctrl') & (adata.obs['cell_type'] == celltype)
    
    has_condition = cond_mask.any()
    has_control = ctrl_mask.any()
    
    return has_condition, has_control


def extract_celltype_from_filename(filename):
    """
    Extract cell type from trVAE prediction filename.
    
    Args:
        filename (str): Prediction filename (e.g., "B_cells_pred.h5ad")
        
    Returns:
        str: Cell type name (e.g., "B_cells")
    """
    return filename.replace("_pred.h5ad", "")


def load_trvae_prediction(pred_file, dataset_name, condition, celltype):
    """
    Load trVAE prediction data with validation.
    
    Args:
        pred_file (str): Path to prediction file
        dataset_name (str): Dataset name for logging
        condition (str): Condition name for logging
        celltype (str): Cell type name for logging
        
    Returns:
        np.ndarray or None: Prediction data or None if loading failed
    """
    try:
        pred_data = sc.read(pred_file).X
        
        if pred_data.shape[0] == 0:
            print(f"Warning: Empty prediction data for {dataset_name}/{condition}/{celltype}")
            return None
            
        return pred_data
        
    except Exception as e:
        print(f"Error loading prediction file {pred_file}: {e}")
        return None


def perform_balanced_data_sampling(pred_data, true_data, ctrl_data, max_samples=500):
    """
    Perform balanced sampling across prediction, true, and control data.
    
    This is important for VAE evaluation to ensure fair comparison across
    different cell populations.
    
    Args:
        pred_data (np.ndarray): Predicted data
        true_data (np.ndarray): True data  
        ctrl_data (np.ndarray): Control data
        max_samples (int): Maximum samples to keep
        
    Returns:
        tuple: (pred_sampled, true_sampled, ctrl_sampled)
    """
    min_n = min(pred_data.shape[0], true_data.shape[0], ctrl_data.shape[0])
    n_sample = min(max_samples, min_n)
    idx = np.random.choice(min_n, n_sample, replace=False)
    
    return pred_data[idx], true_data[idx], ctrl_data[idx]


def process_deg_analysis_for_trvae(adata, dataset, condition, celltype, 
                                  pred_data, true_data, ctrl_data, deg_root, top_n):
    """
    Process top-N differentially expressed gene analysis for trVAE.
    
    Args:
        adata (AnnData): Ground truth data
        dataset (str): Dataset name
        condition (str): Condition name
        celltype (str): Cell type name
        pred_data (np.ndarray): Predicted data
        true_data (np.ndarray): True data
        ctrl_data (np.ndarray): Control data
        deg_root (str): Root directory for DEG files
        top_n (int): Number of top genes to analyze
        
    Returns:
        dict or None: Computed metrics or None if failed
    """
    deg_file = os.path.join(deg_root, dataset, celltype, f"{condition}.csv")
    if not os.path.exists(deg_file):
        print(f"Missing DEG file: {deg_file}")
        return None

    try:
        deg_df = pd.read_csv(deg_file)
        top_degs = deg_df.head(top_n)["names"].tolist()

        # Check which genes exist in the data
        valid_genes = [g for g in top_degs if g in adata.var_names]
        if len(valid_genes) == 0:
            print(f"No valid DEGs found for top{top_n}")
            return None
        
        if len(valid_genes) < top_n:
            print(f"Warning: Only {len(valid_genes)} out of {top_n} DEGs found in data")

        # Extract data for top genes
        gene_indices = [adata.var_names.get_loc(g) for g in valid_genes]
        top_pred_data = pred_data[:, gene_indices]
        top_true_data = true_data[:, gene_indices]
        top_ctrl_data = ctrl_data[:, gene_indices]

        return calculate_metrics_all(top_pred_data, top_true_data, top_ctrl_data)
        
    except Exception as e:
        print(f"Error processing top{top_n} DEGs: {e}")
        return None


def main():
    """Main evaluation pipeline for trVAE model predictions."""
    
    # Path configuration
    trvae_dir = "/data2/lanxiang/perturb_benchmark_v2/model/trVAE"
    raw_data_dir = "/data2/lanxiang/data/Task3_data"
    deg_root = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    output_root = "/data2/lanxiang/perturb_benchmark_v2/Metrics/task3"

    # Define output paths
    output_paths = {
        "allgene": os.path.join(output_root, "Allgene", "trVAE_all_metrics.csv"),
        "top20": os.path.join(output_root, "Top20", "trVAE_top20_metrics.csv"),
        "top50": os.path.join(output_root, "Top50", "trVAE_top50_metrics.csv"),
        "top100": os.path.join(output_root, "Top100", "trVAE_top100_metrics.csv")
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
    for dataset in os.listdir(trvae_dir):
        dataset_path = os.path.join(trvae_dir, dataset)
        raw_file = os.path.join(raw_data_dir, f"{dataset}.h5ad")
        
        if not os.path.exists(raw_file):
            print(f"Raw data file missing: {raw_file}")
            continue

        try:
            adata = sc.read_h5ad(raw_file)
            print(f"Processing trVAE dataset: {dataset}")
        except Exception as e:
            print(f"Failed to load dataset {dataset}: {e}")
            continue

        # Process each condition folder
        for cond_folder in os.listdir(dataset_path):
            # Parse target condition from folder name
            condition = parse_trvae_condition(cond_folder)
            
            cond_path = os.path.join(dataset_path, cond_folder)
            if not os.path.isdir(cond_path):
                continue

            # Process each prediction file
            for file in os.listdir(cond_path):
                if not file.endswith("_pred.h5ad"):
                    continue
                    
                # Extract cell type from filename
                celltype = extract_celltype_from_filename(file)
                pred_file = os.path.join(cond_path, file)

                try:
                    # Validate that required cell data exists
                    has_condition, has_control = validate_cell_type_condition_data(
                        adata, condition, celltype, dataset
                    )
                    if not has_condition or not has_control:
                        continue

                    # Extract ground truth data
                    cond_mask = (adata.obs['condition'] == condition) & (adata.obs['cell_type'] == celltype)
                    ctrl_mask = (adata.obs['condition'] == 'ctrl') & (adata.obs['cell_type'] == celltype)
                    
                    true_data = adata[cond_mask].X
                    ctrl_data = adata[ctrl_mask].X
                    
                    # Load prediction data
                    pred_data = load_trvae_prediction(pred_file, dataset, condition, celltype)
                    if pred_data is None:
                        continue

                    # Perform balanced sampling
                    pred_data, true_data, ctrl_data = perform_balanced_data_sampling(
                        pred_data, true_data, ctrl_data
                    )

                    # Compute all-gene metrics
                    trvae_metrics = calculate_metrics_all(pred_data, true_data, ctrl_data)
                    all_results["allgene"].append({
                        "dataset": dataset, 
                        "condition": condition,
                        "cell_type": celltype, 
                        "model": "trVAE", 
                        **trvae_metrics
                    })

                    # Compute top-N DEG metrics
                    for top_n in [20, 50, 100]:
                        top_metrics = process_deg_analysis_for_trvae(
                            adata, dataset, condition, celltype,
                            pred_data, true_data, ctrl_data, deg_root, top_n
                        )
                        
                        if top_metrics is not None:
                            all_results[f"top{top_n}"].append({
                                "dataset": dataset,
                                "condition": condition,
                                "cell_type": celltype,
                                "model": "trVAE",
                                **top_metrics
                            })

                except Exception as e:
                    print(f"Failed processing {pred_file}: {e}")

    # Save results
    for key, result in all_results.items():
        if len(result) > 0:
            df = pd.DataFrame(result)
            df.to_csv(output_paths[key], index=False)
            print(f"Saved {key} metrics to: {output_paths[key]} ({len(result)} records)")
        else:
            print(f"No results to save for {key}")

    print("\nAll trVAE processing completed!")


if __name__ == "__main__":
    main()