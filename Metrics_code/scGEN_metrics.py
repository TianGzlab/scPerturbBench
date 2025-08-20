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
    their mean embeddings in a reproducing kernel Hilbert space.
    
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
    
    Computes both absolute metrics (comparing predicted vs true expressions directly)
    and delta metrics (comparing perturbation effects: (treated - control)).
    
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
    mmd_list, ws_list = [], []
    for g in range(pred_data.shape[1]):
        pred_gene = pred_data[:, g].reshape(-1, 1)
        true_gene = true_data[:, g].reshape(-1, 1)
        mmd_list.append(compute_mmd(pred_gene, true_gene))
        ws_list.append(compute_wasserstein(pred_gene.flatten(), true_gene.flatten()))
        
    metrics['MMD'] = np.mean(mmd_list)
    metrics['Wasserstein'] = np.mean(ws_list)

    return metrics


def validate_cell_data(adata, condition, cell_type, dataset_name):
    """
    Validate that required cell data exists for the given condition and cell type.
    
    Args:
        adata (AnnData): Annotated data object
        condition (str): Condition name
        cell_type (str): Cell type name
        dataset_name (str): Dataset name for logging
        
    Returns:
        tuple: (has_condition_cells, has_control_cells)
    """
    cond_mask = (adata.obs['condition'] == condition) & (adata.obs['cell_type'] == cell_type)
    ctrl_mask = (adata.obs['condition'] == 'ctrl') & (adata.obs['cell_type'] == cell_type)
    
    has_condition = cond_mask.any()
    has_control = ctrl_mask.any()
    
    if not has_condition:
        print(f"Warning: No cells found for condition={condition}, cell_type={cell_type} in {dataset_name}")
    if not has_control:
        print(f"Warning: No ctrl cells found for cell_type={cell_type} in {dataset_name}")
        
    return has_condition, has_control


def sample_balanced_data(pred_data, true_data, ctrl_data, max_samples=500):
    """
    Sample data to ensure balanced sizes and reduce computational cost.
    
    Args:
        pred_data (np.ndarray): Predicted data
        true_data (np.ndarray): True data  
        ctrl_data (np.ndarray): Control data
        max_samples (int): Maximum number of samples to keep
        
    Returns:
        tuple: Sampled (pred_data, true_data, ctrl_data)
    """
    min_n = min(pred_data.shape[0], true_data.shape[0], ctrl_data.shape[0])
    n_sample = min(max_samples, min_n)
    idx = np.random.choice(min_n, n_sample, replace=False)
    
    return pred_data[idx], true_data[idx], ctrl_data[idx]


def extract_dataset_name(dataset_folder):
    """
    Extract clean dataset name from folder name.
    
    Args:
        dataset_folder (str): Original folder name
        
    Returns:
        str: Cleaned dataset name
    """
    # Remove '_pred_data' suffix if present
    if dataset_folder.endswith('_pred_data'):
        return dataset_folder.replace('_pred_data', '')
    else:
        return dataset_folder


def process_deg_metrics(adata, dataset_name, condition, cell_type, 
                       pred_data, true_data, ctrl_data, deg_root, top_n):
    """
    Process top-N differentially expressed gene metrics.
    
    Args:
        adata (AnnData): Annotated data object
        dataset_name (str): Dataset name
        condition (str): Condition name
        cell_type (str): Cell type name
        pred_data (np.ndarray): Predicted data
        true_data (np.ndarray): True data
        ctrl_data (np.ndarray): Control data
        deg_root (str): Root directory for DEG files
        top_n (int): Number of top genes to analyze
        
    Returns:
        dict or None: Computed metrics or None if failed
    """
    deg_file = os.path.join(deg_root, dataset_name, cell_type, f"{condition}.csv")
    if not os.path.exists(deg_file):
        print(f"Missing DEG file: {deg_file}")
        return None

    try:
        deg_df = pd.read_csv(deg_file)
        if len(deg_df) < top_n:
            print(f"Warning: DEG file has only {len(deg_df)} genes, less than top{top_n}")
            
        top_degs = deg_df.head(top_n)["names"].tolist()

        # Check which genes exist in the data
        valid_genes = [g for g in top_degs if g in adata.var_names]
        if len(valid_genes) < top_n:
            print(f"Warning: Only {len(valid_genes)} out of {top_n} DEGs found in data")
        
        if len(valid_genes) == 0:
            print(f"Error: No valid DEGs found for top{top_n}")
            return None

        # Extract gene indices and subset data
        gene_indices = [adata.var_names.get_loc(g) for g in valid_genes]
        top_pred_data = pred_data[:, gene_indices]
        top_true_data = true_data[:, gene_indices]
        top_ctrl_data = ctrl_data[:, gene_indices]

        return calculate_metrics_all(top_pred_data, top_true_data, top_ctrl_data)
        
    except Exception as e:
        print(f"Error processing top{top_n} DEGs: {e}")
        return None


def main():
    """Main evaluation pipeline for scGEN model predictions."""
    
    # Path configuration  
    scgen_dir = "/data2/lanxiang/perturb_benchmark_v2/model/scGEN"
    raw_data_dir = "/data2/lanxiang/data/Task3_data"
    deg_root = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    output_root = "/data2/lanxiang/perturb_benchmark_v2/Metrics/task3"

    # Define output paths
    output_paths = {
        "allgene": os.path.join(output_root, "Allgene", "scGEN_all_metrics.csv"),
        "top20": os.path.join(output_root, "Top20", "scGEN_top20_metrics.csv"),
        "top50": os.path.join(output_root, "Top50", "scGEN_top50_metrics.csv"),
        "top100": os.path.join(output_root, "Top100", "scGEN_top100_metrics.csv")
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
    for dataset_folder in os.listdir(scgen_dir):
        dataset_path = os.path.join(scgen_dir, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue
        
        # Extract dataset name
        dataset_name = extract_dataset_name(dataset_folder)
        
        # Load corresponding raw data
        raw_file = os.path.join(raw_data_dir, f"{dataset_name}.h5ad")
        if not os.path.exists(raw_file):
            print(f"Raw data file missing: {raw_file}")
            continue

        try:
            adata = sc.read_h5ad(raw_file)
            print(f"Processing scGEN dataset: {dataset_name}")
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue

        # Process each condition folder
        for condition in os.listdir(dataset_path):
            cond_path = os.path.join(dataset_path, condition)
            if not os.path.isdir(cond_path):
                continue

            # Process each prediction file in condition folder
            for file in os.listdir(cond_path):
                if not file.startswith("pred_adata_") or not file.endswith(".h5ad"):
                    continue

                # Extract cell type from filename
                cell_type = file.replace("pred_adata_", "").replace(".h5ad", "")
                pred_file = os.path.join(cond_path, file)

                try:
                    # Validate that required cell data exists
                    has_condition, has_control = validate_cell_data(
                        adata, condition, cell_type, dataset_name
                    )
                    if not (has_condition and has_control):
                        continue

                    # Extract data for this condition and cell type
                    cond_mask = (adata.obs['condition'] == condition) & (adata.obs['cell_type'] == cell_type)
                    ctrl_mask = (adata.obs['condition'] == 'ctrl') & (adata.obs['cell_type'] == cell_type)
                    
                    true_data = adata[cond_mask].X
                    ctrl_data = adata[ctrl_mask].X
                    
                    # Load prediction data
                    pred_data = sc.read(pred_file).X

                    # Sample data to standardize sizes and reduce computation
                    pred_data, true_data, ctrl_data = sample_balanced_data(
                        pred_data, true_data, ctrl_data
                    )

                    # Compute all-gene metrics
                    metrics = calculate_metrics_all(pred_data, true_data, ctrl_data)
                    all_results["allgene"].append({
                        "dataset": dataset_name,
                        "condition": condition,
                        "cell_type": cell_type,
                        "model": "scGEN",
                        **metrics
                    })

                    # Compute top-N DEG metrics
                    for top_n in [20, 50, 100]:
                        top_metrics = process_deg_metrics(
                            adata, dataset_name, condition, cell_type,
                            pred_data, true_data, ctrl_data, deg_root, top_n
                        )
                        
                        if top_metrics is not None:
                            all_results[f"top{top_n}"].append({
                                "dataset": dataset_name,
                                "condition": condition,
                                "cell_type": cell_type,
                                "model": "scGEN",
                                **top_metrics
                            })

                    print(f"Successfully processed: {dataset_name}/{condition}/{cell_type}")

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

    print("\nAll scGEN processing completed!")


if __name__ == "__main__":
    main()