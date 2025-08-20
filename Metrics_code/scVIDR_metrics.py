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


def parse_filename_and_extract_info(dataset, file_path, filename):
    """
    Parse filename to extract condition and cell_type based on dataset type.
    
    This function handles different naming conventions across datasets:
    - eval_results structure: {dataset}_{condition}_{celltype}.h5ad
    - pred_data structure: pred_{condition}_{celltype}.h5ad
    - Special cases for specific datasets like Weinreb_time, Parse_10M_PBMC_sub10, Tahoe100_sub10
    
    Args:
        dataset (str): Dataset name
        file_path (str): Full path to the file
        filename (str): Filename to parse
        
    Returns:
        tuple: (condition, celltype, pred_condition_key) or (None, None, None) if parsing fails
    """
    if dataset in ["Kang", "Haber", "Hagai", "Weinreb_time"]:
        # For eval_results structure: {dataset}_{condition}_{celltype}.h5ad
        name = filename.replace(".h5ad", "")
        parts = name.split("_")
        
        if len(parts) < 3:
            return None, None, None
            
        # Handle special case: Weinreb_time
        if dataset == "Weinreb_time":
            if len(parts) >= 4 and f"{parts[0]}_{parts[1]}" == dataset:
                # For Weinreb_time_developed_ccr7_dc.h5ad
                # parts = ['Weinreb', 'time', 'developed', 'ccr7', 'dc']
                # condition should be 'developed', celltype should be 'ccr7_dc'
                condition = parts[2]  # 'developed'
                celltype = "_".join(parts[3:])  # 'ccr7_dc'
            else:
                return None, None, None
        else:
            # For other datasets: dataset_condition_celltype
            celltype = parts[-1]
            condition = "_".join(parts[1:-1])
        
        pred_condition_key = "pred"  # Simple "pred" for eval_results datasets
        
    else:
        # For pred_data structure: pred_{condition}_{celltype}.h5ad
        if not filename.startswith("pred_"):
            return None, None, None
            
        name = filename.replace("pred_", "").replace(".h5ad", "")
        
        if dataset == "Parse_10M_PBMC_sub10":
            # Special handling for Parse_10M_PBMC_sub10
            # Get condition from folder name
            condition = os.path.basename(os.path.dirname(file_path))
            # Everything after pred_{condition}_ is cell_type
            expected_prefix = f"{condition}_"
            if name.startswith(expected_prefix):
                celltype = name[len(expected_prefix):]
            else:
                return None, None, None
        elif dataset == "Tahoe100_sub10":
            # For Tahoe100_sub10: pred_{condition}_CVCL_XXXX.h5ad
            # Find CVCL_ pattern and split accordingly
            if "CVCL_" not in name:
                return None, None, None
            
            # Split by CVCL_ and take everything before as condition, CVCL_* as celltype
            cvcl_index = name.find("CVCL_")
            if cvcl_index > 0 and name[cvcl_index-1] == "_":
                condition = name[:cvcl_index-1]  # Everything before _CVCL_
                celltype = name[cvcl_index:]      # CVCL_ and everything after
            else:
                return None, None, None
        else:
            # Standard parsing for other pred_data datasets
            parts = name.split("_")
            if len(parts) < 2:
                return None, None, None
            celltype = parts[-1]
            condition = "_".join(parts[:-1])
        
        pred_condition_key = f"pred_{condition}"
    
    return condition, celltype, pred_condition_key


def validate_cell_data_availability(adata, condition, celltype, dataset_name):
    """
    Validate that required cell data exists for analysis.
    
    Args:
        adata (AnnData): Raw annotated data object
        condition (str): Condition name
        celltype (str): Cell type name
        dataset_name (str): Dataset name for logging
        
    Returns:
        tuple: (has_true_cells, has_ctrl_cells)
    """
    true_mask = (adata.obs['condition'] == condition) & (adata.obs['cell_type'] == celltype)
    ctrl_mask = (adata.obs['condition'] == 'ctrl') & (adata.obs['cell_type'] == celltype)
    
    has_true = true_mask.any()
    has_ctrl = ctrl_mask.any()
    
    if not has_true:
        print(f"Warning: No true cells found for condition={condition}, cell_type={celltype} in {dataset_name}")
    if not has_ctrl:
        print(f"Warning: No control cells found for cell_type={celltype} in {dataset_name}")
        
    return has_true, has_ctrl


def extract_prediction_data(pred_adata, pred_condition_key):
    """
    Extract prediction data from prediction AnnData object.
    
    Args:
        pred_adata (AnnData): Prediction AnnData object
        pred_condition_key (str): Key to identify prediction condition
        
    Returns:
        np.ndarray or None: Prediction data or None if not found
    """
    pred_mask = pred_adata.obs['condition'] == pred_condition_key
    if not pred_mask.any():
        print(f"Warning: No predicted cells found for condition={pred_condition_key}")
        return None
    
    return pred_adata[pred_mask].X


def perform_data_sampling(pred_data, true_data, ctrl_data, max_samples=500):
    """
    Perform balanced data sampling to reduce computational cost.
    
    Args:
        pred_data (np.ndarray): Predicted data
        true_data (np.ndarray): True data
        ctrl_data (np.ndarray): Control data
        max_samples (int): Maximum number of samples per dataset
        
    Returns:
        tuple: (pred_sampled, true_sampled, ctrl_sampled)
    """
    # Sample each dataset independently to ensure balanced representation
    n = min(max_samples, true_data.shape[0], pred_data.shape[0], ctrl_data.shape[0])
    
    pred_idx = np.random.choice(pred_data.shape[0], min(n, pred_data.shape[0]), replace=False)
    true_idx = np.random.choice(true_data.shape[0], min(n, true_data.shape[0]), replace=False)
    ctrl_idx = np.random.choice(ctrl_data.shape[0], min(n, ctrl_data.shape[0]), replace=False)
    
    return pred_data[pred_idx], true_data[true_idx], ctrl_data[ctrl_idx]


def compute_top_deg_metrics(deg_file, top_n, adata, pred_adata, 
                           pred_data, true_data, ctrl_data):
    """
    Compute metrics for top N differentially expressed genes.
    
    Args:
        deg_file (str): Path to DEG file
        top_n (int): Number of top genes to analyze
        adata (AnnData): Raw annotated data object
        pred_adata (AnnData): Prediction AnnData object  
        pred_data (np.ndarray): Predicted data
        true_data (np.ndarray): True data
        ctrl_data (np.ndarray): Control data
        
    Returns:
        dict or None: Computed metrics or None if failed
    """
    if not os.path.exists(deg_file):
        print(f"Missing DEG file: {deg_file}")
        return None

    try:
        deg_df = pd.read_csv(deg_file)
        if deg_df.empty or "names" not in deg_df.columns:
            print(f"Invalid DEG file: {deg_file}")
            return None
        
        top_degs = deg_df.head(top_n)["names"].tolist()
        if not top_degs:
            print(f"No DEGs found in: {deg_file}")
            return None
        
        # Get gene indices from raw data (true/ctrl data source)
        gene_indices = [adata.var_names.get_loc(g) for g in top_degs if g in adata.var_names]
        if not gene_indices:
            print(f"No matching genes found in raw data")
            return None
        
        # Get corresponding indices from prediction data
        # Assuming pred_adata and adata have same gene order
        pred_gene_indices = [pred_adata.var_names.get_loc(g) for g in top_degs if g in pred_adata.var_names]
        if not pred_gene_indices:
            print(f"No matching genes found in prediction data")
            return None
        
        # Extract top gene data
        top_pred_data = pred_data[:, pred_gene_indices]
        top_true_data = true_data[:, gene_indices]
        top_ctrl_data = ctrl_data[:, gene_indices]

        return calculate_metrics_all(top_pred_data, top_true_data, top_ctrl_data)
        
    except Exception as e:
        print(f"Error calculating top{top_n} metrics: {e}")
        return None


def main():
    """Main evaluation pipeline for scVIDR model predictions."""
    
    # Path configuration
    scvidr_dir = "/data2/lanxiang/perturb_benchmark_v2/model/scVIDR"
    raw_data_dir = "/data2/lanxiang/data/Task3_data"
    deg_root = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    output_root = "/data2/lanxiang/perturb_benchmark_v2/Metrics/task3"

    # Define output paths
    output_paths = {
        "allgene": os.path.join(output_root, "Allgene", "scVIDR_all_metrics.csv"),
        "top20": os.path.join(output_root, "Top20", "scVIDR_top20_metrics.csv"),
        "top50": os.path.join(output_root, "Top50", "scVIDR_top50_metrics.csv"),
        "top100": os.path.join(output_root, "Top100", "scVIDR_top100_metrics.csv")
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

    # Dataset classifications
    pred_data_datasets = {
        "Srivatsan_sciplex3_sub10", "Burkhardt_sub10", "Perturb_KHP_sub10", 
        "Perturb_cmo_V1_sub10", "Tahoe100_sub10", "Parse_10M_PBMC_sub10"
    }
    eval_results_datasets = {
        "Kang", "Haber", "Hagai", "Weinreb_time"
    }

    # Process each dataset
    for dataset in os.listdir(scvidr_dir):
        ds_path = os.path.join(scvidr_dir, dataset)
        raw_file = os.path.join(raw_data_dir, f"{dataset}.h5ad")
        
        if not os.path.exists(raw_file):
            print(f"Raw data file missing: {raw_file}")
            continue
        
        try:
            raw_adata = sc.read_h5ad(raw_file)
            print(f"Processing scVIDR dataset: {dataset}")
        except Exception as e:
            print(f"Failed to read {raw_file}: {e}")
            continue

        pred_files = []
        
        # Collect prediction files based on dataset structure
        if dataset in pred_data_datasets:
            # pred_data directory structure
            base_pred = os.path.join(ds_path, "pred_data")
            if not os.path.exists(base_pred):
                print(f"pred_data directory not found for {dataset}")
                continue
            
            # Collect all prediction files
            for cond_dir in os.listdir(base_pred):
                cond_path = os.path.join(base_pred, cond_dir)
                if not os.path.isdir(cond_path):
                    continue
                
                for file in os.listdir(cond_path):
                    if file.endswith(".h5ad"):
                        pred_files.append(os.path.join(cond_path, file))

        elif dataset in eval_results_datasets:
            # eval_results directory structure
            eval_base = os.path.join(ds_path, "eval_results")
            if not os.path.exists(eval_base):
                print(f"eval_results directory not found for {dataset}")
                continue
            
            # Collect all prediction files
            for file in os.listdir(eval_base):
                if file.endswith(".h5ad"):
                    pred_files.append(os.path.join(eval_base, file))
        
        else:
            print(f"Unknown dataset type, skipping: {dataset}")
            continue

        # Process each prediction file
        for pred_file in pred_files:
            filename = os.path.basename(pred_file)
            
            # Parse filename to get condition, cell_type, and prediction condition key
            condition, celltype, pred_condition_key = parse_filename_and_extract_info(dataset, pred_file, filename)
            
            if condition is None or celltype is None or pred_condition_key is None:
                print(f"Unable to parse filename: {filename}")
                continue
            
            print(f"  Processing {celltype} × {condition}")
            
            try:
                # Read prediction file
                pred_adata = sc.read_h5ad(pred_file)
                
                # Extract predicted data from prediction file
                pred_data = extract_prediction_data(pred_adata, pred_condition_key)
                if pred_data is None:
                    continue
                
                # Validate data availability in raw data
                has_true, has_ctrl = validate_cell_data_availability(raw_adata, condition, celltype, dataset)
                if not (has_true and has_ctrl):
                    continue
                
                # Extract true and ctrl data from raw data file
                true_mask = (raw_adata.obs['condition'] == condition) & (raw_adata.obs['cell_type'] == celltype)
                ctrl_mask = (raw_adata.obs['condition'] == 'ctrl') & (raw_adata.obs['cell_type'] == celltype)
                
                true_data = raw_adata[true_mask].X
                ctrl_data = raw_adata[ctrl_mask].X
                
                # Check data validity
                if pred_data.shape[0] == 0 or true_data.shape[0] == 0 or ctrl_data.shape[0] == 0:
                    print(f"    Warning: Empty data for {celltype} × {condition}")
                    continue
                
                # Sample data to reduce computation
                pred_data, true_data, ctrl_data = perform_data_sampling(pred_data, true_data, ctrl_data)

                # Calculate all gene metrics
                metrics = calculate_metrics_all(pred_data, true_data, ctrl_data)
                all_results["allgene"].append({
                    "dataset": dataset,
                    "condition": condition,
                    "cell_type": celltype,
                    "model": "scVIDR",
                    **metrics
                })
                print(f"    All gene metrics calculated")

                # Calculate top-N metrics
                for top_n in [20, 50, 100]:
                    deg_file = os.path.join(deg_root, dataset, celltype, f"{condition}.csv")
                    
                    top_metrics = compute_top_deg_metrics(
                        deg_file, top_n, raw_adata, pred_adata,
                        pred_data, true_data, ctrl_data
                    )
                    
                    if top_metrics is not None:
                        all_results[f"top{top_n}"].append({
                            "dataset": dataset,
                            "condition": condition,
                            "cell_type": celltype,
                            "model": "scVIDR",
                            **top_metrics
                        })
                        print(f"    Top{top_n} metrics calculated")
                    
            except Exception as e:
                print(f"    Error processing {filename}: {e}")

    # Save results to CSV
    for key, result in all_results.items():
        if result:
            df = pd.DataFrame(result)
            df.to_csv(output_paths[key], index=False)
            print(f"Saved {key} metrics to: {output_paths[key]} with {len(df)} entries")
        else:
            print(f"No results to save for {key}")

    print("\nAll scVIDR metrics calculation completed!")


if __name__ == "__main__":
    main()