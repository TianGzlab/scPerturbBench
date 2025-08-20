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


def clean_folder_name(folder_name):
    """Clean and normalize folder names for matching."""
    return folder_name.strip().lower()


def find_h5ad_file(folder_name, h5ad_root, manual_mapping):
    """
    Find corresponding H5AD data file for a given folder name.
    
    First checks manual mapping dictionary, then falls back to fuzzy matching.
    
    Args:
        folder_name (str): Name of the folder containing predictions
        h5ad_root (str): Root directory containing H5AD files
        manual_mapping (dict): Manual mapping from folder names to file names
        
    Returns:
        str or None: Path to matching H5AD file, or None if not found
    """
    cleaned_name = clean_folder_name(folder_name).lower()

    # Check manual mapping first
    if cleaned_name in manual_mapping:
        mapped_file = os.path.join(h5ad_root, manual_mapping[cleaned_name] + ".h5ad")
        if os.path.exists(mapped_file):
            return mapped_file
        else:
            print(f"Warning: Manually mapped file does not exist: {mapped_file}")
            return None

    # Fallback to fuzzy matching
    for f in os.listdir(h5ad_root):
        if f.lower().endswith('.h5ad'):
            base = f[:-5].lower()
            if cleaned_name in base or base in cleaned_name:
                return os.path.join(h5ad_root, f)
    return None


def load_prediction(path):
    """
    Load prediction data from NPZ file.
    
    Args:
        path (str): Path to NPZ file containing predictions
        
    Returns:
        np.ndarray or None: Prediction array, or None if loading failed
    """
    try:
        with np.load(path) as npz:
            if 'pred' in npz:
                return npz['pred']
            else:
                print(f"Warning: 'pred' key not found in NPZ file: {path}")
                return None
    except Exception as e:
        print(f"Failed to load prediction: {path}, error: {e}")
        return None


def parse_condition_from_scELMO_filename(filename):
    """
    Parse condition name from scELMO prediction filename.
    
    Handles both single-gene and multi-gene perturbations:
    - Single gene: 'IRF1.npz' -> 'IRF1+ctrl'  
    - Multiple genes: 'CREB1_IRF1.npz' -> 'CREB1+IRF1'
    
    Args:
        filename (str): Prediction filename
        
    Returns:
        str: Parsed condition name
    """
    base = os.path.splitext(filename)[0]
    if '_' in base:
        # Multi-gene perturbation: replace _ with +
        return base.replace('_', '+')
    else:
        # Single-gene perturbation: add +ctrl
        return base + '+ctrl'


def validate_data_availability(adata, condition, dataset_name):
    """
    Validate that required condition data is available in the dataset.
    
    Args:
        adata (AnnData): Annotated data object
        condition (str): Condition name to check
        dataset_name (str): Dataset name for logging
        
    Returns:
        tuple: (has_condition_data, has_control_data)
    """
    cond_mask = adata.obs['condition'] == condition
    ctrl_mask = adata.obs['condition'] == 'ctrl'
    
    has_condition = cond_mask.sum() > 0
    has_control = ctrl_mask.sum() > 0
    
    if not has_condition:
        print(f"Warning: No cells found for condition '{condition}' in dataset {dataset_name}")
    if not has_control:
        print(f"Warning: No control cells found in dataset {dataset_name}")
        
    return has_condition, has_control


def extract_and_sample_data(adata, condition, pred_data, max_samples=500):
    """
    Extract condition and control data, then perform balanced sampling.
    
    Args:
        adata (AnnData): Annotated data object
        condition (str): Condition name
        pred_data (np.ndarray): Predicted data
        max_samples (int): Maximum number of samples to use
        
    Returns:
        tuple: (pred_sampled, true_sampled, ctrl_sampled) or (None, None, None) if failed
    """
    # Extract data for condition and control
    cond_mask = adata.obs['condition'] == condition
    ctrl_mask = adata.obs['condition'] == 'ctrl'
    
    true_data_all = adata[cond_mask].X
    ctrl_data_all = adata[ctrl_mask].X
    
    # Determine sampling size
    min_n = min(pred_data.shape[0], true_data_all.shape[0], ctrl_data_all.shape[0])
    n_sample = min(max_samples, min_n)
    
    if n_sample == 0:
        return None, None, None
    
    # Random sampling
    idx = np.random.choice(min_n, n_sample, replace=False)
    
    pred_sampled = pred_data[idx]
    true_sampled = true_data_all[idx].toarray() if hasattr(true_data_all, 'toarray') else true_data_all[idx]
    ctrl_sampled = ctrl_data_all[idx].toarray() if hasattr(ctrl_data_all, 'toarray') else ctrl_data_all[idx]
    
    return pred_sampled, true_sampled, ctrl_sampled


def main():
    """Main evaluation pipeline for scELMO model predictions."""
    
    # Path configuration
    PRED_ROOT = "/data1/yy_data/pert/scelmo/save2"
    H5AD_ROOT = "/data2/lanxiang/data/Task1_data"
    OUTPUT_ROOT = "/data2/lanxiang/perturb_benchmark_v2/Metrics/task1"
    DEG_DIR = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'

    # Manual mapping for dataset name normalization
    manual_mapping = {
        "dev_perturb_arce_mm_crispri_filtered": "Arce_MM_CRISPRi_sub",
        "dev_perturb_adamson": "Adamson",
        "dev_perturb_junyue_cao_filtered": "Junyue_Cao",
        'dev_perturb_datlingerbock2021': 'DatlingerBock2021',
        'dev_perturb_tiankampmann2021_crispri': 'TianKampmann2021_CRISPRi',
        'dev_perturb_normanweissman2019_filtered': 'NormanWeissman2019_filtered',
        'dev_perturb_frangiehizar2021_rna': 'FrangiehIzar2021_RNA',
        'dev_perturb_dixitregev2016': 'DixitRegev2016',
        'dev_perturb_vcc_train_filtered': 'vcc_train_filtered',
        'dev_perturb_tiankampmann2021_crispra': 'TianKampmann2021_CRISPRa',
        'dev_perturb_fengzhang2023': 'Fengzhang2023',
        'dev_perturb_papalexisatija2021_eccite_rna': 'PapalexiSatija2021_eccite_RNA',
        'dev_perturb_replogleweissman2022_rpe1': 'ReplogleWeissman2022_rpe1',
        'dev_perturb_datlingerbock2017': 'DatlingerBock2017',
        'dev_perturb_sunshine2023_crispri_sarscov2': 'Sunshine2023_CRISPRi_sarscov2'
    }

    # Define output paths for different gene sets
    output_paths = {
        "allgene": os.path.join(OUTPUT_ROOT, "Allgene", "scELMO_all_metrics.csv"),
        "top20": os.path.join(OUTPUT_ROOT, "Top20", "scELMO_top20_metrics.csv"),
        "top50": os.path.join(OUTPUT_ROOT, "Top50", "scELMO_top50_metrics.csv"),
        "top100": os.path.join(OUTPUT_ROOT, "Top100", "scELMO_top100_metrics.csv")
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
    for subfolder in os.listdir(PRED_ROOT):
        # Only process datasets in manual mapping
        if subfolder not in manual_mapping:
            print(f"Skipping dataset not in manual mapping: {subfolder}")
            continue

        folder_path = os.path.join(PRED_ROOT, subfolder)
        if not os.path.isdir(folder_path):
            continue

        # Get mapped dataset name
        mapped_dataset_name = manual_mapping[subfolder]

        pred_folder = os.path.join(folder_path, 'results')
        if not os.path.isdir(pred_folder):
            print(f"Skipping folder without results directory: {subfolder}")
            continue

        # Find corresponding H5AD file
        h5ad_path = find_h5ad_file(subfolder, H5AD_ROOT, manual_mapping)
        if h5ad_path is None:
            print(f"Could not find matching H5AD file for: {subfolder}")
            continue

        print(f"Processing dataset: {subfolder} -> {mapped_dataset_name}")
        
        try:
            adata = sc.read_h5ad(h5ad_path)
        except Exception as e:
            print(f"Failed to load H5AD file {h5ad_path}: {e}")
            continue

        # Validate control data availability
        ctrl_mask = adata.obs['condition'] == 'ctrl'
        if ctrl_mask.sum() == 0:
            print(f"Warning: No control cells found in dataset {subfolder}")
            continue

        # Process each prediction file
        for pred_file in os.listdir(pred_folder):
            if not pred_file.endswith('.npz'):
                continue

            condition = parse_condition_from_scELMO_filename(pred_file)
            
            # Remove file extension if present
            for ext in ['.npz']:
                if condition.endswith(ext):
                    condition = condition[:-len(ext)]
                    break

            # Validate data availability
            has_condition, has_control = validate_data_availability(adata, condition, mapped_dataset_name)
            if not (has_condition and has_control):
                continue

            # Load prediction data
            pred_data = load_prediction(os.path.join(pred_folder, pred_file))
            if pred_data is None:
                continue

            # Extract and sample data
            pred_sampled, true_sampled, ctrl_sampled = extract_and_sample_data(
                adata, condition, pred_data
            )
            
            if pred_sampled is None:
                continue

            # Compute all-gene metrics
            try:
                pert_metrics = compute_metrics(pred_sampled, true_sampled, ctrl_sampled)
                all_results["allgene"].append({
                    "dataset": mapped_dataset_name,
                    "condition": condition,
                    "model": "scELMO",
                    **pert_metrics
                })
            except Exception as e:
                print(f"Error computing all-gene metrics for {mapped_dataset_name} - {condition}: {e}")
                continue

            # Compute top-N DEG metrics
            for top_n in [20, 50, 100]:
                deg_file = os.path.join(DEG_DIR, mapped_dataset_name, f"DEG_{condition}.csv")
                if not os.path.exists(deg_file):
                    print(f"Missing DEG file: {deg_file}")
                    continue

                try:
                    deg_df = pd.read_csv(deg_file)
                    top_degs = deg_df.head(top_n)["names"].tolist()

                    # Ensure top DEGs exist in expression matrix
                    valid_genes = [g for g in top_degs if g in adata.var_names]
                    if len(valid_genes) == 0:
                        print(f"No valid top{top_n} DEGs found for {mapped_dataset_name} - {condition}")
                        continue

                    gene_indices = [adata.var_names.get_loc(g) for g in valid_genes]
                    
                    top_pred_data = pred_sampled[:, gene_indices]
                    top_true_data = true_sampled[:, gene_indices]
                    top_ctrl_data = ctrl_sampled[:, gene_indices]

                    top_metrics = compute_metrics(top_pred_data, top_true_data, top_ctrl_data)
                    all_results[f"top{top_n}"].append({
                        "dataset": mapped_dataset_name,
                        "condition": condition,
                        "model": "scELMO",
                        **top_metrics
                    })
                    
                except Exception as e:
                    print(f"Error processing top{top_n} DEGs for {mapped_dataset_name} - {condition}: {e}")

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