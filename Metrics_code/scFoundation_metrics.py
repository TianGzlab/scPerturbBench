import os
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, wasserstein_distance
from scipy.spatial.distance import cdist
from warnings import catch_warnings, simplefilter
from scipy.sparse import issparse


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
    
    Includes validation to handle empty data cases and proper error handling for
    challenging datasets.
    
    Args:
        pred_data (np.ndarray or pd.DataFrame): Predicted gene expression data
        true_data (np.ndarray or pd.DataFrame): True gene expression data 
        ctrl_data (np.ndarray or pd.DataFrame): Control gene expression data
        
    Returns:
        dict: Dictionary containing all computed metrics, with NaN for failed computations
    """
    # Check for empty data
    if (hasattr(pred_data, 'shape') and pred_data.shape[0] == 0 or 
        hasattr(true_data, 'shape') and true_data.shape[0] == 0 or 
        hasattr(ctrl_data, 'shape') and ctrl_data.shape[0] == 0):
        
        print("Warning: Empty data detected, returning NaN metrics")
        return create_empty_metrics()

    # Convert data types as needed
    pred_data = convert_to_array(pred_data)
    true_data = convert_to_array(true_data)
    ctrl_data = convert_to_array(ctrl_data)

    # Compute mean expression profiles across cells
    mean_true = np.mean(true_data, axis=0)
    mean_pred = np.mean(pred_data, axis=0)
    mean_ctrl = np.mean(ctrl_data, axis=0)

    # Additional validation
    if mean_true.size == 0 or mean_pred.size == 0:
        print("Warning: Empty mean arrays detected")
        return create_empty_metrics()

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
        # Extract gene expression values for this gene
        if isinstance(pred_data, pd.DataFrame):
            pred_gene = pred_data.iloc[:, g].values.reshape(-1, 1)
            true_gene = true_data.iloc[:, g].values.reshape(-1, 1)
        else:
            pred_gene = pred_data[:, g].reshape(-1, 1)
            true_gene = true_data[:, g].reshape(-1, 1)
            
        mmd_vals.append(compute_mmd(pred_gene, true_gene))
        ws_vals.append(wasserstein_distance(pred_gene.flatten(), true_gene.flatten()))

    metrics['MMD'] = np.mean(mmd_vals)
    metrics['Wasserstein'] = np.mean(ws_vals)

    return metrics


def create_empty_metrics():
    """Create a metrics dictionary with NaN values for failed computations."""
    return {
        'R_squared': np.nan, 'R_squared_delta': np.nan,
        'Pearson_Correlation': np.nan, 'Pearson_Correlation_delta': np.nan,
        'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan,
        'Cosine_Similarity': np.nan, 'L2': np.nan,
        'MSE_delta': np.nan, 'RMSE_delta': np.nan, 'MAE_delta': np.nan,
        'Cosine_Similarity_delta': np.nan, 'L2_delta': np.nan,
        'MMD': np.nan, 'Wasserstein': np.nan
    }


def convert_to_array(data):
    """Convert various data types to numpy arrays."""
    if isinstance(data, (np.ndarray, pd.DataFrame)):
        return data.toarray() if hasattr(data, 'toarray') else data
    else:
        return data.toarray() if hasattr(data, 'toarray') else data


def clean_folder_name(folder_name):
    """Clean and normalize folder names for matching."""
    return folder_name.strip().lower()


def get_gene_names(adata, mapped_dataset_name):
    """
    Get gene names based on dataset-specific conventions.
    
    For the Adamson dataset, uses the 'gene_name' column if available.
    For other datasets, uses the variable index.
    
    Args:
        adata (AnnData): Annotated data object
        mapped_dataset_name (str): Name of the mapped dataset
        
    Returns:
        np.ndarray: Array of gene names
    """
    if mapped_dataset_name == "Adamson":
        # Special handling for Adamson dataset
        if "gene_name" in adata.var.columns:
            return adata.var["gene_name"].values
        else:
            print(f"Warning: gene_name column not found in Adamson dataset, using index")
            return adata.var.index.values
    else:
        # For other datasets, use index
        return adata.var.index.values


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


def align_gene_sets(true_data, pred_data, ctrl_data, common_genes):
    """
    Align gene sets across true, predicted, and control data.
    
    Args:
        true_data (pd.DataFrame): True expression data
        pred_data (pd.DataFrame): Predicted expression data  
        ctrl_data (pd.DataFrame): Control expression data
        common_genes (list): List of common gene names
        
    Returns:
        tuple: Aligned (true_data, pred_data, ctrl_data)
    """
    # Select only common genes and ensure same column order
    true_data = true_data[common_genes]
    pred_data = pred_data[common_genes].reindex(columns=true_data.columns)
    ctrl_data = ctrl_data[common_genes].reindex(columns=true_data.columns)
    
    return true_data, pred_data, ctrl_data


def sample_data_frames(pred_data, true_data, ctrl_data, max_samples=500):
    """
    Sample data frames to limit computational cost.
    
    Args:
        pred_data (pd.DataFrame): Predicted data
        true_data (pd.DataFrame): True data
        ctrl_data (pd.DataFrame): Control data
        max_samples (int): Maximum number of samples per dataset
        
    Returns:
        tuple: Sampled (pred_data, true_data, ctrl_data)
    """
    # Sample predicted data
    if pred_data.shape[0] > max_samples:
        pred_sampled = pred_data.iloc[np.random.choice(pred_data.shape[0], max_samples, replace=False), :]
    else:
        pred_sampled = pred_data
        
    # Sample true data
    if true_data.shape[0] > max_samples:
        true_sampled = true_data.iloc[np.random.choice(true_data.shape[0], max_samples, replace=False), :]
    else:
        true_sampled = true_data
        
    # Sample control data
    if ctrl_data.shape[0] > max_samples:
        ctrl_sampled = ctrl_data.iloc[np.random.choice(ctrl_data.shape[0], max_samples, replace=False), :]
    else:
        ctrl_sampled = ctrl_data
        
    return pred_sampled, true_sampled, ctrl_sampled


def main():
    """Main evaluation pipeline for scFoundation model predictions."""
    
    # Manual mapping for dataset name normalization
    manual_mapping = {
        "dev_perturb_arce_mm_crispri_filtered": "Arce_MM_CRISPRi_sub",
        "dev_perturb_perturb_processed_adamson": "Adamson",
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

    # Path configuration
    PRED_ROOT = "/data1/yy_data/pert/scfoundation/save"
    H5AD_ROOT = "/data2/lanxiang/data/Task1_data"
    DEG_DIR = "/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted"
    output_root = "/data2/lanxiang/perturb_benchmark_v2/Metrics/task1"
    scfoundation_index_path = '/data/yy_data/sc_model/scFoundation-main/OS_scRNA_gene_index.19264.tsv'
    
    # Load scFoundation gene index
    try:
        index_data = pd.read_csv(scfoundation_index_path, sep='\t')
        scfoundation_index = index_data['gene_name'].tolist()
    except Exception as e:
        print(f"Failed to load scFoundation gene index: {e}")
        return

    # Define output paths
    output_paths = {
        "allgene": os.path.join(output_root, "Allgene", "scFoundation_all_metrics.csv"),
        "top20": os.path.join(output_root, "Top20", "scFoundation_top20_metrics.csv"),
        "top50": os.path.join(output_root, "Top50", "scFoundation_top50_metrics.csv"),
        "top100": os.path.join(output_root, "Top100", "scFoundation_top100_metrics.csv")
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
        folder_path = os.path.join(PRED_ROOT, subfolder)
        if not os.path.isdir(folder_path):
            continue

        # Get mapped dataset name
        mapped_dataset_name = manual_mapping.get(subfolder, subfolder)

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
            print(f"Failed to load H5AD file: {e}")
            continue

        # Get gene names using dataset-specific method
        gene_names = get_gene_names(adata, mapped_dataset_name)
        print(f"Gene naming method: {mapped_dataset_name}, Gene count: {len(gene_names)}")

        # Prepare control data
        ctrl_indices = adata.obs[adata.obs['condition'] == 'ctrl'].index
        ctrl_counts = adata[ctrl_indices].X
        ctrl_data = pd.DataFrame(ctrl_counts.toarray(), columns=gene_names, index=ctrl_indices)

        # Process each prediction file
        for pred_file in os.listdir(pred_folder):
            if not pred_file.endswith('.npy.gz.npz'):
                continue

            condition = pred_file
            for ext in ['.npy.gz.npz']:
                if condition.endswith(ext):
                    condition = condition[:-len(ext)]
                    break

            # Prepare true data
            true_indices = adata.obs[adata.obs['condition'] == condition].index
            true_counts = adata[true_indices].X
            true_data = pd.DataFrame(true_counts.toarray(), columns=gene_names, index=true_indices)

            # Load prediction data
            pred_data = load_prediction(os.path.join(pred_folder, pred_file))
            if pred_data is None:
                continue
                
            pred_data = pd.DataFrame(pred_data, columns=scfoundation_index)

            # Find common genes between datasets
            common_cols = list(set(scfoundation_index).intersection(true_data.columns))
            print(f"Common genes count: {len(common_cols)}")
            
            if len(common_cols) == 0:
                print(f"No common genes found, skipping condition: {condition}")
                continue

            # Align gene sets
            true_data, pred_data, ctrl_data = align_gene_sets(
                true_data, pred_data, ctrl_data, common_cols
            )

            # Sample data to reduce computational cost
            pred_sampled, true_sampled, ctrl_sampled = sample_data_frames(
                pred_data, true_data, ctrl_data
            )

            # Compute all-gene metrics
            try:
                pert_metrics = compute_metrics(pred_sampled, true_sampled, ctrl_sampled)
                all_results["allgene"].append({
                    "dataset": mapped_dataset_name,
                    "condition": condition,
                    "model": "scFoundation",
                    **pert_metrics
                })
            except Exception as e:
                print(f"Error computing all-gene metrics for {condition}: {e}")
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

                    # Find DEGs that exist in the sampled data
                    top_degs_in_true = [gene for gene in top_degs if gene in true_sampled.columns]
                    
                    if len(top_degs_in_true) == 0:
                        print(f"No valid top{top_n} DEGs found for condition: {condition}")
                        continue

                    # Select top DEG data
                    top_pred_data = pred_sampled[top_degs_in_true]
                    top_true_data = true_sampled[top_degs_in_true]
                    top_ctrl_data = ctrl_sampled[top_degs_in_true]

                    # Compute metrics
                    top_metrics = compute_metrics(top_pred_data, top_true_data, top_ctrl_data)
                    all_results[f"top{top_n}"].append({
                        "dataset": mapped_dataset_name,
                        "condition": condition,
                        "model": "scFoundation",
                        **top_metrics
                    })
                    
                except Exception as e:
                    print(f"Error processing top{top_n} DEGs for {condition}: {e}")

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