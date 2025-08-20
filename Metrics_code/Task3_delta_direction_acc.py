import os
import scanpy as sc
import numpy as np
import pandas as pd
from warnings import catch_warnings, simplefilter
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data2/lanxiang/perturb_benchmark_v2/Delta_gene/delta_calculator_task3_parallel_no_sampling.log'),
        logging.StreamHandler()
    ]
)

def calculate_gene_direction_consistency(pred_data, true_data, ctrl_data):
    """
    Calculate gene-level direction consistency between predicted and true deltas.
    For each gene, check if pred_delta and true_delta have the same sign (both positive or both negative).
    Returns dictionary with delta_agreement_acc (average accuracy across all genes).
    """
    pred_mean = np.mean(pred_data, axis=0)
    true_mean = np.mean(true_data, axis=0)
    ctrl_mean = np.mean(ctrl_data, axis=0)
    
    # Calculate deltas for each gene
    pred_delta = pred_mean - ctrl_mean
    true_delta = true_mean - ctrl_mean
    
    # Check direction consistency for each gene
    pred_positive = pred_delta > 0
    true_positive = true_delta > 0
    
    # Count genes with consistent direction
    consistent_genes = (pred_positive == true_positive)
    delta_agreement_acc = np.mean(consistent_genes)
    
    return {
        'delta_agreement_acc': delta_agreement_acc
    }

def calculate_gene_direction_consistency_single_cell(pred_mean, true_mean, ctrl_mean):
    """
    Calculate gene-level direction consistency for single cell data (already averaged).
    """
    # Calculate deltas for each gene
    pred_delta = pred_mean - ctrl_mean
    true_delta = true_mean - ctrl_mean
    
    # Check direction consistency for each gene
    pred_positive = pred_delta > 0
    true_positive = true_delta > 0
    
    # Count genes with consistent direction
    consistent_genes = (pred_positive == true_positive)
    delta_agreement_acc = np.mean(consistent_genes)
    
    return {
        'delta_agreement_acc': delta_agreement_acc
    }

def process_baseline3():
    """Process Baseline3 model for Task 3 without sampling"""
    logging.info("Processing Baseline3 model...")
    
    task3_datasets = [
        'Burkhardt_sub10', 'Haber', 'Hagai', 'Kang', 'Weinreb_time',
        'Perturb_cmo_V1_sub10', 'Perturb_KHP_sub10',
        'Srivatsan_sciplex3_sub10', 'Tahoe100_sub10', "Parse_10M_PBMC_sub10"
    ]
    
    h5ad_root = "/data2/lanxiang/data/Task3_data"
    pred_root = "/data2/lanxiang/perturb_benchmark_v2/model/Baseline"
    deg_root = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    # Process each dataset
    for dataset in task3_datasets:
        h5ad_path = os.path.join(h5ad_root, f"{dataset}.h5ad")
        if not os.path.exists(h5ad_path):
            logging.warning(f"Data file not found: {h5ad_path}")
            continue
            
        try:
            adata = sc.read_h5ad(h5ad_path)
        except Exception as e:
            logging.error(f"Failed to read {h5ad_path}: {e}")
            continue
            
        logging.info(f"Processing dataset: {dataset}")
        all_cell_types = adata.obs['cell_type'].unique()
        
        for cell_type in all_cell_types:
            ctrl_mask = (adata.obs['condition'] == 'ctrl') & (adata.obs['cell_type'] == cell_type)
            ctrl_data = adata[ctrl_mask].X
            if ctrl_data.shape[0] == 0:
                logging.warning(f"No ctrl data for {dataset}, cell_type={cell_type}")
                continue
            
            # Convert to array if needed and use ALL control data
            if not isinstance(ctrl_data, np.ndarray):
                ctrl_data = ctrl_data.toarray()
                
            pred_folder = os.path.join(pred_root, dataset, cell_type)
            if not os.path.isdir(pred_folder):
                continue
                
            for fname in os.listdir(pred_folder):
                if not fname.endswith("_predicted_values.npy"):
                    continue
                    
                base_name = fname.replace("_predicted_values.npy", "")
                pred_path = os.path.join(pred_folder, fname)
                
                try:
                    pred = np.load(pred_path, allow_pickle=True)
                    
                    # Convert to array if needed
                    if not isinstance(pred, np.ndarray):
                        pred = pred.toarray()
                    
                    # Get true data from h5ad file based on condition and cell_type
                    true_mask = (adata.obs['condition'] == base_name) & (adata.obs['cell_type'] == cell_type)
                    if not true_mask.any():
                        logging.warning(f"No true data found for condition {base_name}, cell_type {cell_type} in {dataset}")
                        continue
                        
                    true_data = adata[true_mask].X
                    if hasattr(true_data, 'toarray'):
                        true_data = true_data.toarray()
                        
                    # Skip if contains NaN or empty data
                    if (pred.shape[0] == 0 or true_data.shape[0] == 0 or 
                        np.isnan(pred).any() or np.isnan(true_data).any() or np.isnan(ctrl_data).any()):
                        logging.warning(f"Contains NaN or empty data, skipping: {dataset}, {cell_type}, {base_name}")
                        continue
                    
                    # Calculate delta for all genes (no sampling)
                    delta_metrics = calculate_gene_direction_consistency(pred, true_data, ctrl_data)
                    all_results["allgene"].append({
                        "dataset": dataset,
                        "cell_type": cell_type,
                        "condition": base_name,
                        "model": "Baseline3",
                        **delta_metrics
                    })
                    
                    # Calculate delta for NoPerturb
                    nop_delta_metrics = calculate_gene_direction_consistency(ctrl_data, true_data, ctrl_data)
                    all_results["allgene"].append({
                        "dataset": dataset,
                        "cell_type": cell_type,
                        "condition": base_name,
                        "model": "NoPerturb",
                        **nop_delta_metrics
                    })
                    
                    # Calculate Top20, Top50, Top100
                    for top_n in [20, 50, 100]:
                        deg_file = os.path.join(deg_root, dataset, cell_type, f"{base_name}.csv")
                        if not os.path.exists(deg_file):
                            logging.warning(f"Missing DEG file: {deg_file}")
                            continue
                            
                        deg_df = pd.read_csv(deg_file)
                        top_degs = deg_df.head(top_n)["names"].tolist()
                        
                        try:
                            top_pred_data = pred[:, [adata.var_names.get_loc(g) for g in top_degs]]
                            top_true_data = true_data[:, [adata.var_names.get_loc(g) for g in top_degs]]
                            top_ctrl_data = ctrl_data[:, [adata.var_names.get_loc(g) for g in top_degs]]
                            
                            top_delta_metrics = calculate_gene_direction_consistency(top_pred_data, top_true_data, top_ctrl_data)
                            all_results[f"top{top_n}"].append({
                                "dataset": dataset,
                                "cell_type": cell_type,
                                "condition": base_name,
                                "model": "Baseline3",
                                **top_delta_metrics
                            })
                            
                            nop_top_delta_metrics = calculate_gene_direction_consistency(top_ctrl_data, top_true_data, top_ctrl_data)
                            all_results[f"top{top_n}"].append({
                                "dataset": dataset,
                                "cell_type": cell_type,
                                "condition": base_name,
                                "model": "NoPerturb",
                                **nop_top_delta_metrics
                            })
                        except Exception as e:
                            logging.error(f"Error processing top{top_n} for {dataset}/{cell_type}/{base_name}: {e}")
                            
                except Exception as e:
                    logging.error(f"Failed to process {pred_path}: {e}")
    
    return ("Baseline3", all_results)

def process_scpram():
    """Process scPRAM model for Task 3 without sampling"""
    logging.info("Processing scPRAM model...")
    
    scpram_dir = "/data2/lanxiang/perturb_benchmark_v2/model/scPRAM"
    raw_data_dir = "/data2/lanxiang/data/Task3_data"
    deg_root = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    for dataset_folder in os.listdir(scpram_dir):
        dataset_path = os.path.join(scpram_dir, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue
            
        dataset_name = dataset_folder
        raw_file = os.path.join(raw_data_dir, f"{dataset_name}.h5ad")
        if not os.path.exists(raw_file):
            logging.warning(f"Missing raw data: {raw_file}")
            continue
            
        try:
            adata = sc.read_h5ad(raw_file)
        except Exception as e:
            logging.error(f"Failed to read {raw_file}: {e}")
            continue
            
        logging.info(f"Processing scPRAM dataset: {dataset_name}")
        
        for condition in os.listdir(dataset_path):
            cond_path = os.path.join(dataset_path, condition)
            if not os.path.isdir(cond_path):
                continue
                
            for file in os.listdir(cond_path):
                if not file.endswith(".h5ad"):
                    continue
                    
                cell_type = file.replace(".h5ad", "")
                pred_file = os.path.join(cond_path, file)
                
                try:
                    cond_mask = (adata.obs['condition'] == condition) & (adata.obs['cell_type'] == cell_type)
                    ctrl_mask = (adata.obs['condition'] == 'ctrl') & (adata.obs['cell_type'] == cell_type)
                    
                    if not cond_mask.any() or not ctrl_mask.any():
                        continue
                        
                    # Use ALL data without sampling
                    true_data = adata[cond_mask].X
                    ctrl_data = adata[ctrl_mask].X
                    
                    pred_adata = sc.read_h5ad(pred_file)
                    pred_data = pred_adata.X
                    
                    # Convert to array if needed
                    if hasattr(true_data, 'toarray'):
                        true_data = true_data.toarray()
                    if hasattr(ctrl_data, 'toarray'):
                        ctrl_data = ctrl_data.toarray()
                    if hasattr(pred_data, 'toarray'):
                        pred_data = pred_data.toarray()
                    
                    # Calculate all gene delta metrics (no sampling)
                    delta_metrics = calculate_gene_direction_consistency(pred_data, true_data, ctrl_data)
                    all_results["allgene"].append({
                        "dataset": dataset_name,
                        "condition": condition,
                        "cell_type": cell_type,
                        "model": "scPRAM",
                        **delta_metrics
                    })
                    
                    # Calculate Top N delta metrics
                    for top_n in [20, 50, 100]:
                        deg_file = os.path.join(deg_root, dataset_name, cell_type, f"{condition}.csv")
                        if not os.path.exists(deg_file):
                            continue
                            
                        deg_df = pd.read_csv(deg_file)
                        if len(deg_df) < top_n:
                            continue
                            
                        top_degs = deg_df.head(top_n)["names"].tolist()
                        valid_genes = [g for g in top_degs if g in adata.var_names]
                        if len(valid_genes) == 0:
                            continue
                            
                        gene_indices = [adata.var_names.get_loc(g) for g in valid_genes]
                        
                        if pred_adata.var_names.equals(adata.var_names):
                            pred_gene_indices = gene_indices
                        else:
                            pred_gene_indices = [pred_adata.var_names.get_loc(g) for g in valid_genes if g in pred_adata.var_names]
                            if len(pred_gene_indices) != len(valid_genes):
                                continue
                        
                        top_pred_data = pred_data[:, pred_gene_indices]
                        top_true_data = true_data[:, gene_indices]
                        top_ctrl_data = ctrl_data[:, gene_indices]
                        
                        top_delta_metrics = calculate_gene_direction_consistency(top_pred_data, top_true_data, top_ctrl_data)
                        all_results[f"top{top_n}"].append({
                            "dataset": dataset_name,
                            "condition": condition,
                            "cell_type": cell_type,
                            "model": "scPRAM",
                            **top_delta_metrics
                        })
                        
                except Exception as e:
                    logging.error(f"Failed on scPRAM {pred_file}: {e}")
    
    return ("scPRAM", all_results)

def process_scpregan():
    """Process scPreGAN model for Task 3 without sampling"""
    logging.info("Processing scPreGAN model...")
    
    scpregan_dir = "/data2/lanxiang/perturb_benchmark_v2/model/scPreGAN_1000"
    raw_data_dir = "/data2/lanxiang/data/Task3_data"
    deg_root = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    for dataset in os.listdir(scpregan_dir):
        dataset_path = os.path.join(scpregan_dir, dataset)
        raw_file = os.path.join(raw_data_dir, f"{dataset}.h5ad")
        if not os.path.exists(raw_file):
            continue
            
        try:
            adata = sc.read_h5ad(raw_file)
        except Exception as e:
            logging.error(f"Failed to read {raw_file}: {e}")
            continue
            
        logging.info(f"Processing scPreGAN dataset: {dataset}")
        
        for cond_folder in os.listdir(dataset_path):
            if '_' in cond_folder:
                condition = cond_folder.split('_')[0]
            else:
                condition = cond_folder
                
            cond_path = os.path.join(dataset_path, cond_folder)
            if not os.path.isdir(cond_path):
                continue
                
            for cell_folder in os.listdir(cond_path):
                if "pred_adata" not in cell_folder:
                    continue
                    
                pred_adata_path = os.path.join(cond_path, cell_folder)
                if not os.path.isdir(pred_adata_path):
                    continue
                    
                pred_files = [f for f in os.listdir(pred_adata_path) if f.startswith('pred_') and f.endswith('.h5ad')]
                
                for pred_filename in pred_files:
                    cell_type = pred_filename.replace('pred_', '').replace('.h5ad', '')
                    pred_file = os.path.join(pred_adata_path, pred_filename)
                    
                    try:
                        cond_mask = (adata.obs['condition'] == condition) & (adata.obs['cell_type'] == cell_type)
                        ctrl_mask = (adata.obs['condition'] == 'ctrl') & (adata.obs['cell_type'] == cell_type)
                        
                        if not cond_mask.any() or not ctrl_mask.any():
                            continue
                            
                        # Use ALL data without sampling
                        true_data = adata[cond_mask].X
                        ctrl_data = adata[ctrl_mask].X
                        pred_data = sc.read(pred_file).X
                        
                        # Convert to array if needed
                        if hasattr(true_data, 'toarray'):
                            true_data = true_data.toarray()
                        if hasattr(ctrl_data, 'toarray'):
                            ctrl_data = ctrl_data.toarray()
                        if hasattr(pred_data, 'toarray'):
                            pred_data = pred_data.toarray()
                        
                        # Calculate delta metrics (no sampling)
                        delta_metrics = calculate_gene_direction_consistency(pred_data, true_data, ctrl_data)
                        all_results["allgene"].append({
                            "dataset": dataset,
                            "condition": condition,
                            "cell_type": cell_type,
                            "model": "scPreGAN",
                            **delta_metrics
                        })
                        
                        # Calculate Top N delta metrics
                        for top_n in [20, 50, 100]:
                            deg_file = os.path.join(deg_root, dataset, cell_type, f"{condition}.csv")
                            if not os.path.exists(deg_file):
                                continue
                                
                            deg_df = pd.read_csv(deg_file)
                            if len(deg_df) < top_n:
                                continue
                                
                            top_degs = deg_df.head(top_n)["names"].tolist()
                            valid_genes = [g for g in top_degs if g in adata.var_names]
                            if len(valid_genes) == 0:
                                continue
                                
                            gene_indices = [adata.var_names.get_loc(g) for g in valid_genes]
                            top_pred_data = pred_data[:, gene_indices]
                            top_true_data = true_data[:, gene_indices]
                            top_ctrl_data = ctrl_data[:, gene_indices]
                            
                            top_delta_metrics = calculate_gene_direction_consistency(top_pred_data, top_true_data, top_ctrl_data)
                            all_results[f"top{top_n}"].append({
                                "dataset": dataset,
                                "condition": condition,
                                "cell_type": cell_type,
                                "model": "scPreGAN",
                                **top_delta_metrics
                            })
                            
                    except Exception as e:
                        logging.error(f"Failed on scPreGAN {pred_file}: {e}")
    
    return ("scPreGAN", all_results)

def parse_filename_and_extract_info(dataset, file_path, filename):
    """Parse filename to extract condition and cell_type based on dataset type"""
    if dataset in ["Kang", "Haber", "Hagai", "Weinreb_time"]:
        name = filename.replace(".h5ad", "")
        parts = name.split("_")
        
        if len(parts) < 3:
            return None, None, None
            
        if dataset == "Weinreb_time":
            if len(parts) >= 4 and f"{parts[0]}_{parts[1]}" == dataset:
                condition = parts[2]
                celltype = "_".join(parts[3:])
            else:
                return None, None, None
        else:
            celltype = parts[-1]
            condition = "_".join(parts[1:-1])
        
        pred_condition_key = "pred"
        
    else:
        if not filename.startswith("pred_"):
            return None, None, None
            
        name = filename.replace("pred_", "").replace(".h5ad", "")
        
        if dataset == "Parse_10M_PBMC_sub10":
            condition = os.path.basename(os.path.dirname(file_path))
            expected_prefix = f"{condition}_"
            if name.startswith(expected_prefix):
                celltype = name[len(expected_prefix):]
            else:
                return None, None, None
        elif dataset == "Tahoe100_sub10":
            if "CVCL_" not in name:
                return None, None, None
            
            cvcl_index = name.find("CVCL_")
            if cvcl_index > 0 and name[cvcl_index-1] == "_":
                condition = name[:cvcl_index-1]
                celltype = name[cvcl_index:]
            else:
                return None, None, None
        else:
            parts = name.split("_")
            if len(parts) < 2:
                return None, None, None
            celltype = parts[-1]
            condition = "_".join(parts[:-1])
        
        pred_condition_key = f"pred_{condition}"
    
    return condition, celltype, pred_condition_key

def process_scvidr():
    """Process scVIDR model for Task 3 without sampling"""
    logging.info("Processing scVIDR model...")
    
    scvidr_dir = "/data2/lanxiang/perturb_benchmark_v2/model/scVIDR"
    raw_data_dir = "/data2/lanxiang/data/Task3_data"
    deg_root = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    pred_data_datasets = {
        "Srivatsan_sciplex3_sub10", "Burkhardt_sub10", "Perturb_KHP_sub10", 
        "Perturb_cmo_V1_sub10", "Tahoe100_sub10", "Parse_10M_PBMC_sub10"
    }
    eval_results_datasets = {
        "Kang", "Haber", "Hagai", "Weinreb_time"
    }
    
    for dataset in os.listdir(scvidr_dir):
        ds_path = os.path.join(scvidr_dir, dataset)
        raw_file = os.path.join(raw_data_dir, f"{dataset}.h5ad")
        
        if not os.path.exists(raw_file):
            continue
            
        try:
            raw_adata = sc.read_h5ad(raw_file)
        except Exception as e:
            logging.error(f"Failed to read {raw_file}: {e}")
            continue
            
        logging.info(f"Processing scVIDR dataset: {dataset}")
        
        pred_files = []
        
        if dataset in pred_data_datasets:
            base_pred = os.path.join(ds_path, "pred_data")
            if not os.path.exists(base_pred):
                continue
                
            for cond_dir in os.listdir(base_pred):
                cond_path = os.path.join(base_pred, cond_dir)
                if not os.path.isdir(cond_path):
                    continue
                    
                for file in os.listdir(cond_path):
                    if file.endswith(".h5ad"):
                        pred_files.append(os.path.join(cond_path, file))
                        
        elif dataset in eval_results_datasets:
            eval_base = os.path.join(ds_path, "eval_results")
            if not os.path.exists(eval_base):
                continue
                
            for file in os.listdir(eval_base):
                if file.endswith(".h5ad"):
                    pred_files.append(os.path.join(eval_base, file))
        else:
            continue
            
        for pred_file in pred_files:
            filename = os.path.basename(pred_file)
            
            condition, celltype, pred_condition_key = parse_filename_and_extract_info(dataset, pred_file, filename)
            
            if condition is None or celltype is None or pred_condition_key is None:
                continue
                
            try:
                pred_adata = sc.read_h5ad(pred_file)
                pred_mask = pred_adata.obs['condition'] == pred_condition_key
                if not pred_mask.any():
                    continue
                    
                pred_data = pred_adata[pred_mask].X
                
                true_mask = (raw_adata.obs['condition'] == condition) & (raw_adata.obs['cell_type'] == celltype)
                ctrl_mask = (raw_adata.obs['condition'] == 'ctrl') & (raw_adata.obs['cell_type'] == celltype)
                
                if not true_mask.any() or not ctrl_mask.any():
                    continue
                    
                # Use ALL data without sampling
                true_data = raw_adata[true_mask].X
                ctrl_data = raw_adata[ctrl_mask].X
                
                if pred_data.shape[0] == 0 or true_data.shape[0] == 0 or ctrl_data.shape[0] == 0:
                    continue

                # Convert to array if needed
                if hasattr(pred_data, 'toarray'):
                    pred_data = pred_data.toarray()
                if hasattr(true_data, 'toarray'):
                    true_data = true_data.toarray()
                if hasattr(ctrl_data, 'toarray'):
                    ctrl_data = ctrl_data.toarray()
                
                # Calculate all gene delta metrics (no sampling)
                delta_metrics = calculate_gene_direction_consistency(pred_data, true_data, ctrl_data)
                all_results["allgene"].append({
                    "dataset": dataset,
                    "condition": condition,
                    "cell_type": celltype,
                    "model": "scVIDR",
                    **delta_metrics
                })
                
                # Calculate Top N delta metrics
                for top_n in [20, 50, 100]:
                    deg_file = os.path.join(deg_root, dataset, celltype, f"{condition}.csv")
                    if not os.path.exists(deg_file):
                        continue
                        
                    try:
                        deg_df = pd.read_csv(deg_file)
                        if deg_df.empty or "names" not in deg_df.columns:
                            continue
                            
                        top_degs = deg_df.head(top_n)["names"].tolist()
                        if not top_degs:
                            continue
                            
                        gene_indices = [raw_adata.var_names.get_loc(g) for g in top_degs if g in raw_adata.var_names]
                        if not gene_indices:
                            continue
                            
                        pred_gene_indices = [pred_adata.var_names.get_loc(g) for g in top_degs if g in pred_adata.var_names]
                        if not pred_gene_indices:
                            continue
                            
                        top_pred_data = pred_data[:, pred_gene_indices]
                        top_true_data = true_data[:, gene_indices]
                        top_ctrl_data = ctrl_data[:, gene_indices]
                        
                        top_delta_metrics = calculate_gene_direction_consistency(top_pred_data, top_true_data, top_ctrl_data)
                        all_results[f"top{top_n}"].append({
                            "dataset": dataset,
                            "condition": condition,
                            "cell_type": celltype,
                            "model": "scVIDR",
                            **top_delta_metrics
                        })
                        
                    except Exception as e:
                        logging.error(f"Error calculating top{top_n} delta metrics: {e}")
                        
            except Exception as e:
                logging.error(f"Error processing scVIDR {filename}: {e}")
    
    return ("scVIDR", all_results)

def process_trvae():
    """Process trVAE model for Task 3 without sampling"""
    logging.info("Processing trVAE model...")
    
    trvae_dir = "/data2/lanxiang/perturb_benchmark_v2/model/trVAE"
    raw_data_dir = "/data2/lanxiang/data/Task3_data"
    deg_root = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    for dataset in os.listdir(trvae_dir):
        dataset_path = os.path.join(trvae_dir, dataset)
        raw_file = os.path.join(raw_data_dir, f"{dataset}.h5ad")
        if not os.path.exists(raw_file):
            continue
            
        try:
            adata = sc.read_h5ad(raw_file)
        except Exception as e:
            logging.error(f"Failed to read {raw_file}: {e}")
            continue
            
        logging.info(f"Processing trVAE dataset: {dataset}")
        
        for cond_folder in os.listdir(dataset_path):
            condition = cond_folder.split("_to_")[-1]
            cond_path = os.path.join(dataset_path, cond_folder)
            if not os.path.isdir(cond_path):
                continue
                
            for file in os.listdir(cond_path):
                if not file.endswith("_pred.h5ad"):
                    continue
                    
                celltype = file.replace("_pred.h5ad", "")
                pred_file = os.path.join(cond_path, file)
                
                try:
                    cond_mask = (adata.obs['condition'] == condition) & (adata.obs['cell_type'] == celltype)
                    ctrl_mask = (adata.obs['condition'] == 'ctrl') & (adata.obs['cell_type'] == celltype)
                    if not cond_mask.any() or not ctrl_mask.any():
                        continue
                        
                    # Use ALL data without sampling
                    true_data = adata[cond_mask].X
                    ctrl_data = adata[ctrl_mask].X
                    pred_data = sc.read(pred_file).X
                    
                    # Convert to array if needed
                    if hasattr(true_data, 'toarray'):
                        true_data = true_data.toarray()
                    if hasattr(ctrl_data, 'toarray'):
                        ctrl_data = ctrl_data.toarray()
                    if hasattr(pred_data, 'toarray'):
                        pred_data = pred_data.toarray()
                    
                    # Calculate delta metrics (no sampling)
                    delta_metrics = calculate_gene_direction_consistency(pred_data, true_data, ctrl_data)
                    all_results["allgene"].append({
                        "dataset": dataset,
                        "condition": condition,
                        "cell_type": celltype,
                        "model": "trVAE",
                        **delta_metrics
                    })
                    
                    # Calculate Top N delta metrics
                    for top_n in [20, 50, 100]:
                        deg_file = os.path.join(deg_root, dataset, celltype, f"{condition}.csv")
                        if not os.path.exists(deg_file):
                            continue
                            
                        deg_df = pd.read_csv(deg_file)
                        top_degs = deg_df.head(top_n)["names"].tolist()
                        
                        top_pred_data = pred_data[:, [adata.var_names.get_loc(g) for g in top_degs]]
                        top_true_data = true_data[:, [adata.var_names.get_loc(g) for g in top_degs]]
                        top_ctrl_data = ctrl_data[:, [adata.var_names.get_loc(g) for g in top_degs]]
                        
                        top_delta_metrics = calculate_gene_direction_consistency(top_pred_data, top_true_data, top_ctrl_data)
                        all_results[f"top{top_n}"].append({
                            "dataset": dataset,
                            "condition": condition,
                            "cell_type": celltype,
                            "model": "trVAE",
                            **top_delta_metrics
                        })
                        
                except Exception as e:
                    logging.error(f"Failed on trVAE {pred_file}: {e}")
    
    return ("trVAE", all_results)

def process_scgen():
    """Process scGEN model for Task 3 without sampling"""
    logging.info("Processing scGEN model...")
    
    scgen_dir = "/data2/lanxiang/perturb_benchmark_v2/model/scGEN"
    raw_data_dir = "/data2/lanxiang/data/Task3_data"
    deg_root = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    for dataset_folder in os.listdir(scgen_dir):
        dataset_path = os.path.join(scgen_dir, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue
            
        if dataset_folder.endswith('_pred_data'):
            dataset_name = dataset_folder.replace('_pred_data', '')
        else:
            dataset_name = dataset_folder
            
        raw_file = os.path.join(raw_data_dir, f"{dataset_name}.h5ad")
        if not os.path.exists(raw_file):
            continue
            
        try:
            adata = sc.read_h5ad(raw_file)
        except Exception as e:
            logging.error(f"Failed to read {raw_file}: {e}")
            continue
            
        logging.info(f"Processing scGEN dataset: {dataset_name}")
        
        for condition in os.listdir(dataset_path):
            cond_path = os.path.join(dataset_path, condition)
            if not os.path.isdir(cond_path):
                continue
                
            for file in os.listdir(cond_path):
                if not file.startswith("pred_adata_") or not file.endswith(".h5ad"):
                    continue
                    
                cell_type = file.replace("pred_adata_", "").replace(".h5ad", "")
                pred_file = os.path.join(cond_path, file)
                
                try:
                    cond_mask = (adata.obs['condition'] == condition) & (adata.obs['cell_type'] == cell_type)
                    ctrl_mask = (adata.obs['condition'] == 'ctrl') & (adata.obs['cell_type'] == cell_type)
                    
                    if not cond_mask.any() or not ctrl_mask.any():
                        continue
                        
                    # Use ALL data without sampling
                    true_data = adata[cond_mask].X
                    ctrl_data = adata[ctrl_mask].X
                    pred_data = sc.read(pred_file).X
                    
                    # Convert to array if needed
                    if hasattr(true_data, 'toarray'):
                        true_data = true_data.toarray()
                    if hasattr(ctrl_data, 'toarray'):
                        ctrl_data = ctrl_data.toarray()
                    if hasattr(pred_data, 'toarray'):
                        pred_data = pred_data.toarray()
                    
                    # Calculate delta metrics (no sampling)
                    delta_metrics = calculate_gene_direction_consistency(pred_data, true_data, ctrl_data)
                    all_results["allgene"].append({
                        "dataset": dataset_name,
                        "condition": condition,
                        "cell_type": cell_type,
                        "model": "scGEN",
                        **delta_metrics
                    })
                    
                    # Calculate Top N delta metrics
                    for top_n in [20, 50, 100]:
                        deg_file = os.path.join(deg_root, dataset_name, cell_type, f"{condition}.csv")
                        if not os.path.exists(deg_file):
                            continue
                            
                        deg_df = pd.read_csv(deg_file)
                        if len(deg_df) < top_n:
                            continue
                            
                        top_degs = deg_df.head(top_n)["names"].tolist()
                        valid_genes = [g for g in top_degs if g in adata.var_names]
                        if len(valid_genes) == 0:
                            continue
                            
                        gene_indices = [adata.var_names.get_loc(g) for g in valid_genes]
                        top_pred_data = pred_data[:, gene_indices]
                        top_true_data = true_data[:, gene_indices]
                        top_ctrl_data = ctrl_data[:, gene_indices]
                        
                        top_delta_metrics = calculate_gene_direction_consistency(top_pred_data, top_true_data, top_ctrl_data)
                        all_results[f"top{top_n}"].append({
                            "dataset": dataset_name,
                            "condition": condition,
                            "cell_type": cell_type,
                            "model": "scGEN",
                            **top_delta_metrics
                        })
                        
                except Exception as e:
                    logging.error(f"Failed on scGEN {pred_file}: {e}")
    
    return ("scGEN", all_results)

def save_results(output_root, model_results_list):
    """Save results from all models to CSV files"""
    # Create output directories
    for gene_subset in ['Allgene', 'Top20', 'Top50', 'Top100']:
        os.makedirs(os.path.join(output_root, gene_subset), exist_ok=True)
    
    # Combine results from all models
    combined_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    for model_name, results in model_results_list:
        for key in combined_results.keys():
            combined_results[key].extend(results[key])
    
    # Save combined results
    for key, result in combined_results.items():
        if result:
            df = pd.DataFrame(result)
            output_path = os.path.join(output_root, key.capitalize(), f"All_models_task3_delta_{key}.csv")
            df.to_csv(output_path, index=False)
            logging.info(f"All models task3 {key} delta results saved: {len(df)} records")
    
    # Save individual model results
    for model_name, results in model_results_list:
        for key, result in results.items():
            if result:
                df = pd.DataFrame(result)
                output_path = os.path.join(output_root, key.capitalize(), f"{model_name}_task3_delta_{key}.csv")
                df.to_csv(output_path, index=False)
                logging.info(f"{model_name} task3 {key} delta results saved: {len(df)} records")

def main():
    output_root = "/data2/lanxiang/perturb_benchmark_v2/Delta_gene"
    
    # Define all model processing functions for Task 3
    model_functions = [
        process_baseline3,
        process_scpram,
        process_scpregan,
        process_scvidr,
        process_trvae,
        process_scgen
    ]
    
    logging.info(f"Starting Task 3 delta calculation with {len(model_functions)} models using parallel processing (NO SAMPLING)")
    
    # Use ProcessPoolExecutor for parallel processing
    max_workers = min(len(model_functions), mp.cpu_count())
    model_results_list = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_model = {executor.submit(func): func.__name__ for func in model_functions}
        
        # Collect results as they complete
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                model_results_list.append(result)
                logging.info(f"Completed processing {result[0]} model")
            except Exception as e:
                logging.error(f"Error processing {model_name}: {e}")
    
    # Save all results
    save_results(output_root, model_results_list)
    
    logging.info("=== Task 3 Delta calculation completed for all models (NO SAMPLING) ===")
    
    # Print summary
    for gene_subset in ['Allgene', 'Top20', 'Top50', 'Top100']:
        folder_path = os.path.join(output_root, gene_subset)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and 'task3' in f]
            logging.info(f"{gene_subset}: {len(files)} Task 3 model results")

if __name__ == "__main__":
    main()