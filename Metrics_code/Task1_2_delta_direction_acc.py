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
        logging.FileHandler('/data2/lanxiang/perturb_benchmark_v2/Delta_gene/delta_calculator_parallel_no_sampling.log'),
        logging.StreamHandler()
    ]
)

skip_datasets = {
    'Burkhardt_sub10', 'Haber', 'Hagai', 'Kang', "Weinreb_time",
    'Perturb_cmo_V1_sub10', 'Perturb_KHP_sub10',
    'Srivatsan_sciplex3_sub10', 'Tahoe100_sub10', "Parse_10M_PBMC_sub10"
}

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

def find_matching_h5ad_file(h5ad_root, dataset_name):
    for fname in os.listdir(h5ad_root):
        if fname == dataset_name + ".h5ad":
            return os.path.join(h5ad_root, fname)
    return None

def clean_folder_name(folder_name):
    return folder_name.strip().lower()

def find_h5ad_file(folder_name, h5ad_root, manual_mapping):
    cleaned_name = clean_folder_name(folder_name).lower()

    if cleaned_name in manual_mapping:
        mapped_file = os.path.join(h5ad_root, manual_mapping[cleaned_name] + ".h5ad")
        if os.path.exists(mapped_file):
            return mapped_file
        else:
            logging.warning(f"Manual mapping file not found: {mapped_file}")
            return None

    for f in os.listdir(h5ad_root):
        if f.lower().endswith('.h5ad'):
            base = f[:-5].lower()
            if cleaned_name in base or base in cleaned_name:
                return os.path.join(h5ad_root, f)
    return None

def load_prediction(path):
    try:
        with np.load(path) as npz:
            if 'pred' in npz:
                return npz['pred']
            else:
                logging.warning(f"No 'pred' key in npz file: {path}")
                return None
    except Exception as e:
        logging.error(f"Failed to load prediction: {path}, error: {e}")
        return None

def parse_condition_from_filename(filename):
    base = os.path.splitext(filename)[0]
    if '_' in base:
        return base.replace('_', '+')
    else:
        return base + '+ctrl'

def process_baseline():
    """Process Baseline1_2 model without sampling - fetch true data from h5ad"""
    logging.info("Processing Baseline model...")
    baseline_root = "/data2/lanxiang/perturb_benchmark_v2/model/Baseline"
    h5ad_root = "/data2/lanxiang/data/Task1_data"
    deg_root = "/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted"
    
    results_list = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    for dataset in os.listdir(baseline_root):
        if dataset in skip_datasets:
            continue
            
        dataset_dir = os.path.join(baseline_root, dataset)
        if not os.path.isdir(dataset_dir):
            continue
            
        h5ad_path = find_matching_h5ad_file(h5ad_root, dataset)
        if h5ad_path is None:
            continue
            
        try:
            adata = sc.read_h5ad(h5ad_path)
        except Exception as e:
            logging.error(f"Failed to read {h5ad_path}: {e}")
            continue
            
        if 'condition' not in adata.obs.columns:
            continue
            
        ctrl_mask = adata.obs['condition'] == 'ctrl'
        if ctrl_mask.sum() == 0:
            continue
            
        # Use ALL control data without sampling
        ctrl_data = adata[ctrl_mask].X
        if hasattr(ctrl_data, 'toarray'):
            ctrl_data = ctrl_data.toarray()
        
        for fname in os.listdir(dataset_dir):
            if not fname.endswith("_predicted_values.npy"):
                continue
                
            base = fname.replace("_predicted_values.npy", "")
            pred_path = os.path.join(dataset_dir, fname)
            
            try:
                pred = np.load(pred_path)
            except Exception as e:
                logging.error(f"Failed to load prediction data for {base}: {e}")
                continue
            
            # Convert condition format: base uses "_", h5ad uses "+"
            condition = base.replace("_", "+")
            
            # Get true data from h5ad file based on condition
            true_mask = adata.obs['condition'] == condition
            if not true_mask.any():
                logging.warning(f"No true data found for condition {condition} in {dataset}")
                continue
                
            true_data = adata[true_mask].X
            if hasattr(true_data, 'toarray'):
                true_data = true_data.toarray()
            
            # Calculate delta for all genes (no sampling)
            delta_metrics = calculate_gene_direction_consistency(pred, true_data, ctrl_data)
            results_list["allgene"].append({
                'dataset': dataset,
                'condition': condition,
                'model': 'Baseline',
                **delta_metrics
            })
            
            # Calculate delta for NoPerturb
            nop_delta_metrics = calculate_gene_direction_consistency(ctrl_data, true_data, ctrl_data)
            results_list["allgene"].append({
                'dataset': dataset,
                'condition': condition,
                'model': 'NoPerturb',
                **nop_delta_metrics
            })
            
            # Calculate Top20, Top50, Top100
            for top_n in [20, 50, 100]:
                deg_folder = os.path.join(deg_root, dataset)
                deg_file = os.path.join(deg_folder, f"DEG_{condition}.csv")
                
                if not os.path.exists(deg_file):
                    continue
                    
                deg_df = pd.read_csv(deg_file)
                top_degs = deg_df.head(top_n)["names"].tolist()
                
                try:
                    top_pred_data = pred[:, [adata.var_names.get_loc(g) for g in top_degs]]
                    top_true_data = true_data[:, [adata.var_names.get_loc(g) for g in top_degs]]
                    top_ctrl_data = ctrl_data[:, [adata.var_names.get_loc(g) for g in top_degs]]
                    
                    top_delta_metrics = calculate_gene_direction_consistency(top_pred_data, top_true_data, top_ctrl_data)
                    results_list[f"top{top_n}"].append({
                        'dataset': dataset,
                        'condition': condition,
                        'model': 'Baseline',
                        **top_delta_metrics
                    })
                    
                    nop_top_delta_metrics = calculate_gene_direction_consistency(top_ctrl_data, top_true_data, top_ctrl_data)
                    results_list[f"top{top_n}"].append({
                        'dataset': dataset,
                        'condition': condition,
                        'model': 'NoPerturb',
                        **nop_top_delta_metrics
                    })
                except Exception as e:
                    logging.error(f"Error processing top{top_n} for {dataset}/{condition}: {e}")
    
    return ("Baseline", results_list)

def process_biolord():
    """Process Biolord model without sampling"""
    logging.info("Processing Biolord model...")
    biolord_dir = "/data2/lanxiang/perturb_benchmark_v2/model/Biolord_go_1000"
    raw_data_dir = "/data2/lanxiang/data/Task1_data"
    deg_dir = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    for dataset in os.listdir(biolord_dir):
        dataset_path = os.path.join(biolord_dir, dataset)
        raw_file = os.path.join(raw_data_dir, f"{dataset}.h5ad")
        if not os.path.exists(raw_file):
            continue
            
        adata = sc.read_h5ad(raw_file)
        
        # Use ALL control data without sampling
        ctrl_mask = adata.obs["condition"] == "ctrl"
        ctrl_data = adata[ctrl_mask].X
        ctrl_data = ctrl_data.toarray() if not isinstance(ctrl_data, np.ndarray) else ctrl_data
        ctrl_mean = np.mean(ctrl_data, axis=0)
        
        for file in os.listdir(dataset_path):
            if not file.endswith("_pred.csv"):
                continue
                
            condition = file.replace("_pred.csv", "")
            pred_file = os.path.join(dataset_path, f"{condition}_pred.csv")
            real_file = os.path.join(dataset_path, f"{condition}_real.csv")
            
            if not os.path.exists(real_file):
                continue
                
            try:
                pred_df = pd.read_csv(pred_file, index_col=0)
                true_df = pd.read_csv(real_file, index_col=0)
                
                if pred_df.shape[0] != 1 or true_df.shape[0] != 1:
                    continue
                    
                pred_mean = pred_df.iloc[0].values
                true_mean = true_df.iloc[0].values
                
                pred_genes = pred_df.columns.tolist()
                true_genes = true_df.columns.tolist()
                adata_genes = adata.var_names.tolist()
                
                common_genes = list(set(pred_genes) & set(true_genes) & set(adata_genes))
                if len(common_genes) == 0:
                    continue
                    
                pred_indices = [pred_genes.index(g) for g in common_genes]
                true_indices = [true_genes.index(g) for g in common_genes]
                ctrl_indices = [adata_genes.index(g) for g in common_genes]
                
                pred_mean_aligned = pred_mean[pred_indices]
                true_mean_aligned = true_mean[true_indices]
                ctrl_mean_aligned = ctrl_mean[ctrl_indices]
                
                # Calculate delta for all genes
                delta_metrics = calculate_gene_direction_consistency_single_cell(pred_mean_aligned, true_mean_aligned, ctrl_mean_aligned)
                all_results["allgene"].append({
                    "dataset": dataset,
                    "condition": condition,
                    "model": "Biolord",
                    **delta_metrics
                })
                
                # Calculate Top20, Top50, Top100
                for top_n in [20, 50, 100]:
                    deg_file = os.path.join(deg_dir, dataset, f"DEG_{condition}.csv")
                    if not os.path.exists(deg_file):
                        continue
                        
                    deg_df = pd.read_csv(deg_file)
                    top_degs = deg_df.head(top_n)["names"].tolist()
                    
                    top_degs_available = [g for g in top_degs if g in common_genes]
                    if len(top_degs_available) == 0:
                        continue
                        
                    top_pred_indices = [common_genes.index(g) for g in top_degs_available]
                    
                    top_pred_data = pred_mean_aligned[top_pred_indices]
                    top_true_data = true_mean_aligned[top_pred_indices]
                    top_ctrl_data = ctrl_mean_aligned[top_pred_indices]
                    
                    top_delta_metrics = calculate_gene_direction_consistency_single_cell(top_pred_data, top_true_data, top_ctrl_data)
                    all_results[f"top{top_n}"].append({
                        "dataset": dataset,
                        "condition": condition,
                        "model": "Biolord",
                        **top_delta_metrics
                    })
                    
            except Exception as e:
                logging.error(f"Failed on {dataset} / {condition}: {e}")
    
    return ("Biolord", all_results)

def process_cpa_task2():
    """Process CPA Task2 model without sampling"""
    logging.info("Processing CPA Task2 model...")
    seen_base = "/data2/lanxiang/perturb_benchmark_v2/model/CPA/Task2/Seen"
    raw_base = "/data2/lanxiang/perturb_benchmark_v2/model/CPA/Task2"
    deg_dir = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    for seen_split in ["seen0", "seen1", "seen2"]:
        seen_path = os.path.join(seen_base, seen_split)
        if not os.path.exists(seen_path):
            continue
            
        for dataset in os.listdir(seen_path):
            csv_dir = os.path.join(seen_path, dataset)
            h5ad_path = os.path.join(raw_base, dataset, f"{dataset}_pred.h5ad")
            if not os.path.exists(h5ad_path):
                continue
                
            adata = sc.read_h5ad(h5ad_path)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            
            # Use ALL control data without sampling
            ctrl_mask = (adata.obs["condition"] == "ctrl")
            ctrl_data = adata[ctrl_mask].X
            if hasattr(ctrl_data, 'toarray'):
                ctrl_data = ctrl_data.toarray()
            
            for csv_file in os.listdir(csv_dir):
                if not csv_file.endswith(".csv"):
                    continue
                    
                condition = csv_file.replace(".csv", "")
                try:
                    df_pred = pd.read_csv(os.path.join(csv_dir, csv_file), index_col=0)
                    pred_data = df_pred.values
                    pred_data = pred_data / pred_data.sum(axis=1, keepdims=True) * 1e4
                    pred_data = np.log1p(pred_data)
                    
                    true_mask = (adata.obs["condition"] == condition)
                    true_data = adata[true_mask].X
                    if hasattr(true_data, 'toarray'):
                        true_data = true_data.toarray()
                    
                    # Calculate delta for all genes (no sampling)
                    delta_metrics = calculate_gene_direction_consistency(pred_data, true_data, ctrl_data)
                    all_results["allgene"].append({
                        "dataset": dataset,
                        "condition": condition,
                        "model": "CPA",
                        "seen_split": seen_split,
                        **delta_metrics
                    })
                    
                    # Calculate Top20, Top50, Top100
                    for top_n in [20, 50, 100]:
                        deg_file = os.path.join(deg_dir, dataset, f"DEG_{condition}.csv")
                        if not os.path.exists(deg_file):
                            continue
                            
                        deg_df = pd.read_csv(deg_file)
                        top_degs = deg_df.head(top_n)["names"].tolist()
                        
                        try:
                            top_pred_data = pred_data[:, [adata.var_names.get_loc(g) for g in top_degs]]
                            top_true_data = true_data[:, [adata.var_names.get_loc(g) for g in top_degs]]
                            top_ctrl_data = ctrl_data[:, [adata.var_names.get_loc(g) for g in top_degs]]
                            
                            top_delta_metrics = calculate_gene_direction_consistency(top_pred_data, top_true_data, top_ctrl_data)
                            all_results[f"top{top_n}"].append({
                                "dataset": dataset,
                                "condition": condition,
                                "model": "CPA",
                                "seen_split": seen_split,
                                **top_delta_metrics
                            })
                        except Exception as e:
                            logging.error(f"Error processing top{top_n} for {dataset}/{condition}: {e}")
                            
                except Exception as e:
                    logging.error(f"Failed: {seen_split}/{dataset}/{condition} | {e}")
    
    return ("CPA_Task2", all_results)

def process_scfoundation():
    """Process scFoundation model without sampling"""
    logging.info("Processing scFoundation model...")
    
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
    
    PRED_ROOT = "/data1/yy_data/pert/scfoundation/save"
    H5AD_ROOT = "/data2/lanxiang/data/Task1_data"
    DEG_DIR = "/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted"
    scfoundation_index_path = '/data/yy_data/sc_model/scFoundation-main/OS_scRNA_gene_index.19264.tsv'
    
    # Load scFoundation index
    index_data = pd.read_csv(scfoundation_index_path, sep='\t')
    scfoundation_index = index_data['gene_name'].tolist()
    
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    def get_gene_names(adata, mapped_dataset_name):
        if mapped_dataset_name == "Adamson":
            if "gene_name" in adata.var.columns:
                return adata.var["gene_name"].values
            else:
                return adata.var.index.values
        else:
            return adata.var.index.values
    
    for subfolder in os.listdir(PRED_ROOT):
        folder_path = os.path.join(PRED_ROOT, subfolder)
        if not os.path.isdir(folder_path):
            continue
            
        mapped_dataset_name = manual_mapping.get(subfolder, subfolder)
        pred_folder = os.path.join(folder_path, 'results')
        if not os.path.isdir(pred_folder):
            continue
            
        h5ad_path = find_h5ad_file(subfolder, H5AD_ROOT, manual_mapping)
        if h5ad_path is None:
            continue
            
        adata = sc.read_h5ad(h5ad_path)
        gene_names = get_gene_names(adata, mapped_dataset_name)
        
        # Use ALL control data without sampling
        ctrl_indices = adata.obs[adata.obs['condition'] == 'ctrl'].index
        ctrl_counts = adata[ctrl_indices].X
        ctrl_data = pd.DataFrame(ctrl_counts.toarray(), columns=gene_names, index=ctrl_indices)
        
        for pred_file in os.listdir(pred_folder):
            if not pred_file.endswith(('.npy.gz.npz')):
                continue
                
            condition = pred_file
            for ext in ['.npy.gz.npz']:
                if condition.endswith(ext):
                    condition = condition[:-len(ext)]
                    break
                    
            true_indices = adata.obs[adata.obs['condition'] == condition].index
            if len(true_indices) == 0:
                continue
                
            # Use ALL true data without sampling
            true_counts = adata[true_indices].X
            true_data = pd.DataFrame(true_counts.toarray(), columns=gene_names, index=true_indices)
            
            pred_array = load_prediction(os.path.join(pred_folder, pred_file))
            if pred_array is None:
                continue
                
            pred_data = pd.DataFrame(pred_array, columns=scfoundation_index)
            
            common_cols = list(set(scfoundation_index).intersection(true_data.columns))
            if len(common_cols) == 0:
                continue
                
            true_data = true_data[common_cols]
            pred_data = pred_data[common_cols].reindex(columns=true_data.columns)
            ctrl_data = ctrl_data[common_cols].reindex(columns=true_data.columns)
            
            # Calculate delta for all genes using all available data
            pred_mean = np.mean(pred_data.values, axis=0)
            true_mean = np.mean(true_data.values, axis=0)  
            ctrl_mean = np.mean(ctrl_data.values, axis=0)
            
            delta_metrics = calculate_gene_direction_consistency_single_cell(pred_mean, true_mean, ctrl_mean)
            all_results["allgene"].append({
                "dataset": mapped_dataset_name,
                "condition": condition,
                "model": "scFoundation",
                **delta_metrics
            })
            
            # Calculate Top20, Top50, Top100
            for top_n in [20, 50, 100]:
                deg_file = os.path.join(DEG_DIR, mapped_dataset_name, f"DEG_{condition}.csv")
                if not os.path.exists(deg_file):
                    continue
                    
                deg_df = pd.read_csv(deg_file)
                top_degs = deg_df.head(top_n)["names"].tolist()
                
                top_degs_in_true = [gene for gene in top_degs if gene in true_data.columns]
                if len(top_degs_in_true) == 0:
                    continue
                    
                top_pred_data = pred_data[top_degs_in_true].values
                top_true_data = true_data[top_degs_in_true].values
                top_ctrl_data = ctrl_data[top_degs_in_true].values
                
                top_pred_mean = np.mean(top_pred_data, axis=0)
                top_true_mean = np.mean(top_true_data, axis=0)
                top_ctrl_mean = np.mean(top_ctrl_data, axis=0)
                
                top_delta_metrics = calculate_gene_direction_consistency_single_cell(top_pred_mean, top_true_mean, top_ctrl_mean)
                all_results[f"top{top_n}"].append({
                    "dataset": mapped_dataset_name,
                    "condition": condition,
                    "model": "scFoundation",
                    **top_delta_metrics
                })
    
    return ("scFoundation", all_results)

def process_gears():
    """Process GEARS model without sampling"""
    logging.info("Processing GEARS model...")
    
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
    
    PRED_ROOT = "/data1/yy_data/pert/gears/save2"
    H5AD_ROOT = "/data2/lanxiang/data/Task1_data"
    DEG_DIR = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    # Process only datasets in manual_mapping
    for subfolder in os.listdir(PRED_ROOT):
        if subfolder not in manual_mapping:
            continue

        folder_path = os.path.join(PRED_ROOT, subfolder)
        if not os.path.isdir(folder_path):
            continue

        mapped_dataset_name = manual_mapping[subfolder]
        pred_folder = os.path.join(folder_path, 'results')
        if not os.path.isdir(pred_folder):
            continue

        h5ad_path = find_h5ad_file(subfolder, H5AD_ROOT, manual_mapping)
        if h5ad_path is None:
            continue

        adata = sc.read_h5ad(h5ad_path)
        
        # Use ALL control data without sampling
        ctrl_mask = adata.obs['condition'] == 'ctrl'
        ctrl_data_all = adata[ctrl_mask].X
        if ctrl_data_all.shape[0] == 0:
            continue
        ctrl_data_all = ctrl_data_all.toarray() if hasattr(ctrl_data_all, 'toarray') else ctrl_data_all

        for pred_file in os.listdir(pred_folder):
            if not pred_file.endswith('.npz'):
                continue

            condition = parse_condition_from_filename(pred_file)
            
            # Use ALL true data without sampling
            cond_mask = adata.obs['condition'] == condition
            true_data_all = adata[cond_mask].X
            if true_data_all.shape[0] == 0:
                continue
            true_data_all = true_data_all.toarray() if hasattr(true_data_all, 'toarray') else true_data_all

            pred_data = load_prediction(os.path.join(pred_folder, pred_file))
            if pred_data is None:
                continue

            # Calculate delta for all genes (no sampling)
            delta_metrics = calculate_gene_direction_consistency(pred_data, true_data_all, ctrl_data_all)
            all_results["allgene"].append({
                "dataset": mapped_dataset_name,
                "condition": condition,
                "model": "GEARS",
                **delta_metrics
            })

            # Calculate Top20, Top50, Top100
            for top_n in [20, 50, 100]:
                deg_file = os.path.join(DEG_DIR, mapped_dataset_name, f"DEG_{condition}.csv")
                if not os.path.exists(deg_file):
                    continue

                deg_df = pd.read_csv(deg_file)
                top_degs = deg_df.head(top_n)["names"].tolist()

                try:
                    top_pred_data = pred_data[:, [adata.var_names.get_loc(g) for g in top_degs if g in adata.var_names]]
                    top_true_data = true_data_all[:, [adata.var_names.get_loc(g) for g in top_degs if g in adata.var_names]]
                    top_ctrl_data = ctrl_data_all[:, [adata.var_names.get_loc(g) for g in top_degs if g in adata.var_names]]

                    top_delta_metrics = calculate_gene_direction_consistency(top_pred_data, top_true_data, top_ctrl_data)
                    all_results[f"top{top_n}"].append({
                        "dataset": mapped_dataset_name,
                        "condition": condition,
                        "model": "GEARS",
                        **top_delta_metrics
                    })
                except Exception as e:
                    logging.error(f"Error processing top{top_n} for {mapped_dataset_name}/{condition}: {e}")
    
    return ("GEARS", all_results)

def process_perturbnet():
    """Process Perturbnet model without sampling"""
    logging.info("Processing Perturbnet model...")
    perturbnet_dir = "/data2/lanxiang/perturb_benchmark_v2/model/PerturbNet"
    raw_data_dir = "/data2/lanxiang/data/Task1_data"
    deg_dir = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    for dataset in os.listdir(perturbnet_dir):
        dataset_path = os.path.join(perturbnet_dir, dataset, "perturbnet_predictions")
        raw_file = os.path.join(raw_data_dir, f"{dataset}.h5ad")
        if not os.path.exists(raw_file):
            continue

        adata = sc.read_h5ad(raw_file)

        # Use ALL control data without sampling
        ctrl_mask = adata.obs["condition"] == "ctrl"
        if "counts" in adata.layers:
            ctrl_data = adata[ctrl_mask].layers["counts"]
        else:
            ctrl_data = adata[ctrl_mask].X
        ctrl_data = ctrl_data.toarray() if not isinstance(ctrl_data, np.ndarray) else ctrl_data
        ctrl_data = np.log1p(ctrl_data / ctrl_data.sum(axis=1, keepdims=True) * 1e4)

        for file in os.listdir(dataset_path):
            if not file.endswith("_predict.npy"):
                continue
            condition = file.replace("_predict.npy", "")
            pred_file = os.path.join(dataset_path, f"{condition}_predict.npy")
            real_file = os.path.join(dataset_path, f"{condition}_real.npy")

            if not os.path.exists(real_file):
                continue

            try:
                pred_data = np.load(pred_file)
                true_data = np.load(real_file)

                # Apply consistent preprocessing
                pred_data = pred_data.toarray() if not isinstance(pred_data, np.ndarray) else pred_data
                true_data = true_data.toarray() if not isinstance(true_data, np.ndarray) else true_data

                pred_data = np.log1p(pred_data / pred_data.sum(axis=1, keepdims=True) * 1e4)
                true_data = np.log1p(true_data / true_data.sum(axis=1, keepdims=True) * 1e4)

                # Calculate delta for all genes (no sampling)
                delta_metrics = calculate_gene_direction_consistency(pred_data, true_data, ctrl_data)
                all_results["allgene"].append({
                    "dataset": dataset,
                    "condition": condition,
                    "model": "PerturbNet",
                    **delta_metrics
                })

                # Calculate Top20, Top50, Top100
                for top_n in [20, 50, 100]:
                    deg_file = os.path.join(deg_dir, dataset, f"DEG_{condition}.csv")
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
                        "model": "PerturbNet",
                        **top_delta_metrics
                    })

            except Exception as e:
                logging.error(f"Failed on {dataset} / {condition}: {e}")
    
    return ("PerturbNet", all_results)

def process_scelmo():
    """Process scELMO model without sampling"""
    logging.info("Processing scELMO model...")
    
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
    
    PRED_ROOT = "/data1/yy_data/pert/scelmo/save2"
    H5AD_ROOT = "/data2/lanxiang/data/Task1_data"
    DEG_DIR = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
    
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    for subfolder in os.listdir(PRED_ROOT):
        if subfolder not in manual_mapping:
            continue

        folder_path = os.path.join(PRED_ROOT, subfolder)
        if not os.path.isdir(folder_path):
            continue

        mapped_dataset_name = manual_mapping[subfolder]
        pred_folder = os.path.join(folder_path, 'results')
        if not os.path.isdir(pred_folder):
            continue

        h5ad_path = find_h5ad_file(subfolder, H5AD_ROOT, manual_mapping)
        if h5ad_path is None:
            continue

        adata = sc.read_h5ad(h5ad_path)

        # Use ALL control data without sampling
        ctrl_mask = adata.obs['condition'] == 'ctrl'
        ctrl_data_all = adata[ctrl_mask].X
        if ctrl_data_all.shape[0] == 0:
            continue
        ctrl_data_all = ctrl_data_all.toarray() if hasattr(ctrl_data_all, 'toarray') else ctrl_data_all

        for pred_file in os.listdir(pred_folder):
            if not pred_file.endswith('.npz'):
                continue

            condition = parse_condition_from_filename(pred_file)

            # Use ALL true data without sampling
            cond_mask = adata.obs['condition'] == condition
            true_data_all = adata[cond_mask].X
            if true_data_all.shape[0] == 0:
                continue
            true_data_all = true_data_all.toarray() if hasattr(true_data_all, 'toarray') else true_data_all

            pred_data = load_prediction(os.path.join(pred_folder, pred_file))
            if pred_data is None:
                continue

            # Calculate delta for all genes (no sampling)
            delta_metrics = calculate_gene_direction_consistency(pred_data, true_data_all, ctrl_data_all)
            all_results["allgene"].append({
                "dataset": mapped_dataset_name,
                "condition": condition,
                "model": "scELMO",
                **delta_metrics
            })

            # Calculate Top20, Top50, Top100
            for top_n in [20, 50, 100]:
                deg_file = os.path.join(DEG_DIR, mapped_dataset_name, f"DEG_{condition}.csv")
                if not os.path.exists(deg_file):
                    continue

                deg_df = pd.read_csv(deg_file)
                top_degs = deg_df.head(top_n)["names"].tolist()

                try:
                    top_pred_data = pred_data[:, [adata.var_names.get_loc(g) for g in top_degs if g in adata.var_names]]
                    top_true_data = true_data_all[:, [adata.var_names.get_loc(g) for g in top_degs if g in adata.var_names]]
                    top_ctrl_data = ctrl_data_all[:, [adata.var_names.get_loc(g) for g in top_degs if g in adata.var_names]]

                    top_delta_metrics = calculate_gene_direction_consistency(top_pred_data, top_true_data, top_ctrl_data)
                    all_results[f"top{top_n}"].append({
                        "dataset": mapped_dataset_name,
                        "condition": condition,
                        "model": "scELMO",
                        **top_delta_metrics
                    })
                except Exception as e:
                    logging.error(f"Error processing top{top_n} for {mapped_dataset_name}/{condition}: {e}")
    
    return ("scELMO", all_results)

def process_scgpt_epoch2():
    """Process scGPT epoch2 model without sampling"""
    logging.info("Processing scGPT epoch2 model...")
    
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
    
    PRED_ROOT = "/data1/yy_data/pert/scgpt/save"
    H5AD_ROOT = "/data2/lanxiang/data/Task1_data"
    DEG_DIR = "/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted"
    
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    for subfolder in os.listdir(PRED_ROOT):
        folder_path = os.path.join(PRED_ROOT, subfolder)
        if not os.path.isdir(folder_path):
            continue

        mapped_dataset_name = manual_mapping.get(subfolder, subfolder)
        pred_folder = os.path.join(folder_path, 'results_epoch2')
        if not os.path.isdir(pred_folder):
            continue

        h5ad_path = find_h5ad_file(subfolder, H5AD_ROOT, manual_mapping)
        if h5ad_path is None:
            continue

        adata = sc.read_h5ad(h5ad_path)

        # Use ALL control data without sampling
        ctrl_mask = adata.obs['condition'] == 'ctrl'
        ctrl_data_all = adata[ctrl_mask].X
        if ctrl_data_all.shape[0] == 0:
            continue
        ctrl_data_all = ctrl_data_all.toarray() if hasattr(ctrl_data_all, 'toarray') else ctrl_data_all

        for pred_file in os.listdir(pred_folder):
            if not pred_file.endswith(('.npz', '.npy.gz.npz', '.npy')):
                continue

            condition = pred_file
            for ext in ['.npy.gz.npz', '.npz', '.npy']:
                if condition.endswith(ext):
                    condition = condition[:-len(ext)]
                    break

            # Use ALL true data without sampling
            cond_mask = adata.obs['condition'] == condition
            true_data_all = adata[cond_mask].X
            if true_data_all.shape[0] == 0:
                continue
            true_data_all = true_data_all.toarray() if hasattr(true_data_all, 'toarray') else true_data_all

            pred_data = load_prediction(os.path.join(pred_folder, pred_file))
            if pred_data is None:
                continue

            # Calculate delta for all genes (no sampling)
            delta_metrics = calculate_gene_direction_consistency(pred_data, true_data_all, ctrl_data_all)
            all_results["allgene"].append({
                "dataset": mapped_dataset_name,
                "condition": condition,
                "model": "scGPT_epoch2",
                **delta_metrics
            })

            # Calculate Top20, Top50, Top100
            for top_n in [20, 50, 100]:
                deg_file = os.path.join(DEG_DIR, mapped_dataset_name, f"DEG_{condition}.csv")
                if not os.path.exists(deg_file):
                    continue

                deg_df = pd.read_csv(deg_file)
                top_degs = deg_df.head(top_n)["names"].tolist()

                try:
                    top_pred_data = pred_data[:, [adata.var_names.get_loc(g) for g in top_degs]]
                    top_true_data = true_data_all[:, [adata.var_names.get_loc(g) for g in top_degs]]
                    top_ctrl_data = ctrl_data_all[:, [adata.var_names.get_loc(g) for g in top_degs]]

                    top_delta_metrics = calculate_gene_direction_consistency(top_pred_data, top_true_data, top_ctrl_data)
                    all_results[f"top{top_n}"].append({
                        "dataset": mapped_dataset_name,
                        "condition": condition,
                        "model": "scGPT_epoch2",
                        **top_delta_metrics
                    })
                except Exception as e:
                    logging.error(f"Error processing top{top_n} for {mapped_dataset_name}/{condition}: {e}")
    
    return ("scGPT_epoch2", all_results)

def process_scgpt_epoch5():
    """Process scGPT epoch5 model without sampling"""
    logging.info("Processing scGPT epoch5 model...")
    
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
    
    PRED_ROOT = "/data1/yy_data/pert/scgpt/save"
    H5AD_ROOT = "/data2/lanxiang/data/Task1_data"
    DEG_DIR = "/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted"
    
    all_results = {
        "allgene": [],
        "top20": [],
        "top50": [],
        "top100": []
    }
    
    for subfolder in os.listdir(PRED_ROOT):
        folder_path = os.path.join(PRED_ROOT, subfolder)
        if not os.path.isdir(folder_path):
            continue

        mapped_dataset_name = manual_mapping.get(subfolder, subfolder)
        pred_folder = os.path.join(folder_path, 'results')
        if not os.path.isdir(pred_folder):
            continue

        h5ad_path = find_h5ad_file(subfolder, H5AD_ROOT, manual_mapping)
        if h5ad_path is None:
            continue

        adata = sc.read_h5ad(h5ad_path)

        # Use ALL control data without sampling
        ctrl_mask = adata.obs['condition'] == 'ctrl'
        ctrl_data_all = adata[ctrl_mask].X
        if ctrl_data_all.shape[0] == 0:
            continue
        ctrl_data_all = ctrl_data_all.toarray() if hasattr(ctrl_data_all, 'toarray') else ctrl_data_all

        for pred_file in os.listdir(pred_folder):
            if not pred_file.endswith(('.npz', '.npy.gz.npz', '.npy')):
                continue

            condition = pred_file
            for ext in ['.npy.gz.npz', '.npz', '.npy']:
                if condition.endswith(ext):
                    condition = condition[:-len(ext)]
                    break

            # Use ALL true data without sampling
            cond_mask = adata.obs['condition'] == condition
            true_data_all = adata[cond_mask].X
            if true_data_all.shape[0] == 0:
                continue
            true_data_all = true_data_all.toarray() if hasattr(true_data_all, 'toarray') else true_data_all

            pred_data = load_prediction(os.path.join(pred_folder, pred_file))
            if pred_data is None:
                continue

            # Calculate delta for all genes (no sampling)
            delta_metrics = calculate_gene_direction_consistency(pred_data, true_data_all, ctrl_data_all)
            all_results["allgene"].append({
                "dataset": mapped_dataset_name,
                "condition": condition,
                "model": "scGPT_epoch5",
                **delta_metrics
            })

            # Calculate Top20, Top50, Top100
            for top_n in [20, 50, 100]:
                deg_file = os.path.join(DEG_DIR, mapped_dataset_name, f"DEG_{condition}.csv")
                if not os.path.exists(deg_file):
                    continue

                deg_df = pd.read_csv(deg_file)
                top_degs = deg_df.head(top_n)["names"].tolist()

                try:
                    top_pred_data = pred_data[:, [adata.var_names.get_loc(g) for g in top_degs]]
                    top_true_data = true_data_all[:, [adata.var_names.get_loc(g) for g in top_degs]]
                    top_ctrl_data = ctrl_data_all[:, [adata.var_names.get_loc(g) for g in top_degs]]

                    top_delta_metrics = calculate_gene_direction_consistency(top_pred_data, top_true_data, top_ctrl_data)
                    all_results[f"top{top_n}"].append({
                        "dataset": mapped_dataset_name,
                        "condition": condition,
                        "model": "scGPT_epoch5",
                        **top_delta_metrics
                    })
                except Exception as e:
                    logging.error(f"Error processing top{top_n} for {mapped_dataset_name}/{condition}: {e}")
    
    return ("scGPT_epoch5", all_results)

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
            output_path = os.path.join(output_root, key.capitalize(), f"All_models_delta_{key}.csv")
            df.to_csv(output_path, index=False)
            logging.info(f"All models {key} delta results saved: {len(df)} records")
    
    # Save individual model results
    for model_name, results in model_results_list:
        for key, result in results.items():
            if result:
                df = pd.DataFrame(result)
                output_path = os.path.join(output_root, key.capitalize(), f"{model_name}_delta_{key}.csv")
                df.to_csv(output_path, index=False)
                logging.info(f"{model_name} {key} delta results saved: {len(df)} records")

def main():
    output_root = "/data2/lanxiang/perturb_benchmark_v2/Delta_gene"
    
    # Define all model processing functions
    model_functions = [
        process_baseline,
        process_biolord,
        process_cpa_task2,
        process_scfoundation,
        process_gears,
        process_perturbnet,
        process_scelmo,
        process_scgpt_epoch2,
        process_scgpt_epoch5
    ]
    
    logging.info(f"Starting delta calculation with {len(model_functions)} models using parallel processing (NO SAMPLING)")
    
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
    
    logging.info("=== Delta calculation completed for all models (NO SAMPLING) ===")
    
    # Print summary
    for gene_subset in ['Allgene', 'Top20', 'Top50', 'Top100']:
        folder_path = os.path.join(output_root, gene_subset)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            logging.info(f"{gene_subset}: {len(files)} model results")

if __name__ == "__main__":
    main()