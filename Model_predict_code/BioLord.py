import os
import sys
import scanpy as sc
import anndata
import numpy as np
import pandas as pd
import pickle
import warnings
import gc
import time
from pathlib import Path
from datetime import datetime, timedelta

import biolord
import torch

# Set GPU device and suppress warnings
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
warnings.filterwarnings('ignore')

# ================================
# Configuration Paths
# ================================
DATA_DIR = "/data2/lanxiang/data/Task1_data"
GO_ESSENTIAL_PATH = "/data2/yue_data/pert/data/go_essential_all/go_essential_all.csv"
ESSENTIAL_GENES_PATH = "/data2/yue_data/pert/data/essential_all_data_pert_genes.pkl"
SPLITS_BASE_PATH = "/data2/yue_data/pert/data"
OUTPUT_DIR = "/data2/lanxiang/perturb_benchmark_v2/model/Biolord"

# Dataset mapping (consistent with Task1_loop.py)
DATASET_MAPPING = {
    "Adamson.h5ad": "perturb_processed_adamson",
    "Arce_MM_CRISPRi_sub.h5ad": "arce_mm_crispri_filtered",
    "DatlingerBock2017.h5ad": "datlingerbock2017",
    "DatlingerBock2021.h5ad": "datlingerbock2021",
    "DixitRegev2016.h5ad": "dixitregev2016",
    "FrangiehIzar2021_RNA.h5ad": "frangiehizar2021_rna",
    "Junyue_Cao.h5ad": "junyue_cao_filtered",
    "NormanWeissman2019_filtered.h5ad": "normanweissman2019_filtered",
    "PapalexiSatija2021_eccite_RNA.h5ad": "papalexisatija2021_eccite_rna",
    "ReplogleWeissman2022_rpe1.h5ad": "replogleweissman2022_rpe1",
    "Sunshine2023_CRISPRi_sarscov2.h5ad": "sunshine2023_crispri_sarscov2",
    "TianKampmann2021_CRISPRa.h5ad": "tiankampmann2021_crispra",
    "TianKampmann2021_CRISPRi.h5ad": "tiankampmann2021_crispri",
    "vcc_train_filtered.h5ad": "vcc_train_filtered",
    "Fengzhang2023.h5ad": "fengzhang2023"
}


def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    # Force wait to ensure memory cleanup
    time.sleep(2)


def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")


def force_cleanup():
    """Force cleanup of all memory"""
    print("🧹 Performing forced memory cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()  # Clear IPC cache
    time.sleep(3)  # Wait longer
    print_gpu_memory()


def format_time(seconds):
    """Format time display"""
    return str(timedelta(seconds=int(seconds)))


def log_time(dataset_name, elapsed_time, log_file):
    """Log runtime to file"""
    formatted_time = format_time(elapsed_time)
    
    # Create log directory
    log_path = Path(log_file).parent
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Write time record
    with open(log_file, 'a', encoding='utf-8') as f:
        # Write header if file is empty
        if os.path.getsize(log_file) == 0:
            f.write("dataset\ttime\n")
        f.write(f"{dataset_name}\t{formatted_time}\n")
    
    print(f"✅ Time record saved: {dataset_name} -> {formatted_time}")


def load_data_and_splits(dataset_name):
    """Load specified dataset and split information (consistent with Task1_loop.py)"""
    # Build data path
    data_path = f"{DATA_DIR}/{dataset_name}.h5ad"
    
    # Get corresponding folder name
    folder_name = DATASET_MAPPING[f"{dataset_name}.h5ad"]
    
    # Build split file path
    split_path = f"{SPLITS_BASE_PATH}/{folder_name}/splits/{folder_name}_simulation_1_0.75.pkl"
    
    print(f"📁 Loading data: {data_path}")
    print(f"📁 Loading splits: {split_path}")
    
    # Check if files exist
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    
    # Load data
    adata = sc.read(data_path)
    
    # Load split information
    with open(split_path, 'rb') as f:
        splits = pickle.load(f)
    
    print(f"Data shape: {adata.shape}")
    print(f"Available columns: {adata.obs.columns.tolist()}")
    
    return adata, splits


def create_split_column(adata, splits):
    """Create split column based on splits information (completely consistent with Task1_loop.py)"""
    # Initialize split column
    adata.obs['split'] = 'unknown'
    
    # Assign splits based on condition
    train_conditions = splits.get('train', [])
    test_conditions = splits.get('test', [])
    val_conditions = splits.get('val', [])
    
    print(f"Train conditions: {len(train_conditions)}")
    print(f"Test conditions: {len(test_conditions)}")  
    print(f"Val conditions: {len(val_conditions)}")
    
    # Check if ctrl is in train conditions
    if 'ctrl' in train_conditions:
        print("✅ ctrl condition in training set")
    else:
        print("⚠️ Warning: ctrl condition not in training set")
    
    # Assign to corresponding splits
    adata.obs.loc[adata.obs['condition'].isin(train_conditions), 'split'] = 'train'
    adata.obs.loc[adata.obs['condition'].isin(test_conditions), 'split'] = 'test'
    adata.obs.loc[adata.obs['condition'].isin(val_conditions), 'split'] = 'val'
    
    print(f"Split distribution:")
    print(adata.obs['split'].value_counts())
    
    return adata, test_conditions


def limit_ctrl_cells(adata, max_ctrl_cells=1000, random_seed=42):
    """Limit ctrl cell count (consistent with Task1_loop.py)"""
    ctrl_mask = adata.obs['condition'] == 'ctrl'
    ctrl_count = ctrl_mask.sum()
    
    if ctrl_count > max_ctrl_cells:
        print(f"⚠️ Ctrl cell count({ctrl_count}) exceeds limit({max_ctrl_cells}), randomly selecting {max_ctrl_cells}")
        
        # Get ctrl cell indices
        ctrl_indices = adata.obs.index[ctrl_mask].tolist()
        
        # Set random seed and randomly select
        np.random.seed(random_seed)
        selected_ctrl_indices = np.random.choice(ctrl_indices, size=max_ctrl_cells, replace=False)
        
        # Get non-ctrl cells
        non_ctrl_mask = adata.obs['condition'] != 'ctrl'
        non_ctrl_indices = adata.obs.index[non_ctrl_mask].tolist()
        
        # Merge indices
        final_indices = list(selected_ctrl_indices) + non_ctrl_indices
        
        # Filter data
        adata = adata[final_indices].copy()
        
        print(f"✅ Filtered data shape: {adata.shape}")
        print(f"✅ Filtered Ctrl cell count: {(adata.obs['condition'] == 'ctrl').sum()}")
    else:
        print(f"✅ Ctrl cell count({ctrl_count}) within limit, no filtering needed")
    
    return adata


def preprocess_original_adata(adata, splits):
    """Preprocess original adata, add necessary metadata"""
    # 1. Create condition_name
    adata.obs['condition_name'] = adata.obs['condition'].copy()
    
    # 2. Create perturbation and perturbation_rep columns
    perturbations = []
    perturbation_reps = []
    
    for condition in adata.obs['condition']:
        if condition == 'ctrl':
            perturbations.append('ctrl')
            perturbation_reps.append('ctrl')
        elif '+ctrl' in condition:
            # Single gene perturbation, e.g. EGTD+ctrl
            gene = condition.replace('+ctrl', '')
            perturbations.append(gene)
            perturbation_reps.append('ctrl')
        elif '+' in condition and '+ctrl' not in condition:
            # Combinatorial perturbation, e.g. SHEG+SHDW
            genes = condition.split('+')
            perturbations.append(genes[0])  # First gene
            perturbation_reps.append(genes[1])  # Second gene
        else:
            # Other cases
            perturbations.append(condition)
            perturbation_reps.append('ctrl')
    
    adata.obs['perturbation'] = pd.Categorical(perturbations)
    adata.obs['perturbation_rep'] = pd.Categorical(perturbation_reps)
    
    # Ensure categories are consistent
    all_categories = list(set(perturbations + perturbation_reps))
    adata.obs['perturbation'] = adata.obs['perturbation'].cat.add_categories(
        [cat for cat in all_categories if cat not in adata.obs['perturbation'].cat.categories]
    )
    adata.obs['perturbation_rep'] = adata.obs['perturbation_rep'].cat.add_categories(
        [cat for cat in all_categories if cat not in adata.obs['perturbation_rep'].cat.categories]
    )
    
    # 3. Add split information
    for seed in range(1, 6):
        adata.obs[f'split{seed}'] = adata.obs['split'].copy()
    
    return adata


def create_pert2neighbor_mapping(adata):
    """Create perturbation to neighbor mapping"""
    print("Creating perturbation to neighbor mapping...")
    
    # Load GO and essential gene data
    df_go = pd.read_csv(GO_ESSENTIAL_PATH)
    with open(ESSENTIAL_GENES_PATH, 'rb') as f:
        essential_genes = pickle.load(f)
    
    # Filter available genes
    available_genes = list(set(essential_genes) & set(adata.var.index))
    df_go_filtered = df_go[df_go["source"].isin(available_genes)].copy()
    
    print(f"Available gene count: {len(available_genes)}")
    
    # Get top 20 important sources for each target
    df_go_top = df_go_filtered.groupby('target').apply(
        lambda x: x.nlargest(21, ['importance'])
    ).reset_index(drop=True)
    
    def get_neighbor_map(pert):
        """Get neighbor mapping for specific perturbation"""
        neighbor_map = np.zeros(len(available_genes))
        if pert in df_go_top['target'].values:
            sources = df_go_top[df_go_top['target'] == pert]['source'].values
            importances = df_go_top[df_go_top['target'] == pert]['importance'].values
            
            for source, importance in zip(sources, importances):
                if source in available_genes:
                    idx = available_genes.index(source)
                    neighbor_map[idx] = importance
        
        return neighbor_map
    
    # Create perturbation to neighbor mapping
    pert2neighbor = {}
    unique_perts = list(adata.obs['perturbation'].cat.categories)
    
    for pert in unique_perts:
        pert2neighbor[pert] = get_neighbor_map(pert)
    
    adata.uns['pert2neighbor'] = pert2neighbor
    
    # Create keep_idx with different thresholds
    pert2neighbor_array = np.array([val for val in pert2neighbor.values()])
    keep_idx = pert2neighbor_array.sum(0) > 0
    keep_idx1 = pert2neighbor_array.sum(0) > 1
    
    return adata, keep_idx, keep_idx1, available_genes


def create_adata_single(adata, keep_idx, keep_idx1):
    """Create adata_single (each row is average expression of one condition) - completely following official logic"""
    print("Creating adata_single (condition average expression data)...")
    
    # 1. Calculate average expression for each condition
    df_perts_expression = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
        index=adata.obs_names, 
        columns=adata.var_names
    )
    df_perts_expression['condition'] = adata.obs['condition']
    df_perts_expression = df_perts_expression.groupby(['condition']).mean()
    df_perts_expression = df_perts_expression.reset_index()
    
    # 2. Separate single and combinatorial perturbations following official logic
    single_perts_condition = []
    single_pert_val = []
    double_perts = []
    
    for pert in adata.obs['condition'].cat.categories:
        if len(pert.split("+")) == 1:
            # Single condition (like ctrl)
            continue
        elif "ctrl" in pert:
            # Condition containing ctrl (single gene perturbation)
            single_perts_condition.append(pert)
            p1, p2 = pert.split("+")
            if p2 == "ctrl":
                single_pert_val.append(p1)
            else:
                single_pert_val.append(p2)
        else:
            # Combinatorial perturbation
            double_perts.append(pert)
    
    # Add ctrl
    single_perts_condition.append("ctrl")
    single_pert_val.append("ctrl")
    
    print(f"Single perturbation condition count: {len(single_perts_condition)}")
    print(f"Combinatorial perturbation condition count: {len(double_perts)}")
    
    # 3. Create single perturbation expression matrix
    df_singleperts_expression = pd.DataFrame(
        df_perts_expression.set_index("condition").loc[single_perts_condition].values,
        index=single_pert_val
    )
    
    # 4. Create perturbation neighbor embeddings
    df_singleperts_emb1 = np.array([
        adata.uns['pert2neighbor'][p1][keep_idx1] 
        for p1 in df_singleperts_expression.index
    ])
    
    # 5. Create adata_single
    adata_single = anndata.AnnData(
        X=df_singleperts_expression.values,
        var=adata.var.copy(),
        dtype=df_singleperts_expression.values.dtype
    )
    
    adata_single.obs_names = pd.Index(single_perts_condition)
    adata_single.obs['condition'] = pd.Categorical(single_perts_condition)
    adata_single.obs['perts_name'] = pd.Categorical(single_pert_val)
    
    # Add perturbation neighbor embeddings
    adata_single.obsm['perturbation_neighbors1'] = df_singleperts_emb1
    
    # 6. Add split information
    for seed in range(1, 6):
        adata_single.obs[f'split{seed}'] = None
        
        for split_type in ['train', 'test', 'val']:
            # Rename split to match official naming
            if split_type == 'test':
                split_name = 'ood'
            elif split_type == 'val':  
                split_name = 'test'
            else:
                split_name = split_type
                
            split_conditions = adata[adata.obs[f'split{seed}'] == split_type].obs['condition'].unique()
            mask = adata_single.obs['condition'].isin(split_conditions)
            adata_single.obs.loc[mask, f'split{seed}'] = split_name
    
    return adata_single, double_perts


def bool2idx(bool_array):
    """Convert boolean array to indices"""
    return np.where(bool_array)[0]


def repeat_n(tensor, n):
    """Repeat tensor n times"""
    if hasattr(tensor, 'repeat'):
        return tensor.repeat(n, 1)
    else:
        return np.tile(tensor, (n, 1))


def get_config_params():
    """Get configuration parameters (completely copied from norman_optimal_config.py)"""
    varying_arg = {
        "seed": 42,
        "unknown_attribute_noise_param": 0.2, 
        "use_batch_norm": False,
        "use_layer_norm": False, 
        "step_size_lr": 45, 
        "attribute_dropout_rate": 0.0, 
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
        "n_latent": 32, 
        "n_latent_attribute_ordered": 32,
        "reconstruction_penalty": 10000.0,
        "attribute_nn_width": 64,
        "attribute_nn_depth": 2, 
        "attribute_nn_lr": 0.001, 
        "attribute_nn_wd": 4e-8,
        "latent_lr": 0.01,
        "latent_wd": 0.00001,
        "decoder_width": 32,
        "decoder_depth": 2,  
        "decoder_activation": True,
        "attribute_nn_activation": True,
        "unknown_attributes": False,
        "decoder_lr": 0.01,
        "decoder_wd": 0.01,
        "max_epochs": 1000,
        "early_stopping_patience": 200,
        "ordered_attributes_key": "perturbation_neighbors1",
        "n_latent_attribute_categorical": 16,
        "unknown_attribute_penalty": 1000.0,  # Add missing parameter
    }
    return varying_arg


def train_biolord_model(adata_single, seed=1):
    """Train biolord model (completely following official method)"""
    print(f"Training biolord model (seed {seed})...")
    
    # Get configuration parameters
    varying_arg = get_config_params()
    
    # Build parameters following official method
    module_params = {
        "attribute_nn_width": varying_arg["attribute_nn_width"],
        "attribute_nn_depth": varying_arg["attribute_nn_depth"],
        "use_batch_norm": varying_arg["use_batch_norm"],
        "use_layer_norm": varying_arg["use_layer_norm"],
        "attribute_dropout_rate": varying_arg["attribute_dropout_rate"],
        "unknown_attribute_noise_param": varying_arg["unknown_attribute_noise_param"],
        "seed": varying_arg["seed"],
        "n_latent_attribute_ordered": varying_arg["n_latent_attribute_ordered"],
        "n_latent_attribute_categorical": varying_arg["n_latent_attribute_categorical"],
        "reconstruction_penalty": varying_arg["reconstruction_penalty"],
        "unknown_attribute_penalty": varying_arg["unknown_attribute_penalty"],
        "decoder_width": varying_arg["decoder_width"],
        "decoder_depth": varying_arg["decoder_depth"],
        "decoder_activation": varying_arg["decoder_activation"],
        "attribute_nn_activation": varying_arg["attribute_nn_activation"],
        "unknown_attributes": varying_arg["unknown_attributes"],
    }

    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": varying_arg["latent_lr"],
        "latent_wd": varying_arg["latent_wd"],
        "attribute_nn_lr": varying_arg["attribute_nn_lr"],
        "attribute_nn_wd": varying_arg["attribute_nn_wd"],
        "step_size_lr": varying_arg["step_size_lr"],
        "cosine_scheduler": varying_arg["cosine_scheduler"],
        "scheduler_final_lr": varying_arg["scheduler_final_lr"],
        "decoder_lr": varying_arg["decoder_lr"],
        "decoder_wd": varying_arg["decoder_wd"]
    }
    
    # Read ordered_attributes_key from configuration following official method
    ordered_attributes_key = varying_arg["ordered_attributes_key"]
    
    # Set biolord data (following official method)
    biolord.Biolord.setup_anndata(
        adata_single,
        ordered_attributes_keys=[ordered_attributes_key],
        categorical_attributes_keys=None,
        retrieval_attribute_key=None,
    )
    
    print_gpu_memory()  # Check memory before training
    
    # Create model (following official method)
    model = biolord.Biolord(
        adata=adata_single,
        n_latent=varying_arg["n_latent"],
        model_name="norman",
        module_params=module_params,
        train_classifiers=False,
        split_key=f"split{seed}"
    )
    
    # Train model (following official method)
    model.train(
        max_epochs=int(varying_arg["max_epochs"]),
        batch_size=32,
        plan_kwargs=trainer_params,
        early_stopping=True,
        early_stopping_patience=int(varying_arg["early_stopping_patience"]),
        check_val_every_n_epoch=5,
        num_workers=1,
        enable_checkpointing=False,
    )
    
    print_gpu_memory()  # Check memory after training
    
    return model, varying_arg


def predict_test_conditions_official_way(model, adata, adata_single, test_conditions, varying_arg, seed=1):
    """Predict test conditions following official method (completely copied official logic)"""
    print(f"Predicting test conditions following official method (seed {seed})...")
    
    # Get training data information
    train_conditions = adata[adata.obs[f'split{seed}'] == 'train'].obs['condition'].unique()
    
    # Calculate average expression data (reference official code)
    df_perts_expression = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
        index=adata.obs_names, 
        columns=adata.var_names
    )
    df_perts_expression['condition'] = adata.obs['condition']
    df_perts_expression = df_perts_expression.groupby(['condition']).mean().reset_index()
    
    # Separate single and combinatorial perturbations (reference official code)
    single_perts_condition = []
    single_pert_val = []
    double_perts = []
    
    for pert in adata.obs['condition'].cat.categories:
        if len(pert.split("+")) == 1:
            continue
        elif "ctrl" in pert:
            single_perts_condition.append(pert)
            p1, p2 = pert.split("+")
            if p2 == "ctrl":
                single_pert_val.append(p1)
            else:
                single_pert_val.append(p2)
        else:
            double_perts.append(pert)
    
    single_perts_condition.append("ctrl")
    single_pert_val.append("ctrl")
    
    df_singleperts_expression = pd.DataFrame(
        df_perts_expression.set_index("condition").loc[single_perts_condition].values,
        index=single_pert_val
    )
    df_doubleperts_expression = df_perts_expression.set_index("condition").loc[double_perts].values
    
    df_singleperts_condition = pd.Index(single_perts_condition)
    df_single_pert_val = pd.Index(single_pert_val)
    df_doubleperts_condition = pd.Index(double_perts)
    
    # Get conditions in training set
    train_idx = df_singleperts_condition.isin(train_conditions)
    train_condition_perts = df_singleperts_condition[train_idx]
    train_condition_perts_double = df_doubleperts_condition[df_doubleperts_condition.isin(train_conditions)]
    train_perts = df_single_pert_val[train_idx]
    
    # Prepare prediction
    adata_control = adata_single[adata_single.obs["condition"] == "ctrl"].copy()
    dataset_control = model.get_dataset(adata_control)
    dataset_reference = model.get_dataset(adata_single)
    n_obs = adata_control.shape[0]
    
    # ctrl baseline
    ctrl = np.array(adata[adata.obs["condition"] == "ctrl"].X.mean(0)).flatten()
    
    # Predict each test condition
    predictions_dict = {}
    ordered_attributes_key = varying_arg["ordered_attributes_key"]  # Read from configuration
    
    for pert in test_conditions:
        print(f"Predicting condition: {pert}")
        
        try:
            # Get real expression (for comparison)
            real_mask = (adata.obs['condition'] == pert) 
            if real_mask.sum() > 0:
                expression_pert = np.array(adata[real_mask].X.mean(0)).flatten()
            else:
                print(f"Warning: Cannot find real data for condition {pert}")
                continue
            
            # Predict following official logic
            if pert in train_condition_perts:
                # If in training set single perturbations
                idx_ref = bool2idx(adata_single.obs["condition"] == pert)[0]
                test_preds_delta = dataset_reference["X"][[idx_ref], :].mean(0).cpu().numpy()
                
            elif pert in train_condition_perts_double:
                # If in training set combinatorial perturbations
                test_preds_delta = df_doubleperts_expression[df_doubleperts_condition.isin([pert]), :]
                if len(test_preds_delta) > 0:
                    test_preds_delta = test_preds_delta[0]
                else:
                    test_preds_delta = ctrl
                    
            elif "ctrl" in pert:
                # Conditions containing ctrl (single gene perturbation)
                idx_ref = bool2idx(adata_single.obs["condition"] == pert)[0]
                dataset_pred = dataset_control.copy()
                dataset_pred[ordered_attributes_key] = repeat_n(
                    dataset_reference[ordered_attributes_key][idx_ref, :], n_obs
                )
                test_preds, _ = model.module.get_expression(dataset_pred)
                test_preds_delta = test_preds.cpu().numpy()
                
            else:
                # Combinatorial perturbation - use additive model
                test_preds_add = []
                for p in pert.split("+"):
                    if p in train_perts:
                        # If gene is in training set
                        test_predsp = df_singleperts_expression.values[
                            df_single_pert_val.isin([p]), :
                        ]
                        if len(test_predsp) > 0:
                            test_preds_add.append(test_predsp[0, :])
                    else:
                        # Use model prediction
                        idx_ref = bool2idx(adata_single.obs["perts_name"].isin([p]))
                        if len(idx_ref) > 0:
                            dataset_pred = dataset_control.copy()
                            dataset_pred[ordered_attributes_key] = repeat_n(
                                dataset_reference[ordered_attributes_key][idx_ref[0], :], n_obs
                            )
                            test_preds, _ = model.module.get_expression(dataset_pred)
                            test_preds_add.append(test_preds.cpu().numpy())
                
                if len(test_preds_add) == 2:
                    test_preds_delta = test_preds_add[0] + test_preds_add[1] - ctrl
                else:
                    test_preds_delta = ctrl
            
            # Ensure correct format
            if len(test_preds_delta.shape) > 1:
                test_preds_delta = test_preds_delta.flatten()
            
            predictions_dict[pert] = {
                'predicted': test_preds_delta,
                'real': expression_pert
            }
            
            print(f"✅ Successfully predicted {pert}")
            
        except Exception as e:
            print(f"❌ Error predicting condition {pert}: {e}")
            continue
    
    return predictions_dict


def save_results(predictions_dict, dataset_name, gene_names):
    """Save prediction results"""
    dataset_dir = Path(OUTPUT_DIR) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to: {dataset_dir}")
    
    # Save prediction and real data for each condition
    summary_data = []
    
    for condition, data in predictions_dict.items():
        pred_expr = data['predicted']
        real_expr = data['real']
        
        # Save prediction results
        pred_df = pd.DataFrame(
            pred_expr.reshape(1, -1), 
            columns=gene_names,
            index=[f"{condition}_predicted"]
        )
        pred_path = dataset_dir / f"{condition}_pred.csv"
        pred_df.to_csv(pred_path)
        
        # Save real data
        real_df = pd.DataFrame(
            real_expr.reshape(1, -1),
            columns=gene_names,
            index=[f"{condition}_real"]
        )
        real_path = dataset_dir / f"{condition}_real.csv"
        real_df.to_csv(real_path)
        
        summary_data.append({
            'condition': condition,
            'genes': len(gene_names),
            'prediction_file': f"{condition}_pred.csv",
            'real_data_file': f"{condition}_real.csv"
        })
        
        print(f"✅ Saved {condition}: prediction and real data")
        
        # Clean DataFrames to save memory
        del pred_df, real_df
    
    # Save summary information
    summary_df = pd.DataFrame(summary_data)
    summary_path = dataset_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n✅ Summary information saved to: {summary_path}")
    print("\nData summary:")
    print(summary_df.to_string(index=False))
    
    # Clean memory
    del summary_df
    gc.collect()
    
    return dataset_dir


def process_single_dataset(dataset_name, base_output_dir=OUTPUT_DIR, 
                          time_log_file=None):
    """Process single dataset"""
    if time_log_file is None:
        time_log_file = os.path.join(base_output_dir, "biolord_time.txt")
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting to process dataset: {dataset_name}")
    print(f"{'='*80}")
    
    # Force memory cleanup before processing
    force_cleanup()
    
    start_time = time.time()
    
    try:
        # 1. Load data and split information
        print("\n=== 1. Load data and split information ===")
        adata, splits = load_data_and_splits(dataset_name)
        
        # Limit ctrl cell count
        adata = limit_ctrl_cells(adata, max_ctrl_cells=1000, random_seed=42)
        
        # Create split column
        adata, test_conditions = create_split_column(adata, splits)
        
        # Clean memory
        clear_gpu_memory()
        print_gpu_memory()
        
        # 2. Preprocess original data
        print("\n=== 2. Preprocess original data ===")
        adata = preprocess_original_adata(adata, splits)
        adata, keep_idx, keep_idx1, available_genes = create_pert2neighbor_mapping(adata)
        
        # Clean memory
        clear_gpu_memory()
        
        # 3. Create adata_single (official method)
        print("\n=== 3. Create adata_single ===")
        adata_single, double_perts = create_adata_single(adata, keep_idx, keep_idx1)
        
        # Clean memory
        clear_gpu_memory()
        print_gpu_memory()
        
        # 4. Train model
        print("\n=== 4. Train model ===")
        model, varying_arg = train_biolord_model(adata_single, seed=1)
        
        # Clean memory immediately after training
        clear_gpu_memory()
        print_gpu_memory()
        
        # 5. Predict following official method
        print("\n=== 5. Predict test conditions ===")
        predictions_dict = predict_test_conditions_official_way(
            model, adata, adata_single, test_conditions, varying_arg, seed=1
        )
        
        # Clean memory after prediction
        clear_gpu_memory()
        
        # 6. Save results
        print("\n=== 6. Save results ===")
        result_dir = save_results(predictions_dict, dataset_name, adata.var_names.tolist())
        
        # Calculate and log runtime
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n🎉 {dataset_name} processing completed!")
        print(f"⏱️ Runtime: {format_time(elapsed_time)}")
        print(f"📁 Results saved in: {result_dir}")
        
        # Log time to file
        log_time(dataset_name, elapsed_time, time_log_file)
        
        success_rate = len(predictions_dict) / len(test_conditions) * 100 if test_conditions else 0
        print(f"📊 Prediction success rate: {success_rate:.1f}% ({len(predictions_dict)}/{len(test_conditions)})")
        
        # Force cleanup of all variables after completion
        pred_count = len(predictions_dict)
        cond_count = len(test_conditions)
        
        del adata, adata_single, model, predictions_dict
        if 'varying_arg' in locals():
            del varying_arg
        
        # Final force cleanup
        force_cleanup()
        
        return True, elapsed_time, pred_count, cond_count
        
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"❌ Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        
        # Log time even if failed
        log_time(f"{dataset_name} (FAILED)", elapsed_time, time_log_file)
        
        # Clean possibly existing variables
        if 'adata' in locals():
            del adata
        if 'adata_single' in locals():
            del adata_single
        if 'model' in locals():
            del model
        
        # Clean memory
        force_cleanup()
        
        return False, elapsed_time, 0, 0
    finally:
        # Ensure memory cleanup
        force_cleanup()


def main():
    """Main function - process datasets sequentially according to DATASET_MAPPING order"""
    print("🌟 Starting sequential batch processing of datasets...")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🎯 Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Time log file
    time_log_file = os.path.join(OUTPUT_DIR, "biolord_time.txt")
    
    # Clear time log file
    if os.path.exists(time_log_file):
        os.remove(time_log_file)
    
    # Get dataset names according to DATASET_MAPPING order
    ordered_dataset_names = []
    for h5ad_file in DATASET_MAPPING.keys():
        dataset_name = h5ad_file.replace('.h5ad', '')
        # Check if file exists
        data_path = os.path.join(DATA_DIR, h5ad_file)
        if os.path.exists(data_path):
            ordered_dataset_names.append(dataset_name)
        else:
            print(f"⚠️ Data file does not exist, skipping: {data_path}")
    
    print(f"📋 Number of datasets to process sequentially: {len(ordered_dataset_names)}")
    print("📋 Processing order:")
    for i, name in enumerate(ordered_dataset_names, 1):
        print(f"  {i:2d}. {name}")
    
    # Initial GPU memory state
    print(f"\n🔥 Initial GPU status:")
    print_gpu_memory()
    
    # Statistics
    total_start_time = time.time()
    successful_datasets = []
    failed_datasets = []
    total_predictions = 0
    total_conditions = 0
    
    # Process datasets sequentially
    for i, dataset_name in enumerate(ordered_dataset_names, 1):
        print(f"\n{'🔄' * 30}")
        print(f"📊 Overall progress: {i}/{len(ordered_dataset_names)}")
        print(f"⏰ Runtime so far: {format_time(time.time() - total_start_time)}")
        print(f"✅ Completed: {len(successful_datasets)} datasets")
        print(f"❌ Failed: {len(failed_datasets)} datasets")
        print(f"⏳ Remaining: {len(ordered_dataset_names) - i} datasets")
        print(f"{'🔄' * 30}")
        
        # Show memory state before processing
        print(f"\n🔥 GPU status before processing {dataset_name}:")
        print_gpu_memory()
        
        # Process single dataset
        success, elapsed_time, pred_count, cond_count = process_single_dataset(
            dataset_name, OUTPUT_DIR, time_log_file
        )
        
        # Show memory state after processing
        print(f"\n🔥 GPU status after processing {dataset_name}:")
        print_gpu_memory()
        
        if success:
            successful_datasets.append((dataset_name, elapsed_time))
            total_predictions += pred_count
            total_conditions += cond_count
            print(f"✅ {dataset_name} completed successfully - Time: {format_time(elapsed_time)}")
        else:
            failed_datasets.append((dataset_name, elapsed_time))
            print(f"❌ {dataset_name} processing failed - Time: {format_time(elapsed_time)}")
        
        # Force cleanup memory between datasets
        print(f"\n🧹 Inter-dataset forced memory cleanup...")
        force_cleanup()
        
        # Show current statistics
        current_time = time.time()
        current_elapsed = current_time - total_start_time
        print(f"\n📈 Current statistics (Runtime: {format_time(current_elapsed)}):")
        print(f"   ✅ Successful: {len(successful_datasets)} datasets")
        print(f"   ❌ Failed: {len(failed_datasets)} datasets")
        print(f"   📊 Predictions: {total_predictions}/{total_conditions} conditions")
        
        if i < len(ordered_dataset_names):
            print(f"   ⏳ Remaining: {len(ordered_dataset_names) - i} datasets")
            estimated_remaining = (current_elapsed / i) * (len(ordered_dataset_names) - i)
            print(f"   🕐 Estimated remaining time: {format_time(estimated_remaining)}")
        
        print(f"\n{'='*80}")
    
    # Final statistics summary
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    
    print(f"\n{'🎊' * 60}")
    print("🏆 Final Statistics Results")
    print(f"{'🎊' * 60}")
    print(f"⏰ Total runtime: {format_time(total_elapsed_time)}")
    print(f"✅ Successfully processed: {len(successful_datasets)} datasets")
    print(f"❌ Processing failed: {len(failed_datasets)} datasets")
    print(f"📈 Total prediction conditions: {total_predictions}/{total_conditions}")
    
    if successful_datasets:
        print(f"\n✅ Successfully processed datasets:")
        for name, duration in successful_datasets:
            print(f"   • {name} - {format_time(duration)}")
    
    if failed_datasets:
        print(f"\n❌ Failed datasets:")
        for name, duration in failed_datasets:
            print(f"   • {name} - {format_time(duration)}")
    
    # Save final statistics
    final_summary = {
        'total_time': format_time(total_elapsed_time),
        'successful_datasets': len(successful_datasets),
        'failed_datasets': len(failed_datasets),
        'total_predictions': total_predictions,
        'total_conditions': total_conditions,
        'success_rate': len(successful_datasets) / len(ordered_dataset_names) * 100 if ordered_dataset_names else 0
    }
    
    summary_file = Path(OUTPUT_DIR) / "final_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== Biolord Batch Processing Final Statistics ===\n")
        f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}\n")
        f.write(f"Total runtime: {final_summary['total_time']}\n")
        f.write(f"Successfully processed: {final_summary['successful_datasets']} datasets\n")
        f.write(f"Processing failed: {final_summary['failed_datasets']} datasets\n")
        f.write(f"Success rate: {final_summary['success_rate']:.1f}%\n")
        f.write(f"Total prediction conditions: {final_summary['total_predictions']}/{final_summary['total_conditions']}\n\n")
        
        f.write("Processing order:\n")
        for i, name in enumerate(ordered_dataset_names, 1):
            status = "✅" if any(name == s[0] for s in successful_datasets) else "❌"
            f.write(f"  {i:2d}. {status} {name}\n")
        
        if successful_datasets:
            f.write("\nSuccessfully processed datasets:\n")
            for name, duration in successful_datasets:
                f.write(f"  • {name} - {format_time(duration)}\n")
        
        if failed_datasets:
            f.write("\nFailed datasets:\n")
            for name, duration in failed_datasets:
                f.write(f"  • {name} - {format_time(duration)}\n")
    
    print(f"\n📄 Final statistics saved to: {summary_file}")
    print(f"📁 All results saved in: {OUTPUT_DIR}")
    print(f"⏰ Time records saved in: {time_log_file}")
    
    # Final memory state
    print(f"\n🔥 Final GPU status:")
    print_gpu_memory()


if __name__ == "__main__":
    try:
        # Initial cleanup
        print("🚀 Program starting, performing initial cleanup...")
        force_cleanup()
        
        main()
        
        print("\n🎉 All datasets processing completed!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ User interrupted program execution")
        force_cleanup()
    except Exception as e:
        print(f"\n\n❌ Serious error occurred during program execution: {e}")
        import traceback
        traceback.print_exc()
        force_cleanup()
        raise
    finally:
        # Final cleanup
        print("\n🧹 Program ending, performing final cleanup...")
        force_cleanup()