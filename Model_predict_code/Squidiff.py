import sys
import os
import time
from datetime import datetime
import subprocess
import shutil
import pandas as pd
import scanpy as sc
import numpy as np
import warnings

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from torch.nn import DataParallel
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import anndata
import seaborn as sns
import matplotlib.font_manager
from matplotlib import rcParams

import argparse

font_list = []
fpaths = matplotlib.font_manager.findSystemFonts()
for i in fpaths:
    try:
        f = matplotlib.font_manager.get_font(i)
        font_list.append(f.family_name)
    except RuntimeError:
        pass

font_list = set(font_list)
plot_font = 'Helvetica' if 'Helvetica' in font_list else 'FreeSans'
rcParams['font.family'] = plot_font
rcParams.update({'font.size': 10})
rcParams.update({'figure.dpi': 300})
rcParams.update({'figure.figsize': (3,3)})
rcParams.update({'savefig.dpi': 500})
warnings.filterwarnings('ignore')

# Remove notebook-specific commands for script execution
# %load_ext autoreload
# %autoreload 2
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

def load_allowed_combinations(csv_path="/home/yunlin/projects/perturb_model/lanxiang_model/Task3_Allgenes_common_data.csv"):
    """Load allowed dataset, cell_type, condition combinations from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        # Extract unique combinations of dataset, cell_type, condition
        combinations = df[['dataset', 'cell_type', 'condition']].drop_duplicates()
        
        # Convert to set of tuples for fast lookup
        allowed_combinations = set()
        for _, row in combinations.iterrows():
            allowed_combinations.add((row['dataset'], row['cell_type'], row['condition']))
        
        print(f"Loaded {len(allowed_combinations)} unique combinations from {csv_path}")
        return allowed_combinations
    except Exception as e:
        print(f"Warning: Failed to load combinations from {csv_path}: {str(e)}")
        print("Will process all combinations in dataset")
        return None

def cleanup_checkpoint_dir(checkpoint_dir):
    """Clean up checkpoint directory to save space"""
    if os.path.exists(checkpoint_dir):
        try:
            shutil.rmtree(checkpoint_dir)
            print(f"    Cleaned up checkpoint directory: {checkpoint_dir}")
        except Exception as e:
            print(f"    Warning: Failed to clean up {checkpoint_dir}: {str(e)}")



def create_output_dir(base_path, dataset_name, condition, cell_type):
    """Create output directory structure if it doesn't exist"""
    output_dir = os.path.join(base_path, dataset_name, condition)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def process_dataset(dataset_path, dataset_name, temp_data_dir='temp_data', results_dir='squidiff_results', temp_model_dir='temp_checkpoints', 
                    temp_log_dir='temp_log', allowed_combinations=None):
    """Process a single dataset for all condition-cell_type combinations"""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    dataset_start_time = time.time()
    
    # Load dataset
    adata = sc.read_h5ad(dataset_path)
    sc.pp.highly_variable_genes(adata, n_top_genes=200)
    adata = adata[:, adata.var.highly_variable].copy()
    
    all_conds = adata.obs['condition'].unique()
    assert 'ctrl' in all_conds
    
    cond2id = {'ctrl': 0}
    cur_id = 1
    for c in all_conds:
        if c != 'ctrl' and c not in cond2id:
            cond2id[c] = cur_id
            cur_id += 1
    adata.obs['Group'] = adata.obs['condition'].map(cond2id)
    
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    adata.obs["condition"] = adata.obs["condition"].astype(str)
    
    print(f"Dataset shape: {adata.shape}")
    print(f"Conditions: {sorted(adata.obs['condition'].unique())}")
    print(f"Cell types: {sorted(adata.obs['cell_type'].unique())}")
    
    # Get unique conditions and cell types
    conditions = adata.obs['condition'].unique()
    cell_types = adata.obs['cell_type'].unique()
    
    # Identify control condition
    ctrl_key = 'ctrl'
    
    # Filter out control condition from conditions to process
    conditions_to_process = [c for c in conditions if c != ctrl_key]
    
    print(f"Control condition: {ctrl_key}")
    print(f"Conditions to process: {len(conditions_to_process)}")
    print(f"Cell types to process: {len(cell_types)}")
    
    # Calculate total combinations considering allowed_combinations filter
    if allowed_combinations is not None:
        allowed_for_dataset = [combo for combo in allowed_combinations if combo[0] == dataset_name]
        total_combinations = len(allowed_for_dataset)
        print(f"Allowed combinations for {dataset_name}: {total_combinations}")
    else:
        total_combinations = len(conditions_to_process) * len(cell_types)
    
    current_combination = 0
    
    for target_condition in conditions_to_process:
        print(f"\nProcessing condition: {target_condition}")
        
        for target_celltype in cell_types:
            current_combination += 1
            print(f"  Processing cell type: {target_celltype} ({current_combination}/{total_combinations})")
            
            # Check if this combination is in the allowed list
            if allowed_combinations is not None:
                combination_key = (dataset_name, target_celltype, target_condition)
                if combination_key not in allowed_combinations:
                    print(f"    Skipping: Combination {combination_key} not in allowed list")
                    continue
            
            # Check if this combination exists in the data
            ctrl_cells = adata[(adata.obs['condition'] == ctrl_key) & 
                              (adata.obs['cell_type'] == target_celltype)]
            target_cells = adata[(adata.obs['condition'] == target_condition) & 
                               (adata.obs['cell_type'] == target_celltype)]
            
            if len(ctrl_cells) == 0:
                print(f"    Skipping: No control cells for {target_celltype}")
                continue
            if len(target_cells) == 0:
                print(f"    Skipping: No target cells for {target_celltype} in {target_condition}")
                continue
            
            try:
                # Check if output file already exists, skip if so
                pred_dir = os.path.join(results_dir, 'predictions', dataset_name, target_condition)
                pred_path = os.path.join(pred_dir, f'{target_celltype}_prediction.h5ad')
                
                if os.path.exists(pred_path):
                    print(f"    Skipping: Output file already exists at {pred_path}")
                    continue
                
                test_mask = (
                    (adata.obs["cell_type"] == target_celltype) &
                    (adata.obs["condition"] == target_condition)
                )
                
                train_mask = (
                    (adata.obs["condition"] == ctrl_key) |
                    (
                        (adata.obs["condition"] == target_condition) &
                        (adata.obs["cell_type"] != target_celltype)
                    )
                )
                
                # sanity check
                print(f"    Train cells: {train_mask.sum()}")
                print(f"    Test cells: {test_mask.sum()}")
                
                if train_mask.sum() == 0 or test_mask.sum() == 0:
                    print(f"    Skipping: Insufficient data for {target_celltype} in {target_condition}")
                    continue
                
                adata_train = adata[train_mask].copy()
                adata_test = adata[test_mask].copy()
                
                # Use configurable paths for intermediate train/test files (will be overwritten)
                train_path = os.path.join(temp_data_dir, 'diff_train.h5ad')
                test_path = os.path.join(temp_data_dir, 'diff_test.h5ad')
                
                # Ensure temp data directory exists
                os.makedirs(temp_data_dir, exist_ok=True)
                
                # Overwrite training data files each time
                adata_train.write(train_path)
                adata_test.write(test_path)
                
                print(f"    Saved train/test data to: {temp_data_dir} (overwritten)")
                
                # Train and predict for this combination
                train_and_predict(dataset_name, target_condition, target_celltype, 
                                temp_data_dir=temp_data_dir, results_dir=results_dir,
                                temp_model_dir=temp_model_dir, temp_log_dir=temp_log_dir)
                
            except Exception as e:
                print(f"    Error processing {target_celltype} in {target_condition}: {str(e)}")
                continue
    
    dataset_end_time = time.time()
    dataset_runtime = dataset_end_time - dataset_start_time
    print(f"\nDataset {dataset_name} completed in {dataset_runtime:.2f} seconds ({dataset_runtime/60:.2f} minutes)")

def train_and_predict(dataset_name, condition, cell_type, temp_data_dir='temp_data', 
                     results_dir='squidiff_results', temp_model_dir='temp_checkpoints', 
                     temp_log_dir='temp_log'):
    """Train model and make predictions for a specific combination"""
    try:
        # Define paths using configurable parameters
        train_path = os.path.join(temp_data_dir, 'diff_train.h5ad')
        test_path = os.path.join(temp_data_dir, 'diff_test.h5ad')
        
        # Create temporary directories
        os.makedirs(temp_model_dir, exist_ok=True)
        os.makedirs(temp_log_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Training command with MKL fix
        train_cmd = f"""export MKL_SERVICE_FORCE_INTEL=1 && python ../Squidiff/train_squidiff.py --logger_path {temp_log_dir} --data_path {train_path} --resume_checkpoint {temp_model_dir} --gene_size 200 --output_dim 200"""
        
        print(f"    Training model for {cell_type} in {condition}...")
        print(f"    Command: {train_cmd}")
        
        # Execute training command
        try:
            result = subprocess.run(train_cmd, shell=True, capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                print(f"    Training completed successfully")
                
                # Save training log to results directory (permanent)
                log_file = os.path.join(results_dir, f'{dataset_name}_{condition}_{cell_type}_training.log')
                with open(log_file, 'w') as f:
                    f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
                    
            else:
                print(f"    Training failed with return code {result.returncode}")
                error_log = os.path.join(results_dir, f'{dataset_name}_{condition}_{cell_type}_error.log')
                with open(error_log, 'w') as f:
                    f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
                return
                
        except Exception as train_error:
            print(f"    Error executing training command: {str(train_error)}")
            return
        
        # Load trained model and make predictions
        model_path = os.path.join(temp_model_dir, 'model.pt')
        if os.path.exists(model_path):
            print(f"    Making predictions...")
            sys.path.append('../Squidiff')
            import sample_squidiff
            
            sampler = sample_squidiff.sampler(
                model_path=model_path,
                gene_size=200,
                output_dim=200,
                use_drug_structure=False
            )
            
            test_adata_scrna = sc.read_h5ad(test_path)
            if hasattr(test_adata_scrna.X, 'toarray'):
                test_adata_scrna.X = test_adata_scrna.X.toarray()
            
            z_sem_scrna = sampler.model.encoder(torch.tensor(test_adata_scrna.X).to('cuda'))
            scrnas_pred = sampler.pred(z_sem_scrna, gene_size=test_adata_scrna.shape[1])
            
            # Save predictions to results directory (permanent storage)
            pred_dir = os.path.join(results_dir, 'predictions', dataset_name, condition)
            os.makedirs(pred_dir, exist_ok=True)
            pred_path = os.path.join(pred_dir, f'{cell_type}_prediction.h5ad')
            
            # Convert predictions to AnnData and save
            pred_adata = sc.AnnData(X=scrnas_pred.cpu().numpy(), 
                                  obs=test_adata_scrna.obs.copy(),
                                  var=test_adata_scrna.var.copy())
            pred_adata.write(pred_path)
            print(f"    Saved predictions to: {pred_path}")
            
            # Clean up temporary checkpoint directory to save space
            cleanup_checkpoint_dir(temp_model_dir)
            cleanup_checkpoint_dir(temp_log_dir)
            
        else:
            print(f"    Warning: Model file not found at {model_path}")
    
    except Exception as e:
        print(f"    Error in training/prediction: {str(e)}")

def main(dataset_file=None, temp_data_dir='temp_data', results_dir='squidiff_results', temp_model_dir='temp_checkpoints', 
                     temp_log_dir='temp_log'):
    """Main function to process multiple datasets"""
    # Load allowed combinations from CSV file
    allowed_combinations = load_allowed_combinations()
    
    # Configuration
    data_base_path = "/data2/lanxiang/data/Task3_data"
    runtime_log_path = os.path.join(results_dir, "squidiff_runtime_summary.txt")
    
    # Datasets to process
    if dataset_file is not None:
        datasets = [dataset_file]
    else:
        datasets = [
            # "Kang.h5ad",
            "Haber.h5ad", 
            "Hagai.h5ad",
            "Weinreb_time.h5ad",
            "Burkhardt_sub10.h5ad",
            "Srivatsan_sciplex3_sub10.h5ad",
            "Perturb_cmo_V1_sub10.h5ad",
            "Perturb_KHP_sub10.h5ad"
        ]
    
    # Initialize runtime log
    runtime_log = []
    total_start_time = time.time()
    
    # Add header to runtime log
    runtime_log.append(f"Squidiff Batch Processing Runtime Summary\n")
    runtime_log.append(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    runtime_log.append(f"{'='*50}\n\n")
    
    # Process each dataset
    processed_count = 0
    for dataset_file in datasets:
        dataset_path = os.path.join(data_base_path, dataset_file)
        dataset_name = dataset_file.replace('.h5ad', '')
        
        if not os.path.exists(dataset_path):
            warning_msg = f"Warning: Dataset {dataset_path} not found, skipping...\n"
            runtime_log.append(warning_msg)
            print(warning_msg.strip())
            continue
        
        try:
            dataset_start_time = time.time()
            process_dataset(dataset_path, dataset_name, temp_data_dir=temp_data_dir, results_dir=results_dir, temp_model_dir=temp_model_dir, 
                     temp_log_dir=temp_log_dir, allowed_combinations=allowed_combinations)
            dataset_end_time = time.time()
            dataset_runtime = dataset_end_time - dataset_start_time
            
            # Log runtime for this dataset
            runtime_entry = f"{dataset_name}: {dataset_runtime:.2f} seconds ({dataset_runtime/60:.2f} minutes)\n"
            runtime_log.append(runtime_entry)
            processed_count += 1
            
        except Exception as e:
            error_msg = f"Error processing dataset {dataset_name}: {str(e)}\n"
            runtime_log.append(error_msg)
            print(f"Error processing dataset {dataset_name}: {str(e)}")
            continue
    
    # Calculate total runtime
    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    
    # Finalize runtime log
    runtime_log.append(f"\n{'='*50}\n")
    runtime_log.append(f"Processed {processed_count}/{len(datasets)} datasets successfully\n")
    runtime_log.append(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes) ({total_runtime/3600:.2f} hours)\n")
    runtime_log.append(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Write runtime log to file
    with open(runtime_log_path, 'w') as f:
        f.writelines(runtime_log)
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETED")
    print(f"{'='*60}")
    print(f"Processed {processed_count}/{len(datasets)} datasets successfully")
    print(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    print(f"Runtime summary saved to: {runtime_log_path}")

if __name__ == "__main__":
    # # Configuration parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset file name, e.g. Kang.h5ad")
    args = parser.parse_args()

    DATASET_TAG = args.dataset.replace(".h5ad", "") if args.dataset else "batch"

    TEMP_DATA_DIR = f'temp_data_{DATASET_TAG}'
    TEMP_MODEL_DIR = f'temp_checkpoints_{DATASET_TAG}'
    TEMP_LOG_DIR = f'temp_log_{DATASET_TAG}'
    RESULTS_DIR = 'squidiff_results'
    # # For single dataset testing, uncomment the lines below:
    # adata_path = '/data2/lanxiang/data/Task3_data/Kang.h5ad'
    # dataset_name = 'Kang'
    # process_dataset(adata_path, dataset_name, temp_data_dir=TEMP_DATA_DIR, results_dir=RESULTS_DIR)
    
    # For multiple datasets processing:
    main(dataset_file=args.dataset, temp_data_dir=TEMP_DATA_DIR, results_dir=RESULTS_DIR, temp_model_dir=TEMP_MODEL_DIR, temp_log_dir=TEMP_LOG_DIR)





