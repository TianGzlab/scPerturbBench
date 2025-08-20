import scanpy as sc
import warnings
import os
import time
from datetime import datetime
import pandas as pd
warnings.filterwarnings('ignore')

from scpram import models
from scpram import evaluate
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
def create_output_dir(base_path, dataset_name, condition, cell_type):
    """Create output directory structure if it doesn't exist"""
    output_dir = os.path.join(base_path, dataset_name, condition)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{cell_type}.h5ad")

def process_dataset(dataset_path, dataset_name, output_base_path, runtime_log):
    """Process a single dataset for all condition-cell_type combinations"""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    dataset_start_time = time.time()
    
    # Load dataset
    adata = sc.read_h5ad(dataset_path)
    adata = sc.AnnData(adata.X, obs=adata.obs.copy(), var=adata.var.copy())
    adata.obs_names_make_unique()
    
    print(f"Dataset shape: {adata.shape}")
    print(f"Conditions: {sorted(adata.obs['condition'].unique())}")
    print(f"Cell types: {sorted(adata.obs['cell_type'].unique())}")
    
    # Get unique conditions and cell types
    conditions = adata.obs['condition'].unique()
    cell_types = adata.obs['cell_type'].unique()
    
    # Identify control condition (usually 'ctrl' or 'control')
    ctrl_conditions = [c for c in conditions if c.lower() in ['ctrl']]
    if not ctrl_conditions:
        print(f"Warning: No control condition found for {dataset_name}")
        return
    ctrl_key = ctrl_conditions[0]
    
    # Filter out control condition from conditions to process
    conditions_to_process = [c for c in conditions if c != ctrl_key]
    
    print(f"Control condition: {ctrl_key}")
    print(f"Conditions to process: {len(conditions_to_process)}")
    print(f"Cell types to process: {len(cell_types)}")
    
    key_dic = {
        'condition_key': 'condition',
        'cell_type_key': 'cell_type',
        'ctrl_key': ctrl_key,
        'stim_key': 'stimulated',  # This will be updated for each condition
        'pred_key': 'predict',
    }
    
    total_combinations = len(conditions_to_process) * len(cell_types)
    current_combination = 0
    
    for condition in conditions_to_process:
        print(f"\nProcessing condition: {condition}")
        
        # Update the stimulated key for current condition
        key_dic['stim_key'] = condition
        
        for cell_type in cell_types:
            current_combination += 1
            print(f"  Processing cell type: {cell_type} ({current_combination}/{total_combinations})")
            
            # Check if this combination exists in the data
            ctrl_cells = adata[(adata.obs['condition'] == ctrl_key) & 
                              (adata.obs['cell_type'] == cell_type)]
            stim_cells = adata[(adata.obs['condition'] == condition) & 
                              (adata.obs['cell_type'] == cell_type)]
            
            if len(ctrl_cells) == 0:
                print(f"    Skipping: No control cells for {cell_type}")
                continue
            if len(stim_cells) == 0:
                print(f"    Skipping: No stimulated cells for {cell_type} in {condition}")
                continue
            
            try:
                # Create and train model
                model = models.SCPRAM(input_dim=adata.n_vars, device='cuda:1')
                model = model.to(model.device)
                
                # Prepare training data (exclude the target condition-cell_type combination)
                train = adata[~((adata.obs['cell_type'] == cell_type) &
                               (adata.obs['condition'] == condition))]
                
                # Train model
                model.train_SCPRAM(train, epochs=100)
                
                # Make prediction
                pred = model.predict(train_adata=train,
                                   cell_to_pred=cell_type,
                                   key_dic=key_dic,
                                   ratio=0.005)
                
                # Create output path and save
                output_path = create_output_dir(output_base_path, dataset_name, condition, cell_type)
                pred.write_h5ad(output_path)
                print(f"    Saved prediction to: {output_path}")
                
            except Exception as e:
                print(f"    Error processing {cell_type} in {condition}: {str(e)}")
                continue
    
    dataset_end_time = time.time()
    dataset_runtime = dataset_end_time - dataset_start_time
    
    # Log runtime
    runtime_entry = f"{dataset_name}: {dataset_runtime:.2f} seconds ({dataset_runtime/60:.2f} minutes)\n"
    runtime_log.append(runtime_entry)
    print(f"\nDataset {dataset_name} completed in {dataset_runtime:.2f} seconds ({dataset_runtime/60:.2f} minutes)")

def main():
    # Configuration
    data_base_path = "/data2/lanxiang/data/Task3_data"
    output_base_path = "/data2/lanxiang/perturb_benchmark_v2/model/scPRAM"
    runtime_log_path = os.path.join(output_base_path, "runtime_summary.txt")
    
    # Datasets to process in order
    datasets = [
        "Kang.h5ad",
        "Haber.h5ad", 
        "Hagai.h5ad",
        "Weinreb_time.h5ad",
        "Burkhardt_sub10.h5ad",
        "Srivatsan_sciplex3_sub10.h5ad",
        "Perturb_cmo_V1_sub10.h5ad",
        "Perturb_KHP_sub10.h5ad",
        "Tahoe100_sub10.h5ad",
        "Parse_10M_PBMC_sub10.h5ad"
    ]
    
    # Initialize runtime log
    runtime_log = []
    total_start_time = time.time()
    
    # Add header to runtime log
    runtime_log.append(f"scPRAM Batch Processing Runtime Summary\n")
    runtime_log.append(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    runtime_log.append(f"{'='*50}\n\n")
    
    # Process each dataset
    for dataset_file in datasets:
        dataset_path = os.path.join(data_base_path, dataset_file)
        dataset_name = dataset_file.replace('.h5ad', '')
        
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset {dataset_path} not found, skipping...")
            continue
        
        try:
            process_dataset(dataset_path, dataset_name, output_base_path, runtime_log)
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
    runtime_log.append(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes) ({total_runtime/3600:.2f} hours)\n")
    runtime_log.append(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Write runtime log to file
    with open(runtime_log_path, 'w') as f:
        f.writelines(runtime_log)
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETED")
    print(f"{'='*60}")
    print(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    print(f"Runtime summary saved to: {runtime_log_path}")

if __name__ == "__main__":
    main()