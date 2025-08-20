import sys
sys.path.insert(1, '/data2/lanxiang/conda/scVIDR/vidr/')
from vidr import VIDR
from utils import *

from sklearn.linear_model import LinearRegression
import importlib
import vidr
importlib.reload(vidr)
from vidr import VIDR  
import scanpy as sc
import pandas as pd
import numpy as np
import torch
import seaborn as sns
from scipy import stats
from scipy import linalg
from scipy import spatial
from anndata import AnnData
from scipy import sparse
from statannotations.Annotator import Annotator
from matplotlib import pyplot as plt
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import time
import scvi
sc.set_figure_params(dpi = 150)
sc.settings.figdir = "../figures/"
sns.set_style("dark")

single_condition_datasets = {
    "Kang.h5ad": {"ctrl": "ctrl", "treat": "stimulated"},
    "Haber.h5ad": {"ctrl": "ctrl", "treat": "Hpoly.Day10"},
    "Hagai.h5ad": {"ctrl": "ctrl", "treat": "LPS6"},
    "Weinreb_time.h5ad": {"ctrl": "ctrl", "treat": "developed"}
}

multi_condition_datasets = [
    "Srivatsan_sciplex3_sub10.h5ad",
    "Burkhardt_sub10.h5ad",
    "Tahoe100_sub10.h5ad",
    "Perturb_KHP_sub10.h5ad",
    "Perturb_cmo_V1_sub10.h5ad",
    "Parse_10M_PBMC_sub10.h5ad"
]

# Main output directory
main_output_dir = "/data2/lanxiang/perturb_benchmark_v2/model/scVIDR"
os.makedirs(main_output_dir, exist_ok=True)

# Timing file
timing_file = os.path.join(main_output_dir, "scvidr_runtime_summary.txt")

def process_dataset(adata_path, cell_type, ctrl_key, treat_key=None):
    start_time = time.time()
    
    try:
        # Load data
        adata = sc.read(adata_path)
        
        # Prepare data
        train_adata, test_adata = prepare_data(adata, "cell_type", "condition", cell_type, treat_key, normalized=True)
        
        # Train model
        model = VIDR(train_adata, linear_decoder=False)
        train_adata.obs["cell_dose"] = [f"{j}_{str(i)}" for (i,j) in zip(train_adata.obs["condition"], train_adata.obs["cell_type"])]
        
        model.train(max_epochs=100,
                   batch_size=128,
                   early_stopping=True,
                   early_stopping_patience=25)
        
        # Create dataset-specific folder
        model_name = os.path.splitext(os.path.basename(adata_path))[0]
        dataset_dir = os.path.join(main_output_dir, model_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create cell_type specific subfolder for model files
        model_subdir = os.path.join(dataset_dir, f"{cell_type}_models")
        os.makedirs(model_subdir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_subdir, f"{cell_type}.pt")
        model.save(model_path)
        
        # Load model
        vae = model.load(model_path, train_adata)
        
        # Predict with regression=True (scVIDR)
        pred, delta, reg = vae.predict(
            ctrl_key=ctrl_key,
            treat_key=treat_key,
            cell_type_to_predict=cell_type,
            regression=True)
        
        pred.obs["condition"] = 'pred'
        ctrl_adata = adata[((adata.obs['cell_type'] == cell_type) & (adata.obs["condition"] == ctrl_key))]
        treat_adata = adata[((adata.obs['cell_type'] == cell_type) & (adata.obs["condition"] == treat_key))]
        eval_adata = ctrl_adata.concatenate(treat_adata, pred)
        
        # Create evaluation subfolder
        eval_subdir = os.path.join(dataset_dir, "eval_results")
        os.makedirs(eval_subdir, exist_ok=True)
        
        # Save the evaluation data
        output_path = os.path.join(eval_subdir, f"{model_name}_{treat_key}_{cell_type}.h5ad")
        eval_adata.write(output_path)
        
        # Return processing time in seconds
        return True, time.time() - start_time
        
    except Exception as e:
        print(f"Error processing {adata_path} for cell type {cell_type}: {str(e)}")
        return False, 0

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h{minutes}m{seconds}s"
    elif minutes > 0:
        return f"{minutes}m{seconds}s"
    else:
        return f"{seconds}s"

# Process single condition datasets
with open(timing_file, "w") as f:
    # Process single condition datasets
    for filename, conditions in single_condition_datasets.items():
        adata_path = os.path.join("/data2/lanxiang/data/Task3_data", filename)
        if not os.path.exists(adata_path):
            print(f"File not found: {adata_path}")
            continue
            
        dataset_start = time.time()
        dataset_name = os.path.splitext(filename)[0]
        print(f"Processing dataset: {dataset_name}")
        
        # Get all cell types in the dataset
        adata = sc.read(adata_path)
        cell_types = adata.obs["cell_type"].unique()
        total_success = 0
        total_time = 0
        
        for cell_type in cell_types:
            print(f"  Processing cell type: {cell_type}")
            success, elapsed_time = process_dataset(
                adata_path, 
                cell_type, 
                conditions["ctrl"], 
                conditions["treat"])
            
            if success:
                total_success += 1
                total_time += elapsed_time
        
        # Write summary for this dataset
        if total_success > 0:
            time_str = format_time(total_time)
            f.write(f"{dataset_name}\t{total_success} cell types\t{time_str}\n")
            f.flush()
            print(f"Completed {dataset_name}: {total_success} cell types in {time_str}")

    # Process multi-condition datasets
    for filename in multi_condition_datasets:
        adata_path = os.path.join("/data2/lanxiang/data/Task3_data", filename)
        if not os.path.exists(adata_path):
            print(f"File not found: {adata_path}")
            continue
            
        dataset_start = time.time()
        dataset_name = os.path.splitext(filename)[0]
        print(f"Processing dataset: {dataset_name}")
        
        # Get all cell types and conditions (excluding ctrl)
        adata = sc.read(adata_path)
        cell_types = adata.obs["cell_type"].unique()
        conditions = [c for c in adata.obs["condition"].unique() if c != "ctrl"]
        total_success = 0
        total_time = 0
        
        for cell_type in cell_types:
            for condition in conditions:
                print(f"  Processing cell type: {cell_type}, condition: {condition}")
                success, elapsed_time = process_dataset(
                    adata_path, 
                    cell_type, 
                    "ctrl", 
                    condition)
                
                if success:
                    total_success += 1
                    total_time += elapsed_time
        
        # Write summary for this dataset
        if total_success > 0:
            time_str = format_time(total_time)
            f.write(f"{dataset_name}\t{total_success} combinations\t{time_str}\n")
            f.flush()
            print(f"Completed {dataset_name}: {total_success} combinations in {time_str}")