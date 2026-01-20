import sys
import os
import pickle
import json
import re
import time
import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import traceback

import requests
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad  
import scvi
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from rdkit import Chem

from perturbnet.util import * 
from perturbnet.cinn.flow import * 
from perturbnet.chemicalvae.chemicalVAE import *
from perturbnet.data_vae.vae import *
from perturbnet.cinn.flow_generate import TFVAEZ_CheckNet2Net, SCVIZ_CheckNet2Net

# Configuration
proxies = {
    "http": "http://127.0.0.1:7895",
    "https": "http://127.0.0.1:7895"
}

# Paths
DATA_DIR = "/data2/lanxiang/data/Task3_data"
OUTPUT_ROOT = Path("/data2/lanxiang/perturb_benchmark_v2/model/PerturbNet/Task3")
CHEM_VAE_DIR = Path("/data2/lanxiang/perturb_benchmarking/test_model_data/perturbNet/pretrained_model/chemicalVAE")

# Dataset order
DATASETS = [
    "Perturb_cmo_V1_sub10.h5ad",
    "Srivatsan_sciplex3_sub10.h5ad", 
    "Burkhardt_sub10.h5ad",
    "Perturb_KHP_sub10.h5ad",
    "Tahoe100_sub10.h5ad"
]

# Device will be set later when loading ChemicalVAE model

# Create output directory
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Initialize timing log
output_time_log = OUTPUT_ROOT / "perturbnet_runtime_summary.txt"
with open(output_time_log, "w") as f:
    f.write("Dataset\tCondition\tCellType\tTime\tStatus\n")

def canonicalize(smiles):
    """Canonicalize SMILES string using RDKit."""
    if not isinstance(smiles, str) or smiles == "":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def fetch_smiles_from_pubchem(drug_name: str, proxies: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Fetch SMILES from PubChem for a given drug name."""
    candidates = [re.sub(r"\s*\(.*?\)", "", drug_name).strip()]
    aliases = re.findall(r"\((.*?)\)", drug_name)
    candidates.extend(a.strip() for a in aliases)
    
    headers = {"User-Agent": "Mozilla/5.0 (PerturbNet pipeline)"}
    
    for candidate in candidates:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{candidate}/property/ConnectivitySMILES/JSON"
        )
        
        for attempt in range(3):
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    timeout=10,
                    proxies=proxies,
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    wait = random.uniform(1, 3)
                    time.sleep(wait)
                    return payload["PropertyTable"]["Properties"][0]["ConnectivitySMILES"]
                elif resp.status_code == 503:
                    wait = random.uniform(1, 3)
                    print(
                        f"[INFO] 503 for {candidate}; retrying in {wait:.1f}s "
                        f"({attempt + 1}/3)"
                    )
                    time.sleep(wait)
                    continue
            except Exception as exc:
                print(f"[WARN] PubChem lookup failed for {candidate}: {exc}")
        print(f"[WARN] No SMILES found for {candidate}")
    return None

def ensure_smiles_column(adata, condition_col: str, cache_path: Optional[Path] = None, 
                        attempt_fetch: bool = True, proxies: Optional[Dict[str, str]] = None):
    """Add SMILES column to adata.obs using cached data and PubChem queries."""
    smiles_cache: Dict[str, str] = {}
    
    # Load existing cache
    if cache_path and cache_path.exists():
        with cache_path.open("r") as handle:
            smiles_cache = json.load(handle)
    
    # Add DMSO SMILES for control condition
    smiles_cache.setdefault('ctrl', 'CS(=O)C')  
    
    # Check if SMILES column already exists and extract existing mappings
    if "smiles" in adata.obs.columns:
        for cond, smiles in zip(adata.obs[condition_col], adata.obs["smiles"].fillna("")):
            if smiles and str(smiles) != "" and str(smiles) != "nan":
                smiles_cache.setdefault(cond, str(smiles))
        # Remove the old column to avoid Categorical issues
        adata.obs.drop(columns=["smiles"], inplace=True)
    
    # Find missing conditions
    missing_conditions = [
        cond for cond in adata.obs[condition_col].unique() 
        if cond not in smiles_cache
    ]
    
    if missing_conditions:
        print(f"[INFO] SMILES lookup: {len(missing_conditions)} conditions missing from cache.")
    else:
        print("[INFO] SMILES lookup: all conditions already cached.")
    
    # Fetch missing SMILES
    if missing_conditions and attempt_fetch:
        for cond in missing_conditions:
            print(f"Fetching SMILES for: {cond}")
            smiles = fetch_smiles_from_pubchem(cond, proxies)
            if smiles:
                smiles_cache[cond] = smiles
                print(f"  -> Found: {smiles[:50]}...")
            else:
                print(f"  -> Not found")
    
    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as handle:
            json.dump(smiles_cache, handle, indent=2)
    
    # Add SMILES to adata - convert to string to avoid Categorical issues
    adata.obs["smiles"] = adata.obs[condition_col].astype(str).map(smiles_cache).fillna("")
    
    # Stats
    unique_conditions = adata.obs[condition_col].nunique()
    unique_with_smiles = adata.obs.loc[adata.obs["smiles"] != "", condition_col].nunique()
    unique_missing = unique_conditions - unique_with_smiles
    
    print(
        "[INFO] SMILES lookup complete: "
        f"{unique_with_smiles} conditions mapped, {unique_missing} missing."
    )

def filter_by_smiles_and_controls(adata, condition_col: str, control_conditions: Sequence[str]):
    """Keep cells with valid SMILES or belonging to control conditions."""
    
    # Canonicalize SMILES strings
    unique_smiles = adata.obs["smiles"].dropna().unique()
    canonicalized_map = {}
    
    for smiles in unique_smiles:
        if smiles:  # Skip empty strings
            canonical = canonicalize(smiles)
            if canonical is not None:
                canonicalized_map[smiles] = canonical
    
    # Map canonicalized SMILES back
    adata.obs["smiles"] = adata.obs["smiles"].map(
        lambda x: canonicalized_map.get(x, x) if pd.notna(x) else x
    )
    
    # Validate SMILES strings
    char_list = [
        "7", "6", "o", "]", "3", "s", "(", "-", "S", "/", "B", "4", "[", ")", 
        "#", "I", "l", "O", "H", "c", "1", "@", "=", "n", "P", "8", "C", "2", 
        "F", "5", "r", "N", "+", "\\", " ",
    ]
    valid_chars = set(char_list)
    
    def is_valid_smiles(smiles: str) -> bool:
        if not isinstance(smiles, str) or smiles == "":
            return False
        if len(smiles) > 120:
            return False
        return all(ch in valid_chars for ch in smiles)
    
    # Filter criteria
    has_valid_smiles = adata.obs["smiles"].apply(is_valid_smiles)
    is_control = adata.obs[condition_col].isin(control_conditions)
    
    if control_conditions:
        keep = has_valid_smiles | is_control
    else:
        keep = has_valid_smiles
    
    print(f"Before filtering: {adata.n_obs} cells")
    print(f"Valid SMILES: {has_valid_smiles.sum()} cells")
    print(f"Control conditions: {is_control.sum()} cells")
    print(f"After filtering: {keep.sum()} cells")
    
    return adata[keep].copy()

def process_dataset_condition_celltype(dataset_name, condition, cell_type, adata_filtered, 
                                       dataset_output_dir):
    """Process single condition-celltype combination."""
    
    print(f"  Processing: {condition} - {cell_type}")
    start_time = time.time()
    
    try:
        # Create output directories
        condition_dir = dataset_output_dir / condition
        celltype_dir = condition_dir / cell_type
        celltype_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up test mask
        test_mask = ((adata_filtered.obs["cell_type"] == cell_type) & 
                     (adata_filtered.obs["condition"] == condition))
        
        # Check if we have test samples
        if not test_mask.any():
            print(f"    No samples found for {condition} - {cell_type}, skipping")
            return False
        
        # Create split
        adata_filtered.obs["split"] = "train"
        adata_filtered.obs.loc[test_mask, "split"] = "test"
        
        print(f"    Train: {(adata_filtered.obs['split'] == 'train').sum()}, Test: {test_mask.sum()}")
        
        # Get train and test data
        adata_train = adata_filtered[adata_filtered.obs.split == "train", :].copy()
        adata_test = adata_filtered[adata_filtered.obs.split == "test", :].copy()
        
        if adata_train.n_obs == 0:
            print(f"    No training samples available, skipping")
            return False
        
        # Train SCVI model (without device specification like in Jupyter)
        scvi_model_save_path = str(celltype_dir / "cellvae")
        scvi.data.setup_anndata(adata_train, layer="counts")
        scvi_model = scvi.model.SCVI(adata_train, n_latent=10)
        scvi_model.train(n_epochs=200, frequency=20)
        scvi_model.save(scvi_model_save_path)
        
        # Load SCVI model with CPU (matching Jupyter code)
        scvi.data.setup_anndata(adata_train, layer="counts")
        scvi_model = scvi.model.SCVI.load(scvi_model_save_path, adata_train, use_cuda=False)
        
        # Load ChemicalVAE model with device detection (matching Jupyter code)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_chemvae = ChemicalVAE(n_char=zinc_onehot.shape[2], max_len=zinc_onehot.shape[1]).to(device)
        model_chemvae.load_state_dict(torch.load(path_chemvae_model, map_location=device))
        model_chemvae.eval()
        
        std_model = Standardize(data_all=zinc_onehot, model=model_chemvae, device=device)
        
        # Prepare embeddings
        cond_stage_data, embeddings, perturbToEmbed = prepare_embeddings_cinn(
            adata_train,
            perturbation_key="smiles",
            trt_key="ordered_all_smile_example", 
            embed_key="ordered_all_onehot_example"
        )
        
        # Get all cell types dynamically
        all_cell_types = sorted(adata_train.obs["cell_type"].unique())
        cell_type_df = pd.get_dummies(adata_train.obs["cell_type"])[all_cell_types]
        covariates_embedding = cell_type_df.to_numpy()
        
        # Calculate conditioning_dim dynamically
        conditioning_dim = 196 + covariates_embedding.shape[1]
        
        # Train CINN model
        path_cinn_model_save = celltype_dir / "cinn"
        path_cinn_model_save.mkdir(parents=True, exist_ok=True)
        
        torch.manual_seed(42)
        flow_model = ConditionalFlatCouplingFlow(
            conditioning_dim=conditioning_dim,
            embedding_dim=10, 
            conditioning_depth=2, 
            n_flows=20, 
            in_channels=10, 
            hidden_dim=1024, 
            hidden_depth=2, 
            activation="none", 
            conditioner_use_bn=True
        )
        
        model_c = Net2NetFlow_TFVAE_Covariate_Flow(
            configured_flow=flow_model,
            first_stage_data=adata_train.X.A,
            cond_stage_data=cond_stage_data,
            perturbToOnehotLib=perturbToEmbed,
            oneHotData=embeddings, 
            model_con=model_chemvae, 
            std_model=std_model, 
            covariates=covariates_embedding,
            scvi_model=scvi_model
        )
        model_c.to(device=device)
        model_c.train_cinn(n_epochs=25, batch_size=128, lr=4.5e-6, 
                          auto_save=10, auto_save_path=str(path_cinn_model_save / "Auto/"))
        model_c.save(str(path_cinn_model_save / "25ep/"))
        
        # Reload model for prediction (matching Jupyter code structure)
        torch.manual_seed(42)
        flow_model = ConditionalFlatCouplingFlow(
            conditioning_dim=conditioning_dim,
            embedding_dim=10, 
            conditioning_depth=2, 
            n_flows=20, 
            in_channels=10, 
            hidden_dim=1024, 
            hidden_depth=2, 
            activation="none", 
            conditioner_use_bn=True
        )
        
        model_c = Net2NetFlow_TFVAE_Covariate_Flow(
            configured_flow=flow_model,
            first_stage_data=adata_train.X.A,
            cond_stage_data=cond_stage_data,
            perturbToOnehotLib=perturbToEmbed,
            oneHotData=embeddings, 
            model_con=model_chemvae, 
            std_model=std_model, 
            covariates=covariates_embedding,
            scvi_model=scvi_model
        )
        model_c.load(str(path_cinn_model_save / "25ep/"))
        model_c.to(device=device)
        model_c.eval()
        
        # Prepare prediction
        scvi.data.setup_anndata(adata_test, layer="counts")
        Zsample_test = scvi_model.get_latent_representation(adata=adata_test, give_mean=False)
        
        scvi_model_de = scvi_predictive_z(scvi_model)
        perturbnet_model = SCVIZ_CheckNet2Net(model_c, device, scvi_model_de)
        Lsample_obs = scvi_model.get_latent_library_size(adata=adata_train, give_mean=False)
        
        # Get test condition SMILES
        test_smiles = adata_filtered.obs.loc[test_mask, "smiles"].unique()[0]
        pert_idx = np.where(adata_filtered.uns["ordered_all_smile_example"] == test_smiles)[0][0]
        test_pert_embed = adata_filtered.uns["ordered_all_onehot_example"][pert_idx]
        
        # Prepare cell type embedding for test
        cell_type_test = pd.get_dummies(adata_test.obs["cell_type"]).reindex(
            columns=all_cell_types, fill_value=0).to_numpy()
        covariates_embedding_test = cell_type_test
        
        covariates_embedding_target = covariates_embedding_test[
            np.where(adata_test.obs.smiles == test_smiles)[0]]
        
        n_cells = covariates_embedding_target.shape[0]
        
        # Generate predictions
        trt_input_onehot = np.tile(test_pert_embed, (n_cells, 1, 1))
        _, _, _, embdata_torch = model_c.model_con(torch.tensor(trt_input_onehot).float().to(device))
        embdata_np = std_model.standardize_z(embdata_torch.cpu().detach().numpy())
        pert_embed = np.concatenate([embdata_np, covariates_embedding_target], axis=1)
        
        Lsample_idx = np.random.choice(range(Lsample_obs.shape[0]), n_cells, replace=True)
        library_trt_latent = Lsample_obs[Lsample_idx]
        predict_latent, predict_data = perturbnet_model.sample_data(pert_embed, library_trt_latent)
        
        # Get real data
        real_latent = Zsample_test[np.where(adata_test.obs.smiles == test_smiles)[0]]
        real_data = adata_test.layers["counts"].A[np.where(adata_test.obs.smiles == test_smiles)[0]]
        
        # Save results
        safe_name = f"{condition}_{cell_type}"
        predict_save_path = celltype_dir / f"{safe_name}_predict.npz"
        real_save_path = celltype_dir / f"{safe_name}_real.npz"
        
        np.savez_compressed(predict_save_path, predict_data=predict_data)
        np.savez_compressed(real_save_path, real_data=real_data)
        
        # Calculate and log timing
        end_time = time.time()
        duration_sec = end_time - start_time
        duration_str = f"{int(duration_sec // 3600)}h{int((duration_sec % 3600) // 60)}min"
        
        with open(output_time_log, "a") as f:
            f.write(f"{dataset_name}\t{condition}\t{cell_type}\t{duration_str}\tSuccess\n")
        
        print(f"    Completed in {duration_str}")
        return True
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"    {error_msg}")
        traceback.print_exc()
        
        # Log error
        with open(output_time_log, "a") as f:
            f.write(f"{dataset_name}\t{condition}\t{cell_type}\t0h0min\tFailed: {error_msg}\n")
        
        return False

# Load chemical VAE model once (will be used in each dataset processing)
print("Loading Chemical VAE model...")
zinc_onehot = np.load(CHEM_VAE_DIR / "onehot_zinc.npy")
path_chemvae_model = CHEM_VAE_DIR / "model_params_525.pt"

# Main loop
for dataset_file in DATASETS:
    dataset_name = dataset_file.replace(".h5ad", "")
    dataset_path = Path(DATA_DIR) / dataset_file
    
    if not dataset_path.exists():
        print(f"Dataset {dataset_file} not found, skipping...")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Load data
        adata = sc.read_h5ad(dataset_path)
        print(f"Dataset shape: {adata.shape}")
        
        # Create dataset output directory
        dataset_output_dir = OUTPUT_ROOT / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up SMILES cache path
        smiles_cache_path = dataset_output_dir / "smiles_cache.json"
        
        # Add SMILES information
        ensure_smiles_column(
            adata,
            condition_col="condition",
            cache_path=smiles_cache_path,
            attempt_fetch=True,
            proxies=proxies,
        )
        
        # Filter data
        control_conditions = ["ctrl"]
        adata_filtered = filter_by_smiles_and_controls(adata, "condition", control_conditions)
        print(f"Filtered data shape: {adata_filtered.shape}")
        
        # Prepare SMILES embeddings
        smiles_list = adata_filtered.obs["smiles"].unique()
        onehot = smiles_to_hot(smiles=smiles_list, max_len=120, padding="right", nchars=35)
        
        adata_filtered.uns["ordered_all_smile_example"] = smiles_list
        adata_filtered.uns["ordered_all_onehot_example"] = onehot
        
        # Standard scanpy preprocessing
        adata_filtered.X = adata_filtered.layers["counts"].copy()
        
        print("Performing preprocessing...")
        sc.pp.normalize_total(adata_filtered, target_sum=1e4)
        sc.pp.log1p(adata_filtered)
        sc.pp.highly_variable_genes(adata_filtered, min_mean=0.0125, max_mean=5, min_disp=0.5)
        sc.tl.rank_genes_groups(
            adata_filtered, 
            n_genes=50, 
            method="t-test", 
            corr_method="benjamini-hochberg",
            groupby="condition", 
            reference="ctrl"
        )
        
        # Get all conditions and cell types (excluding control)
        all_conditions = sorted([c for c in adata_filtered.obs["condition"].unique() 
                               if c not in control_conditions and adata_filtered.obs.loc[adata_filtered.obs["condition"] == c, "smiles"].iloc[0] != ""])
        all_cell_types = sorted(adata_filtered.obs["cell_type"].unique())
        
        print(f"Conditions to process: {all_conditions}")
        print(f"Cell types to process: {all_cell_types}")
        
        # Process each condition-celltype combination
        for condition in all_conditions:
            for cell_type in all_cell_types:
                # Check if combination exists
                combo_mask = ((adata_filtered.obs["condition"] == condition) & 
                             (adata_filtered.obs["cell_type"] == cell_type))
                
                if not combo_mask.any():
                    print(f"  Skipping {condition} - {cell_type}: No samples found")
                    continue
                
                success = process_dataset_condition_celltype(
                    dataset_name, condition, cell_type, adata_filtered,
                    dataset_output_dir
                )
                
                if success:
                    print(f"  ✓ Completed: {condition} - {cell_type}")
                else:
                    print(f"  ✗ Failed: {condition} - {cell_type}")
                    
    except Exception as e:
        print(f"Dataset {dataset_name} failed completely: {e}")
        traceback.print_exc()
        continue

print(f"\n{'='*60}")
print("All datasets processed!")
print(f"Results saved to: {OUTPUT_ROOT}")
print(f"Runtime log: {output_time_log}")
print(f"{'='*60}")