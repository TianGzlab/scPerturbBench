import sys
import os
import pickle
import shutil
from scipy import sparse
import scanpy as sc
import anndata as ad
import scvi
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from datetime import timedelta
from perturbnet.util import * 
from perturbnet.cinn.flow import * 
from perturbnet.genotypevae.genotypeVAE import *
from perturbnet.data_vae.vae import *
from perturbnet.cinn.flow_generate import SCVIZ_CheckNet2Net

# Set GPU device
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

# ================================
# Configuration Paths
# ================================
DATA_DIR = "/data2/lanxiang/data/Task1_data"
PKL_BASE_DIR = "/data2/yue_data/pert/data"
OUTPUT_BASE_DIR = "/data2/lanxiang/perturb_benchmark_v2/model/PerturbNet"
TIME_LOG_PATH = os.path.join(OUTPUT_BASE_DIR, "time_log.txt")
PRETRAINED_GENOVAE_PATH = "/data2/lanxiang/perturb_benchmarking/test_model_data/perturbNet/pretrained_model/genotypeVAE"

# Special dataset name mapping for inconsistent naming
SPECIAL_MAPPING = {
    "Arce_MM_CRISPRi_sub": "arce_mm_crispri_filtered",
    "Junyue_Cao": "junyue_cao_filtered"
}

# Datasets to skip during processing
SKIP_DATASETS = ["Adamson.h5ad"]


def get_dataset_files():
    """Get all valid dataset files for processing"""
    all_datasets = [f for f in os.listdir(DATA_DIR) 
                   if f.endswith(".h5ad") and f not in SKIP_DATASETS]
    return sorted(all_datasets)


def initialize_time_log():
    """Initialize time log file with header"""
    with open(TIME_LOG_PATH, "w") as time_log:
        time_log.write("Dataset\tTime\n")


def get_pkl_folder_name(dataset_name):
    """Get PKL folder name with special mapping handling"""
    if dataset_name in SPECIAL_MAPPING:
        return SPECIAL_MAPPING[dataset_name]
    else:
        # Standard mapping: convert to lowercase and replace spaces
        return dataset_name.lower().replace(" ", "_")


def load_and_preprocess_data(data_path):
    """Load and preprocess single-cell data"""
    print(f"Loading data from: {data_path}")
    
    # Load data
    adata = sc.read(data_path)
    adata.X = adata.layers["counts"].copy()
    
    # Replace '+' with '/' in condition names for consistency
    adata.obs["condition"] = adata.obs["condition"].str.replace('+', '/', regex=False)
    
    # Basic preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, 
        min_mean=0.0125, 
        max_mean=5, 
        min_disp=0.5
    )
    
    # Differential expression analysis
    sc.tl.rank_genes_groups(
        adata, 
        n_genes=50, 
        method="t-test", 
        corr_method="benjamini-hochberg",
        groupby="condition", 
        reference="ctrl"
    )
    
    print(f"Preprocessed data shape: {adata.shape}")
    return adata


def create_ordered_perturbation_list(adata):
    """Create ordered perturbation list with ctrl first"""
    conditions = sorted(set(adata.obs["condition"]))
    ctrl_first = ['ctrl'] if 'ctrl' in conditions else []
    others = [c for c in conditions if c != 'ctrl']
    ordered_perturbations = ctrl_first + others
    
    adata.uns["ordered_all_trt"] = np.array(ordered_perturbations)
    print(f"Created ordered perturbation list with {len(ordered_perturbations)} conditions")
    
    return adata


def load_gene_annotation_data():
    """Load pretrained gene annotation data for one-hot encoding"""
    print("Loading gene annotation data...")
    
    gene_data = np.load(os.path.join(PRETRAINED_GENOVAE_PATH, 'gene.npy'))
    data_npz = sparse.load_npz(
        os.path.join(PRETRAINED_GENOVAE_PATH, 'sparse_gene_anno_matrix.npz')
    ).toarray()
    
    print(f"Gene annotation data shape: {data_npz.shape}")
    return gene_data, data_npz


def create_onehot_matrix(adata, gene_data, data_npz):
    """Create one-hot encoding matrix for perturbations"""
    print("Creating one-hot encoding matrix...")
    
    treatment_list = adata.uns["ordered_all_trt"]
    onehot_matrix = []
    
    for trt in treatment_list:
        if trt == "ctrl":
            genes = []
        else:
            genes = trt.split('/')
        
        # Find gene indices in annotation data
        indices = [np.where(gene_data == g)[0][0] for g in genes if g in gene_data]
        
        if len(indices) == 0:
            # No genes found, create zero vector
            onehot = np.zeros(data_npz.shape[1], dtype=int)
        else:
            # Combine annotations for multiple genes using logical OR
            onehot = data_npz[indices[0]]
            for idx in indices[1:]:
                onehot = np.logical_or(onehot, data_npz[idx])
            onehot = onehot.astype(int)
        
        onehot_matrix.append(onehot)
    
    adata.uns["ordered_all_onehot"] = np.array(onehot_matrix)
    print(f"Created one-hot matrix shape: {adata.uns['ordered_all_onehot'].shape}")
    
    return adata


def load_split_information(pkl_prefix):
    """Load train/validation/test split information"""
    print(f"Loading split information from: {pkl_prefix}")
    
    # Load main split data
    with open(pkl_prefix + ".pkl", "rb") as f:
        split_data = pickle.load(f)
    
    # Load subgroup split data
    with open(pkl_prefix + "_subgroup.pkl", "rb") as f:
        subgroup_data = pickle.load(f)
    
    return split_data, subgroup_data


def assign_data_splits(adata, split_data, subgroup_data):
    """Assign train/validation/test splits to data"""
    print("Assigning data splits...")
    
    # Map conditions back to original format for split assignment
    perturb_map = adata.obs["condition"].map(lambda x: x.replace("/", "+"))
    adata.obs["split_example"] = "unknown"
    
    # Assign train split
    adata.obs.loc[perturb_map.isin(split_data["train"]), "split_example"] = "train"
    
    # Assign validation split
    valid_conditions = set()
    for subgroup in subgroup_data["val_subgroup"].values():
        valid_conditions.update(subgroup)
    adata.obs.loc[perturb_map.isin(valid_conditions), "split_example"] = "valid"
    
    # Assign test split (out-of-distribution)
    ood_conditions = set()
    for subgroup in subgroup_data["test_subgroup"].values():
        ood_conditions.update(subgroup)
    adata.obs.loc[perturb_map.isin(ood_conditions), "split_example"] = "test"
    
    # Check for unassigned perturbations
    unknown_perturbs = adata.obs.loc[adata.obs["split_example"] == "unknown", "condition"].unique()
    if len(unknown_perturbs) > 0:
        print(f"⚠️ Unassigned perturbations: {unknown_perturbs}")
    
    # Print split statistics
    print("Split distribution:")
    print(adata.obs["split_example"].value_counts())
    
    return adata


def split_train_test_data(adata):
    """Split data into training and test sets"""
    adata_train = adata[adata.obs.split_example == "train", :].copy()
    adata_test = adata[adata.obs.split_example == "test", :].copy()
    
    print(f"Training data shape: {adata_train.shape}")
    print(f"Test data shape: {adata_test.shape}")
    
    return adata_train, adata_test


def train_scvi_model(adata_train, save_path):
    """Train scVI model for cell representation learning"""
    print("Training scVI model...")
    
    # Setup data and create model
    scvi.data.setup_anndata(adata_train, layer="counts")
    scvi_model = scvi.model.SCVI(adata_train, n_latent=10)
    
    # Train model
    scvi_model.train(n_epochs=500, frequency=20)
    
    # Save model
    scvi_model.save(save_path)
    print(f"scVI model saved to: {save_path}")
    
    return scvi_model


def load_scvi_model(save_path, adata_train):
    """Load pretrained scVI model"""
    print(f"Loading scVI model from: {save_path}")
    
    scvi.data.setup_anndata(adata_train, layer="counts")
    scvi_model = scvi.model.SCVI.load(save_path, adata_train, use_cuda=False)
    
    return scvi_model


def load_pretrained_genotype_vae():
    """Load pretrained genotype VAE model"""
    print("Loading pretrained genotype VAE...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(PRETRAINED_GENOVAE_PATH, "model_params.pt")
    
    model_genovae = GenotypeVAE().to(device)
    model_genovae.load_state_dict(torch.load(model_path, map_location=device))
    model_genovae.eval()
    
    print(f"GenotypeVAE loaded on device: {device}")
    return model_genovae


def prepare_flow_embeddings(adata_train):
    """Prepare embeddings for conditional flow training"""
    print("Preparing embeddings for conditional flow...")
    
    cond_stage_data, embeddings, perturbToEmbed = prepare_embeddings_cinn(
        adata_train,
        perturbation_key="condition", 
        trt_key="ordered_all_trt", 
        embed_key="ordered_all_onehot"
    )
    
    print("Embeddings prepared successfully")
    return cond_stage_data, embeddings, perturbToEmbed


def create_conditional_flow():
    """Create conditional normalizing flow model"""
    print("Creating conditional normalizing flow...")
    
    torch.manual_seed(42)
    flow_model = ConditionalFlatCouplingFlow(
        conditioning_dim=10,
        embedding_dim=10,
        conditioning_depth=2,
        n_flows=20,
        in_channels=10,
        hidden_dim=1024,
        hidden_depth=2,
        activation="none",
        conditioner_use_bn=True
    )
    
    print("Conditional flow model created")
    return flow_model


def train_perturbnet_model(flow_model, cond_stage_data, model_genovae, scvi_model, 
                          perturbToEmbed, embeddings, save_path):
    """Train complete PerturbNet model"""
    print("Training PerturbNet model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create combined model
    model_c = Net2NetFlow_scVIGenoFlow(
        configured_flow=flow_model,
        cond_stage_data=cond_stage_data,
        model_con=model_genovae,
        scvi_model=scvi_model,
        perturbToOnehotLib=perturbToEmbed,
        oneHotData=embeddings
    )
    
    # Train model
    model_c.to(device=device)
    model_c.train(n_epochs=45, batch_size=128, lr=4.5e-6)
    
    # Save model
    model_c.save(save_path)
    print(f"PerturbNet model saved to: {save_path}")
    
    return model_c


def load_perturbnet_model(flow_model, cond_stage_data, model_genovae, scvi_model,
                         perturbToEmbed, embeddings, model_path):
    """Load trained PerturbNet model"""
    print(f"Loading PerturbNet model from: {model_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_c = Net2NetFlow_scVIGenoFlow(
        configured_flow=flow_model,
        cond_stage_data=cond_stage_data,
        model_con=model_genovae,
        scvi_model=scvi_model,
        perturbToOnehotLib=perturbToEmbed,
        oneHotData=embeddings
    )
    
    model_c.load(model_path)
    model_c.to(device=device)
    
    return model_c


def setup_prediction_models(model_c, scvi_model, adata_train):
    """Setup models for prediction"""
    print("Setting up prediction models...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set models to evaluation mode
    model_c.eval()
    
    # Create predictive models
    scvi_model_de = scvi_predictive_z(scvi_model)
    perturbnet_model = SCVIZ_CheckNet2Net(model_c, device, scvi_model_de)
    
    # Get latent library size samples from training data
    Lsample_obs = scvi_model.get_latent_library_size(adata=adata_train, give_mean=False)
    
    print("Prediction models ready")
    return perturbnet_model, Lsample_obs, device


def create_perturbation_mapping(adata_test):
    """Create mapping from perturbation names to one-hot encodings"""
    pert_names = adata_test.uns["ordered_all_trt"]
    pert_to_onehot = {
        name: adata_test.uns["ordered_all_onehot"][i] for i, name in enumerate(pert_names)
    }
    
    print(f"Created perturbation mapping for {len(pert_to_onehot)} perturbations")
    return pert_to_onehot


def predict_perturbation_effects(adata_test, model_c, perturbnet_model, 
                                Lsample_obs, pert_to_onehot, save_dir, device):
    """Predict effects for all test perturbations"""
    print("Predicting perturbation effects...")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all unique perturbations from test data
    perturbations = adata_test.obs["condition"].unique().tolist()
    prediction_count = 0
    
    for pert in perturbations:
        # Find cells for this perturbation
        cell_idx = np.where(adata_test.obs["condition"] == pert)[0]
        if len(cell_idx) == 0:
            print(f"[Skipped] {pert} has no cells")
            continue

        if pert not in pert_to_onehot:
            print(f"[Skipped] {pert} not in perturbation list")
            continue

        n_cells = len(cell_idx)
        pert_onehot = pert_to_onehot[pert]
        trt_input_onehot = np.tile(pert_onehot, (n_cells, 1))

        # Compute perturbation embedding
        with torch.no_grad():
            _, _, _, embdata_torch = model_c.model_con(
                torch.tensor(trt_input_onehot).float().to(device)
            )
        pert_embed = embdata_torch.cpu().numpy()

        # Sample background latent representations
        Lsample_idx = np.random.choice(range(Lsample_obs.shape[0]), n_cells, replace=True)
        library_trt_latent = Lsample_obs[Lsample_idx]

        # Predict expression
        predict_latent, predict_data = perturbnet_model.sample_data(pert_embed, library_trt_latent)

        # Get real expression data (dense matrix)
        real_data = adata_test.layers["counts"].A[cell_idx]

        # Create safe filename by replacing problematic characters
        safe_name = pert.replace("/", "+")

        # Save predictions and real data as numpy arrays
        np.save(os.path.join(save_dir, f"{safe_name}_predict.npy"), predict_data)
        np.save(os.path.join(save_dir, f"{safe_name}_real.npy"), real_data)

        prediction_count += 1
        print(f"[Completed] {pert}, cells: {n_cells}, saved as: {safe_name}_predict.npy and {safe_name}_real.npy")
    
    print(f"Completed predictions for {prediction_count} perturbations")
    return prediction_count


def log_processing_time(dataset_name, start_time, success=True, error_msg=""):
    """Log processing time to file"""
    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed))).split(":")
    hours, minutes = int(elapsed_str[0]), int(elapsed_str[1])
    
    time_str = f"{hours}h{minutes}min" if hours > 0 else f"{minutes}min"
    
    with open(TIME_LOG_PATH, "a") as time_log:
        if success:
            time_log.write(f"{dataset_name}\t{time_str}\n")
        else:
            time_log.write(f"{dataset_name}\tERROR ({error_msg})\n")
    
    return time_str


def process_single_dataset(dataset_file):
    """Process a single dataset through the complete PerturbNet pipeline"""
    dataset_name = os.path.splitext(dataset_file)[0]
    start_time = time.time()
    
    try:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_file}")
        print(f"{'='*50}")
        
        # ===== 1. Setup paths =====
        data_path = os.path.join(DATA_DIR, dataset_file)
        pkl_folder = get_pkl_folder_name(dataset_name)
        pkl_prefix = os.path.join(
            PKL_BASE_DIR, 
            pkl_folder,
            "splits",
            f"{pkl_folder}_simulation_1_0.75"
        )
        dataset_output_dir = os.path.join(OUTPUT_BASE_DIR, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # ===== 2. Load and preprocess data =====
        adata = load_and_preprocess_data(data_path)
        adata = create_ordered_perturbation_list(adata)
        
        # ===== 3. Create one-hot encodings =====
        gene_data, data_npz = load_gene_annotation_data()
        adata = create_onehot_matrix(adata, gene_data, data_npz)
        
        # ===== 4. Load and assign data splits =====
        split_data, subgroup_data = load_split_information(pkl_prefix)
        adata = assign_data_splits(adata, split_data, subgroup_data)
        adata_train, adata_test = split_train_test_data(adata)
        
        # ===== 5. Train scVI model =====
        scvi_model_save_path = os.path.join(dataset_output_dir, "cellvae")
        scvi_model = train_scvi_model(adata_train, scvi_model_save_path)
        
        # Reload model for consistency
        scvi_model = load_scvi_model(scvi_model_save_path, adata_train)
        
        # ===== 6. Load pretrained genotype VAE =====
        model_genovae = load_pretrained_genotype_vae()
        
        # ===== 7. Prepare embeddings and train flow =====
        cond_stage_data, embeddings, perturbToEmbed = prepare_flow_embeddings(adata_train)
        flow_model = create_conditional_flow()
        
        cinn_model_save_path = os.path.join(dataset_output_dir, "cinn")
        model_c = train_perturbnet_model(
            flow_model, cond_stage_data, model_genovae, scvi_model,
            perturbToEmbed, embeddings, cinn_model_save_path
        )
        
        # ===== 8. Reload trained model for prediction =====
        flow_model_pred = create_conditional_flow()
        model_c = load_perturbnet_model(
            flow_model_pred, cond_stage_data, model_genovae, scvi_model,
            perturbToEmbed, embeddings, cinn_model_save_path
        )
        
        # ===== 9. Setup prediction pipeline =====
        # Setup test data with scVI
        scvi.data.setup_anndata(adata_test, layer="counts")
        Zsample_test = scvi_model.get_latent_representation(adata=adata_test, give_mean=False)
        
        perturbnet_model, Lsample_obs, device = setup_prediction_models(model_c, scvi_model, adata_train)
        
        # ===== 10. Make predictions =====
        save_dir = os.path.join(dataset_output_dir, "perturbnet_predictions")
        pert_to_onehot = create_perturbation_mapping(adata_test)
        
        prediction_count = predict_perturbation_effects(
            adata_test, model_c, perturbnet_model, Lsample_obs, 
            pert_to_onehot, save_dir, device
        )
        
        # ===== 11. Log success =====
        time_str = log_processing_time(dataset_name, start_time, success=True)
        print(f"✅ Completed {dataset_file} in {time_str}")
        print(f"📊 Predictions made: {prediction_count}")
        
        return True
        
    except Exception as e:
        time_str = log_processing_time(dataset_name, start_time, success=False, error_msg=str(e))
        print(f"❌ Failed {dataset_file} after {time_str}: {str(e)}")
        return False


def main():
    """Main function to process all datasets"""
    print("🌟 Starting PerturbNet batch processing...")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"📁 Output directory: {OUTPUT_BASE_DIR}")
    print(f"🎯 Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
    
    # Create output directory and initialize time log
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    initialize_time_log()
    
    # Get all dataset files
    dataset_files = get_dataset_files()
    print(f"📋 Found {len(dataset_files)} datasets to process")
    
    # Process each dataset
    successful_count = 0
    failed_count = 0
    
    for i, dataset_file in enumerate(dataset_files, 1):
        print(f"\n{'🔄' * 20}")
        print(f"Progress: {i}/{len(dataset_files)}")
        print(f"Successfully processed: {successful_count}")
        print(f"Failed: {failed_count}")
        print(f"{'🔄' * 20}")
        
        success = process_single_dataset(dataset_file)
        
        if success:
            successful_count += 1
        else:
            failed_count += 1
    
    # Print final statistics
    print(f"\n{'🎊' * 50}")
    print("🏆 Final Results")
    print(f"{'🎊' * 50}")
    print(f"✅ Successfully processed: {successful_count} datasets")
    print(f"❌ Failed: {failed_count} datasets")
    print(f"📈 Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%")
    print(f"📁 Results saved in: {OUTPUT_BASE_DIR}")
    print(f"⏰ Time log saved in: {TIME_LOG_PATH}")


if __name__ == "__main__":
    try:
        main()
        print("\n🎉 All datasets processing completed!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Processing interrupted by user")
        
    except Exception as e:
        print(f"\n\n❌ Fatal error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        print("\n🧹 Cleaning up...")