import sys
import numpy as np
import scanpy as sc
import os
os.chdir('/home/wenyu/perturb-benchmark/VAE/trvae-pytorch')
import sys
sys.path.append("../")
import time
import trvae
import pandas as pd
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


# Directory containing all .h5ad files
data_dir = "/data2/lanxiang/data/Task3_data"

# Output root directory
output_root = "/data2/lanxiang/perturb_benchmark_v2/model/trVAE"

# Runtime summary output
runtime_log_path = os.path.join(output_root, "trvae_runtime_summary.txt")

# Category 1 datasets: only one perturbation vs control
single_condition_datasets = {
    "Kang": "stimulated",
    "Haber": "Hpoly.Day10",
    "Hagai": "LPS6",
    "Weinreb_time": "developed",
}

# Category 2 datasets: multiple perturbations vs control
multi_condition_datasets = [
    "Srivatsan_sciplex3_sub10",
    "Burkhardt_sub10",
    "Tahoe100_sub10",
    "Perturb_KHP_sub10",
    "Perturb_cmo_V1_sub10",
    "Parse_10M_PBMC_sub10"
]

# Constants
DEFAULT_CELL_TYPE_KEY = "cell_type"
DEFAULT_CONDITION_KEY = "condition"
DEFAULT_CONTROL = "ctrl"

# Function to run trVAE model on a dataset
def run_trvae_on_dataset(data_name, data_path, control_condition, target_conditions, cell_type_key, condition_key):
    start_time = time.time()
    print(f"\n================ Processing {data_name} =================")

    adata = sc.read(data_path)
    cell_types = adata.obs[cell_type_key].unique().tolist()

    for target_condition in target_conditions:
        labelencoder = {control_condition: 0, target_condition: 1}

        # Filter samples to only include current control and target
        filtered_adata = adata[adata.obs[condition_key].isin([control_condition, target_condition])]

        for specific_celltype in cell_types:
            print(f"---- Training {data_name} | Condition: {target_condition} | Cell type: {specific_celltype}")

            # Exclude target-condition cells of current cell type from training
            net_train_adata = filtered_adata[
                ~((filtered_adata.obs[cell_type_key] == specific_celltype) &
                  (filtered_adata.obs[condition_key] == target_condition))
            ]

            # Create and train model
            model = trvae.CVAE(
                net_train_adata.n_vars, num_classes=2,
                encoder_layer_sizes=[128, 32], decoder_layer_sizes=[32, 128],
                latent_dim=10, alpha=0.0001, use_mmd=True, beta=10
            )
            trainer = trvae.Trainer(model, net_train_adata)
            trainer.train_trvae(n_epochs=1000, batch_size=512, early_patience=20)

            # Prepare test data: source condition cells of current cell type
            source_adata = filtered_adata[
                (filtered_adata.obs[cell_type_key] == specific_celltype) &
                (filtered_adata.obs[condition_key] == control_condition)
            ]

            source_adata = source_adata[:, net_train_adata.var_names]  # Ensure same genes

            # Run prediction
            pred = model.predict(
                x=source_adata.X.toarray().astype(np.float32),
                y=source_adata.obs[condition_key].tolist(),
                target=target_condition
            )

            adata_pred = sc.AnnData(pred)
            adata_pred.obs[condition_key] = [f"pred_{target_condition}"] * pred.shape[0]
            adata_pred.obs[cell_type_key] = specific_celltype
            adata_pred.var["feature"] = net_train_adata.var_names

            # Save results
            output_dir = os.path.join(output_root, data_name, f"{control_condition}_to_{target_condition}")
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{specific_celltype}_pred.h5ad")
            adata_pred.write_h5ad(save_path)

            print(f"Saved prediction to {save_path}")

    # Record runtime
    elapsed_time = time.time() - start_time
    elapsed_min = round(elapsed_time / 60, 2)
    with open(runtime_log_path, "a") as f:
        f.write(f"{data_name}\t{elapsed_min} min\n")

    print(f"Finished {data_name} in {elapsed_min} min.")

# ---------------- Run All Datasets ----------------
if __name__ == "__main__":
    # Clear old runtime summary
    if os.path.exists(runtime_log_path):
        os.remove(runtime_log_path)

    # Run single-condition datasets
    for data_name, target_condition in single_condition_datasets.items():
        data_path = os.path.join(data_dir, f"{data_name}.h5ad")
        run_trvae_on_dataset(
            data_name=data_name,
            data_path=data_path,
            control_condition=DEFAULT_CONTROL,
            target_conditions=[target_condition],
            cell_type_key=DEFAULT_CELL_TYPE_KEY,
            condition_key=DEFAULT_CONDITION_KEY,
        )

    # Run multi-condition datasets
    for data_name in multi_condition_datasets:
        data_path = os.path.join(data_dir, f"{data_name}.h5ad")
        adata = sc.read(data_path)
        all_conditions = adata.obs[DEFAULT_CONDITION_KEY].unique().tolist()
        target_conditions = [cond for cond in all_conditions if cond != DEFAULT_CONTROL]

        run_trvae_on_dataset(
            data_name=data_name,
            data_path=data_path,
            control_condition=DEFAULT_CONTROL,
            target_conditions=target_conditions,
            cell_type_key=DEFAULT_CELL_TYPE_KEY,
            condition_key=DEFAULT_CONDITION_KEY,
        )



