import os
import time
import pickle
from pathlib import Path
import scanpy as sc
import cpa
from cpa.helper import rank_genes_groups_by_cov
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
# Set scanpy plotting settings
sc.settings.set_figure_params(dpi=100)

# Dictionary containing dataset info: .h5ad path and prefix to pkl files
datasets = {
    "DixitRegev2016": {
        "h5ad": "/data2/lanxiang/data/Task2_data/DixitRegev2016.h5ad",
        "pkl_prefix": "/data2/yue_data/pert/data/dixitregev2016/splits/dixitregev2016_simulation_1_0.75"
    },
    "NormanWeissman2019_filtered": {
        "h5ad": "/data2/lanxiang/data/Task2_data/NormanWeissman2019_filtered.h5ad",
        "pkl_prefix": "/data2/yue_data/pert/data/normanweissman2019_filtered/splits/normanweissman2019_filtered_simulation_1_0.75"
    },
    "Sunshine2023_CRISPRi_sarscov2": {
        "h5ad": "/data2/lanxiang/data/Task2_data/Sunshine2023_CRISPRi_sarscov2.h5ad",
        "pkl_prefix": "/data2/yue_data/pert/data/sunshine2023_crispri_sarscov2/splits/sunshine2023_crispri_sarscov2_simulation_1_0.75"
    },
    "Arce_MM_CRISPRi_sub": {
        "h5ad": "/data2/lanxiang/data/Task2_data/Arce_MM_CRISPRi_sub.h5ad",
        "pkl_prefix": "/data2/yue_data/pert/data/arce_mm_crispri_filtered/splits/arce_mm_crispri_filtered_simulation_1_0.75"
    }
}

# Output directory for all results
output_base = "/data2/lanxiang/perturb_benchmark_v2/model/CPA"
os.makedirs(output_base, exist_ok=True)

# File to store training time logs
time_log_file = os.path.join(output_base, "cpa_runtime_summary.txt")
with open(time_log_file, "w") as f:
    f.write("Dataset\tTime\n")

# Loop over datasets
for dataset_name, paths in datasets.items():
    print(f"\n🔧 Processing {dataset_name}...")
    start_time = time.time()

    # Load .h5ad
    adata = sc.read(paths["h5ad"])

    # Load split and subgroup files
    with open(paths["pkl_prefix"] + ".pkl", "rb") as f:
        split_data = pickle.load(f)
    with open(paths["pkl_prefix"] + "_subgroup.pkl", "rb") as f:
        subgroup_data = pickle.load(f)

    # Assign split: default unknown
    adata.obs["split"] = "unknown"
    adata.obs.loc[adata.obs["condition"].isin(split_data["train"]), "split"] = "train"

    # Assign valid from val_subgroup
    valid_conditions = set()
    for subgroup in subgroup_data["val_subgroup"].values():
        valid_conditions.update(subgroup)
    adata.obs.loc[adata.obs["condition"].isin(valid_conditions), "split"] = "valid"

    # Assign ood from test_subgroup
    ood_conditions = set()
    for subgroup in subgroup_data["test_subgroup"].values():
        ood_conditions.update(subgroup)
    adata.obs.loc[adata.obs["condition"].isin(ood_conditions), "split"] = "ood"

    # Check for unassigned conditions
    unknown_conditions = adata.obs.loc[adata.obs["split"] == "unknown", "condition"].unique()
    if len(unknown_conditions) > 0:
        print(f"Unassigned conditions: {unknown_conditions}")
    else:
        print("All conditions labeled.")

    # Preprocess condition names and dose values
    adata.obs["condition"] = adata.obs["condition"].str.replace(r"\+ctrl", "", regex=True)
    adata.obs["dose_value"] = adata.obs["condition"].apply(lambda x: '+'.join(['1.0' for _ in x.split('+')]))
    adata.obs["cov_cond"] = adata.obs["cell_type"].astype(str) + '_' + adata.obs["condition"].astype(str)

    # Run differential expression
    rank_genes_groups_by_cov(adata, groupby="cov_cond", covariate="cell_type", control_group="ctrl", n_genes=20)

    # Assign counts layer to .X
    adata.X = adata.layers["counts"].copy()

    # Setup CPA model
    cpa.CPA.setup_anndata(
        adata,
        perturbation_key="condition",
        control_group="ctrl",
        dosage_key="dose_value",
        categorical_covariate_keys=["cell_type"],
        is_count_data=True,
        deg_uns_key="rank_genes_groups_cov",
        deg_uns_cat_key="cov_cond",
        max_comb_len=2
    )

    # Define model hyperparameters
    model_params = {
        "n_latent": 32,
        "recon_loss": "nb",
        "doser_type": "linear",
        "n_hidden_encoder": 256,
        "n_layers_encoder": 4,
        "n_hidden_decoder": 256,
        "n_layers_decoder": 2,
        "use_batch_norm_encoder": True,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": False,
        "use_layer_norm_decoder": False,
        "dropout_rate_encoder": 0.2,
        "dropout_rate_decoder": 0.0,
        "variational": False,
        "seed": 8206,
    }

    trainer_params = {
        "n_epochs_kl_warmup": None,
        "n_epochs_adv_warmup": 50,
        "n_epochs_mixup_warmup": 10,
        "n_epochs_pretrain_ae": 10,
        "mixup_alpha": 0.1,
        "lr": 0.0001,
        "wd": 3.2e-6,
        "adv_steps": 3,
        "reg_adv": 10.0,
        "pen_adv": 20.0,
        "adv_lr": 0.0001,
        "adv_wd": 7e-6,
        "n_layers_adv": 2,
        "n_hidden_adv": 128,
        "use_batch_norm_adv": True,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.3,
        "step_size_lr": 25,
        "do_clip_grad": False,
        "adv_loss": "cce",
        "gradient_clip_value": 5.0,
    }

    # Define output folder for current dataset
    output_dir = os.path.join(output_base, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Train CPA model
    model = cpa.CPA(
        adata=adata,
        split_key="split",
        train_split="train",
        valid_split="valid",
        test_split="ood",
        **model_params,
    )

    model.train(
        max_epochs=2000,
        use_gpu=True,
        batch_size=2048,
        plan_kwargs=trainer_params,
        early_stopping_patience=5,
        check_val_every_n_epoch=5,
        save_path=output_dir
    )

    # Predict and save results
    model.predict(adata, batch_size=2048)
    adata.write(os.path.join(output_dir, f"{dataset_name}_pred.h5ad"))

    # Record runtime
    elapsed = time.time() - start_time
    if elapsed >= 3600:
        run_time_str = f"{round(elapsed / 3600, 1)}h"
    else:
        run_time_str = f"{round(elapsed / 60)}min"
    print(f"⏱️ {dataset_name} finished in {run_time_str}")

    # Append to log file
    with open(time_log_file, "a") as f:
        f.write(f"{dataset_name}\t{run_time_str}\n")

print("All datasets completed. Runtime summary saved.")
