import sys
import logging
import scanpy as sc
import scgen
import os
import time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
logger = logging.getLogger("scvi.inference.autotune")
logger.setLevel(logging.WARNING)
os.chdir('/data2/lanxiang/perturb_benchmark_v2/model/scGEN')


# Directory containing all h5ad datasets
data_dir = "/data2/lanxiang/data/Task3_data"

# File to save runtime summary
output_time_log = "scgen_runtime_summary.txt"

# Define single-condition datasets and their 'case' condition
single_condition_config = {
    "Kang": "stimulated",
    "Haber": "Hpoly.Day10",
    "Hagai": "LPS6",
    "Weinreb_time": "developed"
}
control_label = "ctrl"

# Clear the runtime summary file
with open(output_time_log, "w") as f:
    f.write("Dataset\tTime\n")

# Loop through each .h5ad file in the directory
for file in os.listdir(data_dir):
    if not file.endswith(".h5ad"):
        continue

    dataset_name = file.replace(".h5ad", "")
    file_path = os.path.join(data_dir, file)

    print(f"================= Processing dataset: {dataset_name} =================")
    start_time = time.time()

    # Load the dataset
    adata = sc.read_h5ad(file_path)

    # Define the keys for cell type and condition
    cell_type_key = "cell_type"
    condition_key = "condition"
    # Get unique cell types
    cell_type_list = adata.obs[cell_type_key].unique().tolist()

    # Check whether this is a single-condition or multi-condition dataset
    if dataset_name in single_condition_config:
        # Single-condition case
        case_condition = single_condition_config[dataset_name]
        for cell_type in cell_type_list:
            print(f"Processing {cell_type} for condition {case_condition}")

            # Exclude the case condition for this cell type from training data
            train_filtered = adata[~((adata.obs[cell_type_key] == cell_type) &
                                     (adata.obs[condition_key] == case_condition))].copy()

            # Prepare AnnData for scGen
            scgen.SCGEN.setup_anndata(train_filtered, batch_key=condition_key, labels_key=cell_type_key)

            # Initialize and train scGen model
            model = scgen.SCGEN(train_filtered)
            model.train(max_epochs=100, batch_size=32, early_stopping=True, early_stopping_patience=25)

            # Predict the response of this cell type to the case condition
            pred_adata, delta = model.predict(ctrl_key=control_label, stim_key=case_condition,
                                              celltype_to_predict=cell_type)
            pred_adata.obs[condition_key] = f"pred_{case_condition}"

            # Save prediction
            output_dir = f"./{dataset_name}_pred_data/{case_condition}"
            os.makedirs(output_dir, exist_ok=True)
            pred_adata.write_h5ad(f"{output_dir}/pred_adata_{cell_type}.h5ad")

    else:
        # Multi-condition case
        # Extract all conditions except the control
        condition_list = sorted(list(set(adata.obs[condition_key]) - {control_label}))
        for condition in condition_list:
            for cell_type in cell_type_list:
                print(f"Processing {cell_type} for condition {condition}")

                # Exclude the case condition for this cell type from training data
                train_filtered = adata[~((adata.obs[cell_type_key] == cell_type) &
                                         (adata.obs[condition_key] == condition))].copy()

                # Prepare AnnData for scGen
                scgen.SCGEN.setup_anndata(train_filtered, batch_key=condition_key, labels_key=cell_type_key)

                # Initialize and train scGen model
                model = scgen.SCGEN(train_filtered)
                model.train(max_epochs=100, batch_size=32, early_stopping=True, early_stopping_patience=25)

                # Predict the response of this cell type to the current condition
                pred_adata, delta = model.predict(ctrl_key=control_label, stim_key=condition,
                                                  celltype_to_predict=cell_type)
                pred_adata.obs[condition_key] = f"pred_{condition}"

                # Save prediction
                output_dir = f"./{dataset_name}_pred_data/{condition}"
                os.makedirs(output_dir, exist_ok=True)
                pred_adata.write_h5ad(f"{output_dir}/pred_adata_{cell_type}.h5ad")

    # Calculate runtime
    end_time = time.time()
    duration_sec = end_time - start_time
    duration_str = f"{int(duration_sec // 3600)}h{int((duration_sec % 3600) // 60)}min"

    # Log runtime to summary file
    with open(output_time_log, "a") as f:
        f.write(f"{dataset_name}\t{duration_str}\n")
