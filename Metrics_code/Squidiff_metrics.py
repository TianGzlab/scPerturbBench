import os
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, wasserstein_distance
from scipy.spatial.distance import cdist
from warnings import catch_warnings, simplefilter

np.random.seed(42)

def compute_mmd(pred_array, truth_array, kernel='rbf', gamma=1.0):
    if kernel == 'rbf':
        dist_pred = cdist(pred_array, pred_array, metric='sqeuclidean')
        dist_truth = cdist(truth_array, truth_array, metric='sqeuclidean')
        dist_cross = cdist(pred_array, truth_array, metric='sqeuclidean')

        Kxx = np.exp(-gamma * dist_pred)
        Kyy = np.exp(-gamma * dist_truth)
        Kxy = np.exp(-gamma * dist_cross)

        return np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)
    else:
        raise ValueError("Unsupported kernel type. Use 'rbf'.")

def compute_wasserstein(pred_array, truth_array):
    return wasserstein_distance(pred_array.flatten(), truth_array.flatten())

# ========== Main metric computation function ==========
def calculate_metrics_all(pred_data, true_data, ctrl_data):
    if isinstance(true_data, pd.DataFrame):
        true_data = true_data.values
    if isinstance(pred_data, pd.DataFrame):
        pred_data = pred_data.values
    if isinstance(ctrl_data, pd.DataFrame):
        ctrl_data = ctrl_data.values

    pred_data = pred_data.toarray() if not isinstance(pred_data, np.ndarray) else pred_data
    true_data = true_data.toarray() if not isinstance(true_data, np.ndarray) else true_data
    ctrl_data = ctrl_data.toarray() if not isinstance(ctrl_data, np.ndarray) else ctrl_data

    metrics = {}
    mean_true = np.mean(true_data, axis=0)
    mean_pred = np.mean(pred_data, axis=0)
    mean_ctrl = np.mean(ctrl_data, axis=0)

    with catch_warnings():
        simplefilter("ignore")
        try:
            metrics['R_squared'] = r2_score(mean_true, mean_pred)
            metrics['R_squared_delta'] = r2_score(mean_true - mean_ctrl, mean_pred - mean_ctrl)
        except:
            metrics['R_squared'] = np.nan
            metrics['R_squared_delta'] = np.nan

        try:
            metrics['Pearson_Correlation'] = pearsonr(mean_true, mean_pred)[0]
            metrics['Pearson_Correlation_delta'] = pearsonr(mean_true - mean_ctrl, mean_pred - mean_ctrl)[0]
        except:
            metrics['Pearson_Correlation'] = np.nan
            metrics['Pearson_Correlation_delta'] = np.nan

    metrics.update({
        'MSE': mean_squared_error(mean_true, mean_pred),
        'RMSE': np.sqrt(mean_squared_error(mean_true, mean_pred)),
        'MAE': mean_absolute_error(mean_true, mean_pred),
        'Cosine_Similarity': cosine_similarity([mean_true], [mean_pred])[0, 0],
        'L2': np.linalg.norm(mean_true - mean_pred),
        'MSE_delta': mean_squared_error(mean_true - mean_ctrl, mean_pred - mean_ctrl),
        'RMSE_delta': np.sqrt(mean_squared_error(mean_true - mean_ctrl, mean_pred - mean_ctrl)),
        'MAE_delta': mean_absolute_error(mean_true - mean_ctrl, mean_pred - mean_ctrl),
        'Cosine_Similarity_delta': cosine_similarity([(mean_true - mean_ctrl)], [(mean_pred - mean_ctrl)])[0, 0],
        'L2_delta': np.linalg.norm((mean_true - mean_ctrl) - (mean_pred - mean_ctrl)),
    })

    # Compute MMD and Wasserstein per gene
    mmd_list, ws_list = [], []
    for g in range(pred_data.shape[1]):
        pred_gene = pred_data[:, g].reshape(-1, 1)
        true_gene = true_data[:, g].reshape(-1, 1)
        mmd_list.append(compute_mmd(pred_gene, true_gene))
        ws_list.append(compute_wasserstein(pred_gene.flatten(), true_gene.flatten()))
    metrics['MMD'] = np.mean(mmd_list)
    metrics['Wasserstein'] = np.mean(ws_list)

    return metrics

squidiff_dir = "/home/yunlin/projects/perturb_model/lanxiang_model/new_models/squidiff_model/Squidiff_reproducibility/squidiff_results/predictions"
raw_data_dir = "/data2/lanxiang/data/Task3_data"
deg_root = '/home/yunlin/projects/perturb_model/lanxiang_model/model_check/de_method_comparison/DEG_abs_score_sorted'
output_root = "/data2/yunlin/projects/perturb_model/lanxiang_model/metrics_calculation/task3/squidiff_metrics"

output_paths = {
    "allgene": os.path.join(output_root, "Allgene", "squidiff_all_metrics.csv"),
    "top20": os.path.join(output_root, "Top20", "squidiff_top20_metrics.csv"),
    "top50": os.path.join(output_root, "Top50", "squidiff_top50_metrics.csv"),
    "top100": os.path.join(output_root, "Top100", "squidiff_top100_metrics.csv")
}

for path in output_paths.values():
    os.makedirs(os.path.dirname(path), exist_ok=True)

all_results = {
    "allgene": [],
    "top20": [],
    "top50": [],
    "top100": []
}


for dataset_folder in os.listdir(squidiff_dir):
    dataset_path = os.path.join(squidiff_dir, dataset_folder)
    if not os.path.isdir(dataset_path):
        continue
    

    dataset_name = dataset_folder
    
    raw_file = os.path.join(raw_data_dir, f"{dataset_name}.h5ad")
    if not os.path.exists(raw_file):
        print(f"❌ Missing raw data: {raw_file}")
        continue


    adata = sc.read_h5ad(raw_file)
    print(f"▶ [Squidiff] Processing {dataset_name}")


    for condition in os.listdir(dataset_path):
        cond_path = os.path.join(dataset_path, condition)
        if not os.path.isdir(cond_path):
            continue


        for file in os.listdir(cond_path):
            if not file.endswith("_prediction.h5ad"):
                continue


            cell_type = file.replace("_prediction.h5ad", "")
            pred_file = os.path.join(cond_path, file)

            try:
                print(f"  Processing: {condition}/{cell_type}")
                
                cond_mask = (adata.obs['condition'] == condition) & (adata.obs['cell_type'] == cell_type)
                ctrl_mask = (adata.obs['condition'] == 'ctrl') & (adata.obs['cell_type'] == cell_type)
                
                if not cond_mask.any():
                    print(f"⚠️  No cells found for condition={condition}, cell_type={cell_type}")
                    continue
                if not ctrl_mask.any():
                    print(f"⚠️  No ctrl cells found for cell_type={cell_type}")
                    continue

                pred_adata = sc.read_h5ad(pred_file)
                pred_data = pred_adata.X
                

                pred_genes = pred_adata.var_names.tolist()

                if not all(gene in adata.var_names for gene in pred_genes):
                    print(f"❌ Some prediction genes not found in raw data")
                    continue
                

                gene_indices = [adata.var_names.get_loc(gene) for gene in pred_genes]
                
                true_data = adata[cond_mask][:, gene_indices].X
                ctrl_data = adata[ctrl_mask][:, gene_indices].X

                print(f"    Data shapes - Pred: {pred_data.shape}, True: {true_data.shape}, Ctrl: {ctrl_data.shape}")
                print(f"    Using {len(pred_genes)} genes from prediction")

                min_n = min(pred_data.shape[0], true_data.shape[0], ctrl_data.shape[0])
                n_sample = min(500, min_n)
                
                if min_n > 0:
                    idx_pred = np.random.choice(pred_data.shape[0], n_sample, replace=False)
                    idx_true = np.random.choice(true_data.shape[0], n_sample, replace=False)
                    idx_ctrl = np.random.choice(ctrl_data.shape[0], n_sample, replace=False)
                    
                    pred_data_sampled = pred_data[idx_pred]
                    true_data_sampled = true_data[idx_true]
                    ctrl_data_sampled = ctrl_data[idx_ctrl]
                else:
                    print(f"❌ No valid data found for {dataset_name}/{condition}/{cell_type}")
                    continue

                metrics = calculate_metrics_all(pred_data_sampled, true_data_sampled, ctrl_data_sampled)
                all_results["allgene"].append({
                    "dataset": dataset_name,
                    "condition": condition,
                    "cell_type": cell_type,
                    "model": "Squidiff",
                    **metrics
                })

                for top_n in [20, 50, 100]:
                    deg_file = os.path.join(deg_root, dataset_name, cell_type, f"{condition}.csv")
                    if not os.path.exists(deg_file):
                        print(f"    ❌ Missing DEG file: {deg_file}")
                        continue

                    deg_df = pd.read_csv(deg_file)
                    if len(deg_df) < top_n:
                        print(f"    ⚠️  DEG file has only {len(deg_df)} genes, less than top{top_n}")
                        continue
                        
                    top_degs = deg_df.head(top_n)["names"].tolist()

                    valid_degs = [g for g in top_degs if g in pred_genes]
                    if len(valid_degs) == 0:
                        print(f"    ❌ No DEGs from top{top_n} found in prediction genes")
                        continue
                    
                    if len(valid_degs) < top_n:
                        print(f"    ⚠️  Only {len(valid_degs)} out of {top_n} DEGs found in prediction genes")

                    pred_gene_indices = [pred_genes.index(g) for g in valid_degs]
                    
                    top_pred_data = pred_data_sampled[:, pred_gene_indices]
                    top_true_data = true_data_sampled[:, pred_gene_indices]
                    top_ctrl_data = ctrl_data_sampled[:, pred_gene_indices]

                    top_metrics = calculate_metrics_all(top_pred_data, top_true_data, top_ctrl_data)
                    all_results[f"top{top_n}"].append({
                        "dataset": dataset_name,
                        "condition": condition,
                        "cell_type": cell_type,
                        "model": "Squidiff",
                        "n_genes_used": len(valid_degs),
                        **top_metrics
                    })

                print(f"    ✅ Processed: {dataset_name}/{condition}/{cell_type}")

            except Exception as e:
                print(f"❌ Failed on {pred_file}: {e}")
                import traceback
                traceback.print_exc()

for key, result in all_results.items():
    if len(result) > 0:
        df = pd.DataFrame(result)
        df.to_csv(output_paths[key], index=False)
        print(f"✅ {key} metrics saved to: {output_paths[key]} ({len(result)} records)")
    else:
        print(f"⚠️  No results for {key}")

print("\n🎉 All processing completed!")