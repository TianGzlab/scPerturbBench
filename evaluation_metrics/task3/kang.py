import scanpy as sc
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from warnings import catch_warnings, simplefilter


np.random.seed(42)
def compute_mmd(pred_data, true_data, kernel='rbf', gamma=1.0):
    if kernel == 'rbf':
        dist_pred = cdist(pred_data, pred_data, metric='sqeuclidean')
        dist_truth = cdist(true_data, true_data, metric='sqeuclidean')
        dist_cross = cdist(pred_data, true_data, metric='sqeuclidean')

        Kxx = np.exp(-gamma * dist_pred)
        Kyy = np.exp(-gamma * dist_truth)
        Kxy = np.exp(-gamma * dist_cross)

        return np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)
    else:
        raise ValueError("Unsupported kernel type. Use 'rbf'.")

def compute_wasserstein(pred_data, true_data):
    return wasserstein_distance(pred_data.flatten(), true_data.flatten())

    
def calculate_metrics_all(pred_data, true_data, ctrl_data):
    if isinstance(true_data, pd.DataFrame):
        true_data = true_data.values
    if isinstance(pred_data, pd.DataFrame):
        pred_data = pred_data.values
    if isinstance(ctrl_data, pd.DataFrame):
        ctrl_data = ctrl_data.values

    metrics = {}

    mean_true = np.mean(true_data, axis=0)
    mean_pred = np.mean(pred_data, axis=0)
    mean_ctrl = np.mean(ctrl_data, axis=0)

    with catch_warnings():
        simplefilter("ignore")
        try:
            metrics['R_squared'] = r2_score(mean_true, mean_pred)
            metrics['R_squared_delta'] = r2_score(mean_true - mean_ctrl, mean_pred - mean_ctrl)
        except Exception:
            metrics['R_squared'] = np.nan
            metrics['R_squared_delta'] = np.nan

        try:
            corr, _ = pearsonr(mean_true, mean_pred)
            metrics['Pearson_Correlation'] = corr
            corr_delta, _ = pearsonr(mean_true - mean_ctrl, mean_pred - mean_ctrl)
            metrics['Pearson_Correlation_delta'] = corr_delta
        except Exception:
            metrics['Pearson_Correlation'] = np.nan
            metrics['Pearson_Correlation_delta'] = np.nan

    mse = mean_squared_error(mean_true, mean_pred)
    mse_delta = mean_squared_error(mean_true - mean_ctrl, mean_pred - mean_ctrl)
    metrics.update({
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(mean_true, mean_pred),
        'Cosine_Similarity': cosine_similarity([mean_true], [mean_pred])[0, 0],
        'L2': np.linalg.norm(mean_true - mean_pred),
        'MSE_delta': mse_delta,
        'RMSE_delta': np.sqrt(mse_delta),
        'MAE_delta': mean_absolute_error(mean_true - mean_ctrl, mean_pred - mean_ctrl),
        'Cosine_Similarity_delta': cosine_similarity(
            [(mean_true - mean_ctrl)], [(mean_pred - mean_ctrl)]
        )[0, 0],
        'L2_delta': np.linalg.norm((mean_true - mean_ctrl) - (mean_pred - mean_ctrl)),
    })

    return metrics

datasets = {
    'Kang': {
        'path': "/data2/lanxiang/data/Task3_data/Kang.h5ad",
        'condition': lambda obs, cell_type: (obs['condition'] == 'stimulated') & (obs['cell_type'] == cell_type),
        'control': lambda obs, cell_type: (obs['condition'] == 'control') & (obs['cell_type'] == cell_type),
        'model_paths': {
            'scgen': "/data2/lanxiang/perturb_benchmarking/test_model_data/scgen/Kang_pred_data",
            'trvae': "/data2/lanxiang/perturb_benchmarking/test_model_data/trvae_new_all_gene/Kang_adata_pred",
            'scpregan': "/data2/lanxiang/perturb_benchmarking/test_model_data/scpregan/run_data/Kang/pred_adata"
        }
    },
}

kang_results = {
    'dataset': [],
    'condition': [],
    'cell_type': [],
    'model': [],
    'R_squared': [],
    'R_squared_delta': [],
    'Pearson_Correlation': [],
    'Pearson_Correlation_delta': [],
    'MSE': [],
    'MSE_delta': [],
    'RMSE': [],
    'RMSE_delta': [],
    'MAE': [],
    'MAE_delta': [],
    'Cosine_Similarity': [],
    'Cosine_Similarity_delta': [],
    'L2': [],
    'L2_delta': []
}

def generate_file_name(cell_type, model):
    if model == 'scgen':
        return f"pred_adata_{cell_type}.h5ad"
    elif model == 'trvae':
        return f"{cell_type}_pred.h5ad"
    elif model == 'scpregan':
        return f"pred_{cell_type}.h5ad"
    else:
        return None

for dataset_name, dataset_info in datasets.items():
    adata = sc.read(dataset_info['path'])
    if dataset_name == 'Hagai':
        all_cell_types = adata.obs['species'].unique()
    else:
        all_cell_types = adata.obs['cell_type'].unique()

    for cell_type in all_cell_types:
        cond_mask = dataset_info['condition'](adata.obs, cell_type)
        ctrl_mask = dataset_info['control'](adata.obs, cell_type)

        true_data = adata[cond_mask, :].X.toarray()
        ctrl_data = adata[ctrl_mask, :].X.toarray()

        # Noperturb
        noperturb_metrics = calculate_metrics_all(ctrl_data, true_data, ctrl_data)

        kang_results['dataset'].append(dataset_name)
        kang_results['condition'].append("IFN-β-stimulated")
        kang_results['cell_type'].append(cell_type)
        kang_results['model'].append("Noperturb")
        for metric, value in noperturb_metrics.items():
            kang_results[metric].append(value)

        for model in dataset_info['model_paths'].keys():
            file_name = generate_file_name(cell_type, model)
            pred_file_path = os.path.join(dataset_info['model_paths'][model], file_name)

            if os.path.exists(pred_file_path):
                pred_data = sc.read(pred_file_path).X

                metrics = calculate_metrics_all(pred_data, true_data, ctrl_data)
                mmd_per_gene = []
                for gene_idx in range(pred_data.shape[1]):
                    gene_pred = pred_data[:, gene_idx].reshape(-1, 1)
                    gene_truth = true_data[:, gene_idx].reshape(-1, 1)
                    mmd = compute_mmd(gene_pred, gene_truth, kernel='rbf', gamma=1.0)
                    mmd_per_gene.append(mmd)
                
                average_mmd = np.mean(mmd_per_gene)

    
                ws_per_gene = []
                for gene_idx in range(pred_data.shape[1]):
                    gene_pred = pred_data[:, gene_idx].reshape(-1, 1)
                    gene_truth = true_data[:, gene_idx].reshape(-1, 1)
                    ws = compute_wasserstein(gene_pred, gene_truth)
                    ws_per_gene.append(ws)
                
                average_ws = np.mean(ws_per_gene)
          
                metrics['MMD'] = average_mmd
                metrics['Wasserstein'] = average_ws

                kang_results['dataset'].append(dataset_name)
                kang_results['condition'].append("IFN-β-stimulated")
                kang_results['cell_type'].append(cell_type)
                kang_results['model'].append(model)
                for metric, value in metrics.items():
                    kang_results[metric].append(value)
            else:
                print(f"file {pred_file_path} not exist")

kang_results_df = pd.DataFrame(kang_results)
kang_results_df.to_csv("/task3/kang_metrics_results.csv", index=False)
