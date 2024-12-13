import os
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, pearsonr
from warnings import catch_warnings, simplefilter
import scipy.sparse

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


datasets = {
    'Kang': {
        'path': "/data2/lanxiang/data/Task3_data/Kang.h5ad",
        'condition': lambda obs, cell_type: (obs['condition'] == 'stimulated') & (obs['cell_type'] == cell_type),
        'control_condition': lambda obs, cell_type: (obs['condition'] == 'control') & (obs['cell_type'] == cell_type)
    },
    'Haber': {
        'path': "/data2/lanxiang/data/Task3_data/Haber.h5ad",
        'condition': lambda obs, cell_type: (obs['condition'] == 'Hpoly.Day10') & (obs['cell_type'] == cell_type),
        'control_condition': lambda obs, cell_type: (obs['condition'] == 'Control') & (obs['cell_type'] == cell_type)
    },
    'Hagai': {
        'path': "/data2/lanxiang/data/Task3_data/Hagai.h5ad",
        'condition': lambda obs, cell_type: (obs['condition'] == 'LPS6') & (obs['species'] == cell_type),
        'control_condition': lambda obs, cell_type: (obs['condition'] == 'unst') & (obs['species'] == cell_type)
    },
    'Weinreb': {
        'path': "/data2/lanxiang/data/Task3_data/Weinreb_time.h5ad",
        'condition': lambda obs, cell_type: (obs['condition'] == 'developed') & (obs['cell_type'] == cell_type),
        'control_condition': lambda obs, cell_type: (obs['condition'] == 'control') & (obs['cell_type'] == cell_type)
    },
    'Srivatsan': {
        'path': "/data2/lanxiang/data/Task3_data/SrivatsanTrapnell2020_sciplex3_sub.h5ad",
        'condition': lambda obs, cell_type, condition: (obs['condition'] == condition) & (obs['cell_type'] == cell_type),
        'control_condition': lambda obs, cell_type: (obs['condition'] == 'ctrl') & (obs['cell_type'] == cell_type)
    },
    'Open_Problem': {
        'path': "/data2/lanxiang/data/Task3_data/Open_Problems_sub.h5ad",
        'condition': lambda obs, cell_type, condition: (obs['condition'] == condition) & (obs['cell_type'] == cell_type),
        'control_condition': lambda obs, cell_type: (obs['condition'] == 'ctrl') & (obs['cell_type'] == cell_type)
    },
    'This_study': {
        'path': "/data2/lanxiang/data/Task3_data/This_study_sub.h5ad",
        'condition': lambda obs, cell_type, condition: (obs['condition'] == condition) & (obs['cell_type'] == cell_type),
        'control_condition': lambda obs, cell_type: (obs['condition'] == 'Control') & (obs['cell_type'] == cell_type)
    }
}

def calculate_metrics_all(pred_data, true_data, ctrl_data):
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
        'Cosine_Similarity_delta': cosine_similarity([(mean_true - mean_ctrl)], [(mean_pred - mean_ctrl)])[0, 0],
        'L2_delta': np.linalg.norm((mean_true - mean_ctrl) - (mean_pred - mean_ctrl))
    })

    return metrics

results_list = []
base_path = '/data2/lanxiang/perturb_benchmarking/12.60_new_metrics/task3/task3'
dataset_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]


for dataset_name, dataset_info in datasets.items():
    dataset_path = dataset_info['path']
    condition_func = dataset_info['condition']
    control_condition_func = dataset_info['control_condition']

    adata = sc.read_h5ad(dataset_path)
    obs = adata.obs
    X = adata.X


    if dataset_name == "Hagai":
        cell_types = adata.obs['species'].unique()
    else:
        cell_types = adata.obs['cell_type'].unique()


    for cell_type in cell_types:
   
        control_condition = control_condition_func(obs, cell_type)
        control_data = adata[control_condition, :]
        control_counts = control_data.X.toarray()

   
        if control_counts.shape[0] == 0:
            print(f"No control data for dataset '{dataset_name}', cell type '{cell_type}'")
            continue

        dataset_folder = os.path.join(base_path, dataset_name)
        cell_type_folder = os.path.join(dataset_folder, cell_type)

        if not os.path.exists(cell_type_folder):
            continue

        for pred_file in os.listdir(cell_type_folder):
            if pred_file.endswith("_predicted_values.npy"):
                pred_file_path = os.path.join(cell_type_folder, pred_file)
                true_file_path = pred_file_path.replace("_predicted_values.npy", "_true_values.npy")

          
                if not os.path.exists(true_file_path):
                    continue

          
                pred_data = np.load(pred_file_path, allow_pickle=True)
                true_data = np.load(true_file_path, allow_pickle=True)

         
                pred_data = pred_data.toarray() if scipy.sparse.issparse(pred_data) else pred_data
                true_data = true_data.toarray() if scipy.sparse.issparse(true_data) else true_data

         
                metrics = calculate_metrics_all(pred_data, true_data, control_counts)
             
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

             
                condition_name = pred_file.replace("_ctrl_predicted_values.npy", "")
                condition_name = condition_name.replace("_predicted_values.npy", "")
                condition_name = condition_name.replace("_true_values.npy", "")

   
                results_list.append({
                    'dataset': dataset_name,
                    'condition': condition_name,
                    'cell_type': cell_type,
                    'model': 'Baseline',
                    **metrics
                })


results_df = pd.DataFrame(results_list)
results_df.to_csv("/task3/baseline_result.csv", index=False)

