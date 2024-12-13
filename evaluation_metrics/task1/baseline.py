import os
import pandas as pd
import scanpy as sc
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance, pearsonr
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

results_df = pd.DataFrame(columns=[
    'dataset', 'condition', 'cell_type', 'model',
    'R_squared', 'R_squared_delta',
    'Pearson_Correlation', 'Pearson_Correlation_delta',
    'MSE', 'MSE_delta',
    'RMSE', 'RMSE_delta',
    'MAE', 'MAE_delta',
    'Cosine_Similarity', 'Cosine_Similarity_delta',
    'L2', 'L2_delta',
    'Wasserstein', 'MMD'
])

base_path = '/data2/yue_data/pert/baseline/task1/'
datasets = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
datasets

results_list = []

for dataset in datasets:
    print(dataset)
    dataset_path = os.path.join(base_path, dataset)
    

    ctrl_path = os.path.join('/data2/lanxiang/perturb_benchmarking/Task_pred_out/Ctrl_csv', dataset, 'ctrl_data.csv')
    if not os.path.exists(ctrl_path):
        print(f"Control file '{ctrl_path}' not found!")
        continue
    
    ctrl_data = pd.read_csv(ctrl_path, index_col=0)
    ctrl_data = np.nan_to_num(ctrl_data.values)  
    
    for pred_file in os.listdir(dataset_path):
        if pred_file.endswith("_predicted_values.npy"):
            true_file = pred_file.replace("_predicted_values.npy", "_true_values.npy")
            pred_file_path = os.path.join(dataset_path, pred_file)
            true_file_path = os.path.join(dataset_path, true_file)
            
            if os.path.exists(true_file_path):
                print(true_file_path)
                
                
                pred_data = np.load(pred_file_path)
                true_data = np.load(true_file_path)
                
                pred_data = np.nan_to_num(pred_data) 
                true_data = np.nan_to_num(true_data)  
                
            
                if pred_data.shape[0] > 300:
                    pred_data = pred_data[np.random.choice(pred_data.shape[0], 300, replace=False), :]
                if true_data.shape[0] > 300:
                    true_data = true_data[np.random.choice(true_data.shape[0], 300, replace=False), :]
                if ctrl_data.shape[0] > 300:
                    ctrl_data = ctrl_data[np.random.choice(ctrl_data.shape[0], 300, replace=False), :]
                
        
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

       
                condition_name = pred_file.replace("_ctrl_predicted_values.npy", "")
                condition_name = condition_name.replace("_predicted_values.npy", "")
                condition_name = condition_name.replace("_true_values.npy", "")
                condition_name = condition_name.replace("_", "+")
             
                results_list.append({
                    'dataset': dataset,
                    'condition': condition_name, 
                    'model': "Baseline", 
                    **metrics
                })


results_df = pd.DataFrame(results_list)

results_df.to_csv("/task1/baseline_results.csv", index=False)

