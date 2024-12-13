import pandas as pd
import numpy as np
import os
import scanpy as sc
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance, pearsonr
from warnings import catch_warnings, simplefilter


np.random.seed(42)

# MMD
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


# Wasserstein
def compute_wasserstein(pred_data, true_data):
    return wasserstein_distance(pred_data.flatten(), true_data.flatten())


# metrics
def calculate_metrics_all(pred_data, true_data, ctrl_data):
    # Make sure the input is a numpy array
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
    'dataset', 'condition', 'model',
    'R_squared', 'R_squared_delta',
    'Pearson_Correlation', 'Pearson_Correlation_delta',
    'MSE', 'MSE_delta',
    'RMSE', 'RMSE_delta',
    'MAE', 'MAE_delta',
    'Cosine_Similarity', 'Cosine_Similarity_delta',
    'L2', 'L2_delta',
    'MMD','Wasserstein'
])


datasets = [
    'datlingerbock2017',
    'datlingerbock2021',
    'dixitregev2016',
    'frangiehizar2021_rna',
    "normanweissman2019_filtered",
    'papalexisatija2021_eccite_rna',
    'replogleweissman2022_rpe1',
    'sunshine2023_crispri_sarscov2',
    "tiankampmann2021_crispra",
    "tiankampmann2021_crispri"
]

scelmo_result = {}


for dataset in datasets:
    scelmo_pred_path = f'/data1/yy_data/pert/scelmo/save2/dev_perturb_{dataset}/results'
    true_path = f'/data2/lanxiang/perturb_benchmarking/Task_pred_out/True_csv/{dataset}'
    ctrl_path = f'/data2/lanxiang/perturb_benchmarking/Task_pred_out/Ctrl_csv/{dataset}/ctrl_data.csv'

    ctrl_data = pd.read_csv(ctrl_path, index_col=0)

    dataset_results = []

    for file_name in os.listdir(scelmo_pred_path):
        if file_name.endswith('.npz') and '_' not in file_name:

            npz_file_path = os.path.join(scelmo_pred_path, file_name)
            with np.load(npz_file_path) as npz:
                if 'pred' in npz:
                    pred_data = npz['pred']
                    pred_data = pd.DataFrame(pred_data)  # 转为 DataFrame

                  
                    true_file_path = os.path.join(true_path, file_name.replace('.npz', '.csv'))
                    true_data = pd.read_csv(true_file_path, index_col=0)
                    
                  
                    if pred_data.shape[0] > 300:
                        pred_data = pred_data.iloc[np.random.choice(pred_data.shape[0], 300, replace=False), :]
                    if true_data.shape[0] > 300:
                        true_data = true_data.iloc[np.random.choice(true_data.shape[0], 300, replace=False), :]
                    if ctrl_data.shape[0] > 300:
                        ctrl_data = ctrl_data.iloc[np.random.choice(ctrl_data.shape[0], 300, replace=False), :]

                    
                    metrics = calculate_metrics_all(pred_data, true_data, ctrl_data)
                    mmd_per_gene = []
                    for gene_idx in range(pred_data.shape[1]):
                        gene_pred = pred_data.iloc[:, gene_idx].values.reshape(-1, 1)
                        gene_truth = true_data.iloc[:, gene_idx].values.reshape(-1, 1)
                        mmd = compute_mmd(gene_pred, gene_truth, kernel='rbf', gamma=1.0)
                        mmd_per_gene.append(mmd)
                
                    average_mmd = np.mean(mmd_per_gene)

                
                    ws_per_gene = []
                    for gene_idx in range(pred_data.shape[1]):
                        gene_pred = pred_data.iloc[:, gene_idx].values.reshape(-1, 1)
                        gene_truth = true_data.iloc[:, gene_idx].values.reshape(-1, 1)
                        ws = compute_wasserstein(gene_pred, gene_truth)
                        ws_per_gene.append(ws)
                
                    average_ws = np.mean(ws_per_gene)
                    metrics['MMD'] = average_mmd
                    metrics['Wasserstein'] = average_ws
                    metrics['condition'] = file_name.replace('.npz', '')
                    metrics['dataset'] = dataset
                    
                    metrics["model"]= "scELMO"

                    results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)


print(results_df)
results_df

results_df.to_csv('/task1/scelmo_results.csv', index=False)
