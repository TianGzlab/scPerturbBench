import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from sklearn.linear_model import LinearRegression


class Task3ContextMeanBaseline:
    def __init__(self):
        self.celltype_context_means = {}
    
    def fit(self, adata, target_celltype, available_conditions):
        celltype_data = []

        for condition in available_conditions:
            condition_celltype_data = adata.X[
                (adata.obs['cell_type'] == target_celltype) & 
                (adata.obs['condition'] == condition)
            ]
            if condition_celltype_data.shape[0] > 0:
                if scipy.sparse.issparse(condition_celltype_data):
                    condition_celltype_data = condition_celltype_data.toarray()
                celltype_data.append(condition_celltype_data)
        
        if celltype_data:
            all_data = np.vstack(celltype_data)
            self.celltype_context_means[target_celltype] = np.mean(all_data, axis=0)
        else:
            print(f"Warning: No data found for celltype {target_celltype}")
            self.celltype_context_means[target_celltype] = np.zeros(adata.shape[1])
    
    def predict(self, target_celltype, target_condition, sample_size):

        if target_celltype in self.celltype_context_means:
            context_mean = self.celltype_context_means[target_celltype]
            return np.tile(context_mean, (sample_size, 1))
        else:
            print(f"Warning: No context mean found for celltype {target_celltype}")
            return np.zeros((sample_size, adata.shape[1]))


class Task3PerturbationMeanBaseline:
    
    def __init__(self):
        self.celltype_ctrl_means = {}
        self.condition_effects = {}
    
    def fit(self, adata, target_celltype, target_condition, other_celltypes):
        target_ctrl_data = adata.X[
            (adata.obs['cell_type'] == target_celltype) & 
            (adata.obs['condition'] == 'ctrl')
        ]
        if target_ctrl_data.shape[0] > 0:
            if scipy.sparse.issparse(target_ctrl_data):
                target_ctrl_data = target_ctrl_data.toarray()
            self.celltype_ctrl_means[target_celltype] = np.mean(target_ctrl_data, axis=0)
        else:
            print(f"Warning: No ctrl data found for celltype {target_celltype}")
            self.celltype_ctrl_means[target_celltype] = np.zeros(adata.shape[1])
        
        condition_effects = []
        for other_celltype in other_celltypes:

            other_ctrl_data = adata.X[
                (adata.obs['cell_type'] == other_celltype) & 
                (adata.obs['condition'] == 'ctrl')
            ]
            if other_ctrl_data.shape[0] > 0:
                if scipy.sparse.issparse(other_ctrl_data):
                    other_ctrl_data = other_ctrl_data.toarray()
                other_ctrl_mean = np.mean(other_ctrl_data, axis=0)
                
                other_condition_data = adata.X[
                    (adata.obs['cell_type'] == other_celltype) & 
                    (adata.obs['condition'] == target_condition)
                ]
                if other_condition_data.shape[0] > 0:
                    if scipy.sparse.issparse(other_condition_data):
                        other_condition_data = other_condition_data.toarray()
                    other_condition_mean = np.mean(other_condition_data, axis=0)
                    
                    effect = other_condition_mean - other_ctrl_mean
                    condition_effects.append(effect)

        if condition_effects:
            self.condition_effects[target_condition] = np.mean(condition_effects, axis=0)
        else:
            print(f"Warning: No condition effects found for condition {target_condition}")
            self.condition_effects[target_condition] = np.zeros(adata.shape[1])
    
    def predict(self, target_celltype, target_condition, sample_size):
        if (target_celltype in self.celltype_ctrl_means and 
            target_condition in self.condition_effects):
            
            predicted_expr = (self.celltype_ctrl_means[target_celltype] + 
                             self.condition_effects[target_condition])
            return np.tile(predicted_expr, (sample_size, 1))
        else:
            print(f"Warning: Missing data for {target_celltype} or {target_condition}")
            return np.zeros((sample_size, len(self.celltype_ctrl_means.get(target_celltype, []))))


class Task3LinearBaseline:

    
    def __init__(self, gene_names, cell_types, conditions):
        self.gene_names = gene_names
        self.cell_types = list(cell_types)
        self.conditions = list(conditions)
        self.model = LinearRegression() 
        
        self.celltype_map = {ct: i for i, ct in enumerate(self.cell_types)}
        self.condition_map = {cond: i for i, cond in enumerate(self.conditions)}
    
    def _encode_celltype_condition(self, celltype, condition):
        celltype_onehot = np.zeros(len(self.cell_types))
        condition_onehot = np.zeros(len(self.conditions))
        
        if celltype in self.celltype_map:
            celltype_onehot[self.celltype_map[celltype]] = 1
        if condition in self.condition_map:
            condition_onehot[self.condition_map[condition]] = 1
        
        return np.concatenate([celltype_onehot, condition_onehot])
    
    def fit(self, adata, excluded_celltype=None, excluded_condition=None):
        X_train = []
        y_train = []
        
        for celltype in self.cell_types:
            for condition in self.conditions:
                if (celltype == excluded_celltype and condition == excluded_condition):
                    continue
                
                mask = ((adata.obs['cell_type'] == celltype) & 
                       (adata.obs['condition'] == condition))
                condition_data = adata.X[mask]
                
                if condition_data.shape[0] > 0:
                    if scipy.sparse.issparse(condition_data):
                        condition_data = condition_data.toarray()

                    encoding = self._encode_celltype_condition(celltype, condition)
                    
                    for sample in condition_data:
                        X_train.append(encoding)
                        y_train.append(sample)
        
        if len(X_train) == 0:
            print("Warning: No training data found")
            return
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"Training multi-output linear regression for {len(self.gene_names)} genes...")
        self.model.fit(X_train, y_train)
        print("Linear regression training completed!")
    
    def predict(self, target_celltype, target_condition, sample_size):
        encoding = self._encode_celltype_condition(target_celltype, target_condition)

        predicted_expr = self.model.predict([encoding])[0]

        return np.tile(predicted_expr, (sample_size, 1))


def run_task3_baselines_auto(dataset_paths):
    
    for data_path in dataset_paths:
        print(f"Processing dataset: {data_path}")
        
        adata = sc.read(data_path)

        if 'cell_type' not in adata.obs.columns:
            if 'species' in adata.obs.columns:
                adata.obs.rename(columns={'species': 'cell_type'}, inplace=True)
            elif 'celltype' in adata.obs.columns:
                adata.obs.rename(columns={'celltype': 'cell_type'}, inplace=True)

        if 'cell_type' not in adata.obs or 'condition' not in adata.obs:
            print(f"Dataset {data_path} is missing 'cell_type' or 'condition' columns. Skipping this dataset.")
            continue

        adata.obs['condition'] = adata.obs['condition'].replace({'control': 'ctrl', 'Control': 'ctrl', 'unst': 'ctrl'})

        cell_types = adata.obs['cell_type'].unique() 
        conditions = adata.obs['condition'].unique()
        conditions = [cond for cond in conditions if cond != 'ctrl']
        gene_names = adata.var_names 

        dataset_name = os.path.basename(data_path).split('.')[0]
        print(f"Dataset: {dataset_name}")
        print(f"Cell types: {list(cell_types)}")
        print(f"Conditions: {list(conditions)}")

        for condition in conditions:
            print(f"Processing condition: {condition}")

            for target_cell in cell_types:
                print(f"  Processing target cell: {target_cell}")

                target_ctrl_count = np.sum(
                    (adata.obs['cell_type'] == target_cell) & 
                    (adata.obs['condition'] == 'ctrl')
                )
                target_condition_count = np.sum(
                    (adata.obs['cell_type'] == target_cell) & 
                    (adata.obs['condition'] == condition)
                )
                
                select_num = min(target_ctrl_count, target_condition_count)
                
                if select_num == 0:
                    print(f"    Warning: No data found for {target_cell} + {condition}. Skipping.")
                    continue

                other_celltypes = [ct for ct in cell_types if ct != target_cell]

                available_conditions = [cond for cond in conditions if cond != condition]
                available_conditions.append('ctrl')
                

                target_cell_data_true = adata.X[
                    (adata.obs['cell_type'] == target_cell) & 
                    (adata.obs['condition'] == condition)
                ]
                if scipy.sparse.issparse(target_cell_data_true):
                    target_cell_data_true = target_cell_data_true.toarray()
                
                if target_cell_data_true.shape[0] > select_num:
                    indices_true = np.random.choice(target_cell_data_true.shape[0], select_num, replace=False)
                    target_cell_data_true = target_cell_data_true[indices_true]
                elif target_cell_data_true.shape[0] < select_num:
                    indices_true = np.random.choice(target_cell_data_true.shape[0], select_num, replace=True)
                    target_cell_data_true = target_cell_data_true[indices_true]
 
                baselines = {}
                
                # Context Mean Baseline
                context_mean_baseline = Task3ContextMeanBaseline()
                context_mean_baseline.fit(adata, target_cell, available_conditions)
                baselines['context_mean'] = context_mean_baseline
                
                # Perturbation Mean Baseline
                perturbation_mean_baseline = Task3PerturbationMeanBaseline()
                perturbation_mean_baseline.fit(adata, target_cell, condition, other_celltypes)
                baselines['perturbation_mean'] = perturbation_mean_baseline
                
                # Linear Regression Baseline
                linear_baseline = Task3LinearBaseline(gene_names, cell_types, list(conditions) + ['ctrl'])
                linear_baseline.fit(adata, target_cell, condition)
                baselines['linear_regression'] = linear_baseline
                
                for baseline_name, baseline_model in baselines.items():
                    if baseline_name == 'linear_regression':
                        predicted_data = baseline_model.predict(target_cell, condition, select_num)
                    else:
                        predicted_data = baseline_model.predict(target_cell, condition, select_num)

                    save_dir = f"/data2/lanxiang/perturb_benchmark_v2/model/Baseline_review/Task3/{baseline_name}/{dataset_name}/{target_cell}"
                    os.makedirs(save_dir, exist_ok=True)

                    true_file = os.path.join(save_dir, f"{condition}_true_values.npy")
                    pred_file = os.path.join(save_dir, f"{condition}_predicted_values.npy")
                    
                    np.save(true_file, target_cell_data_true)
                    np.save(pred_file, predicted_data)
                    
                    print(f"    {baseline_name} - True: {true_file}")
                    print(f"    {baseline_name} - Pred: {pred_file}")
                
                for baseline_name in baselines.keys():
                    save_dir = f"/data2/lanxiang/perturb_benchmark_v2/model/Baseline_review/Task3/{baseline_name}/{dataset_name}/{target_cell}"
                    gene_names_file = os.path.join(save_dir, "gene_names.npy")
                    np.save(gene_names_file, gene_names)


def run_task3_baselines(data_path, target_celltype, target_condition, output_base_path):
    print(f"Running Task3 baselines for celltype={target_celltype}, condition={target_condition}")
    
    adata = sc.read(data_path)
    
    if 'cell_type' not in adata.obs.columns:
        if 'species' in adata.obs.columns:
            adata.obs.rename(columns={'species': 'cell_type'}, inplace=True)
        elif 'celltype' in adata.obs.columns:
            adata.obs.rename(columns={'celltype': 'cell_type'}, inplace=True)

    if 'cell_type' not in adata.obs or 'condition' not in adata.obs:
        print(f"Dataset {data_path} is missing 'cell_type' or 'condition' columns.")
        return

    adata.obs['condition'] = adata.obs['condition'].replace({
        'control': 'ctrl', 'Control': 'ctrl', 'unst': 'ctrl'
    })

    gene_names = adata.var_names
    cell_types = adata.obs['cell_type'].unique()
    conditions = adata.obs['condition'].unique()
    other_celltypes = [ct for ct in cell_types if ct != target_celltype]
    available_conditions = [cond for cond in conditions if cond != target_condition]

    target_ctrl_count = np.sum(
        (adata.obs['cell_type'] == target_celltype) & 
        (adata.obs['condition'] == 'ctrl')
    )
    target_condition_count = np.sum(
        (adata.obs['cell_type'] == target_celltype) & 
        (adata.obs['condition'] == target_condition)
    )
    sample_size = min(target_ctrl_count, target_condition_count)
    
    if sample_size == 0:
        print(f"Warning: No data found for {target_celltype} + {target_condition}")
        return

    true_data = adata.X[
        (adata.obs['cell_type'] == target_celltype) & 
        (adata.obs['condition'] == target_condition)
    ]
    if scipy.sparse.issparse(true_data):
        true_data = true_data.toarray()
    
    if true_data.shape[0] > sample_size:
        indices = np.random.choice(true_data.shape[0], sample_size, replace=False)
        true_data = true_data[indices]
    elif true_data.shape[0] < sample_size:
        indices = np.random.choice(true_data.shape[0], sample_size, replace=True)
        true_data = true_data[indices]

    baselines = {}
    
    # Context Mean Baseline
    print("Training Context Mean Baseline...")
    context_mean_baseline = Task3ContextMeanBaseline()
    context_mean_baseline.fit(adata, target_celltype, available_conditions)
    baselines['context_mean'] = context_mean_baseline
    
    # Perturbation Mean Baseline
    print("Training Perturbation Mean Baseline...")
    perturbation_mean_baseline = Task3PerturbationMeanBaseline()
    perturbation_mean_baseline.fit(adata, target_celltype, target_condition, other_celltypes)
    baselines['perturbation_mean'] = perturbation_mean_baseline
    
    # Linear Regression Baseline
    print("Training Linear Regression Baseline...")
    linear_baseline = Task3LinearBaseline(gene_names, cell_types, conditions)
    linear_baseline.fit(adata, target_celltype, target_condition)
    baselines['linear_regression'] = linear_baseline
    
    for baseline_name, baseline_model in baselines.items():
        print(f"Predicting with {baseline_name}...")

        if baseline_name == 'linear_regression':
            predicted_data = baseline_model.predict(target_celltype, target_condition, sample_size)
        else:
            predicted_data = baseline_model.predict(target_celltype, target_condition, sample_size)

        baseline_output_path = os.path.join(output_base_path, baseline_name)
        os.makedirs(baseline_output_path, exist_ok=True)

        true_file = os.path.join(baseline_output_path, f"{target_condition}_true_values.npy")
        pred_file = os.path.join(baseline_output_path, f"{target_condition}_predicted_values.npy")
        
        np.save(true_file, true_data)
        np.save(pred_file, predicted_data)
        
        print(f"{baseline_name} - True: {true_file}")
        print(f"{baseline_name} - Pred: {pred_file}")

    for baseline_name in baselines.keys():
        baseline_output_path = os.path.join(output_base_path, baseline_name)
        gene_names_file = os.path.join(baseline_output_path, "gene_names.npy")
        np.save(gene_names_file, gene_names)


if __name__ == "__main__":
    dat_list = ['Kang.h5ad', 'Haber.h5ad', "Hagai.h5ad", "Weinreb_time.h5ad", 
                'Srivatsan_sciplex3_sub10.h5ad', 'Burkhardt_sub10.h5ad', 
                'Perturb_cmo_V1_sub10.h5ad', 'Perturb_KHP_sub10.h5ad', 
                'Tahoe100_sub10.h5ad', "Parse_10M_PBMC_sub10.h5ad"]

    dataset_paths = [f"/data2/lanxiang/data/Task3_data/{dat}" for dat in dat_list]
    
    print("Starting Task3 Baseline Analysis...")
    print(f"Processing {len(dataset_paths)} datasets in order:")
    for i, path in enumerate(dataset_paths, 1):
        print(f"  {i}. {os.path.basename(path)}")
    print()
    
    run_task3_baselines_auto(dataset_paths)