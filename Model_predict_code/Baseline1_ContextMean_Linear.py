import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import scanpy as sc
import os
from sklearn.linear_model import LinearRegression
from typing import Counter
import scipy.sparse


def parse_conditions(conditions):
    parsed_conditions = []
    for cond in conditions:
        if "+" in cond:
            A, B = cond.split("+")
            A, B = A.strip(), B.strip()
            parsed_conditions.append((A, B))
    return parsed_conditions


def sample_conditions(adata, condition_list, sample_size=32):
    condition_dict = {}
    
    for A, B in condition_list:
        A_ctrl = f"{A}+ctrl"
        B_ctrl = f"{B}+ctrl"
        
        condition_A_data = adata[adata.obs['condition'] == A_ctrl].X
        if condition_A_data.shape[0] > 0:
            if condition_A_data.shape[0] <= sample_size:
                indices = np.random.choice(condition_A_data.shape[0], sample_size, replace=True)
            else:
                indices = np.random.choice(condition_A_data.shape[0], sample_size, replace=False)
            if scipy.sparse.issparse(condition_A_data):
                condition_dict[A_ctrl] = condition_A_data[indices].toarray()
            else:
                condition_dict[A_ctrl] = condition_A_data[indices]
        else:
            print(f"Warning: No data found for condition {A_ctrl}. Filling with zeros.")
            condition_dict[A_ctrl] = np.zeros((sample_size, adata.shape[1]))

        condition_B_data = adata[adata.obs['condition'] == B_ctrl].X
        if condition_B_data.shape[0] > 0:
            if condition_B_data.shape[0] <= sample_size:
                indices = np.random.choice(condition_B_data.shape[0], sample_size, replace=True)
            else:
                indices = np.random.choice(condition_B_data.shape[0], sample_size, replace=False)
            if scipy.sparse.issparse(condition_B_data):
                condition_dict[B_ctrl] = condition_B_data[indices].toarray()
            else:
                condition_dict[B_ctrl] = condition_B_data[indices]
        else:
            print(f"Warning: No data found for condition {B_ctrl}. Filling with zeros.")
            condition_dict[B_ctrl] = np.zeros((sample_size, adata.shape[1]))

    return condition_dict


def match_condition_with_genes(gene_names, condition):
    if "+ctrl" in condition:
        target_gene = condition.replace("+ctrl", "").strip()
    elif "+" in condition:
        target_gene = condition.strip()
    else:
        target_gene = condition.strip()
    
    match_vector = np.array([1 if gene == target_gene else 0 for gene in gene_names], dtype=np.float32)
    
    matched_count = np.sum(match_vector)
    if matched_count == 0:
        print(f"Warning: No exact match found for '{target_gene}' in condition '{condition}'")
    elif matched_count > 1:
        print(f"Warning: Multiple matches found for '{target_gene}' in condition '{condition}'")
    
    return match_vector


class Task1ContextMeanBaseline:
    
    def __init__(self):
        self.context_mean = None
    
    def fit(self, adata, train_conditions):
        all_train_data = []
        
        for condition in train_conditions:
            condition_data = adata[adata.obs['condition'] == condition].X
            if condition_data.shape[0] > 0:
                if scipy.sparse.issparse(condition_data):
                    condition_data = condition_data.toarray()
                all_train_data.append(condition_data)

        if all_train_data:
            all_data = np.vstack(all_train_data)
            self.context_mean = np.mean(all_data, axis=0)
        else:
            print("Warning: No training data found.")
            self.context_mean = np.zeros(adata.shape[1])
    
    def predict(self, target_conditions, sample_size):
        predictions = {}
        for condition in target_conditions:
            predictions[condition] = np.tile(self.context_mean, (sample_size, 1))
        return predictions


class Task1LinearBaseline:
    def __init__(self, gene_names):
        self.gene_names = gene_names
        self.model = LinearRegression()
    
    def fit(self, adata, train_conditions):
        X_train = []
        y_train = []
        for condition in train_conditions:
            condition_data = adata[adata.obs['condition'] == condition].X
            if condition_data.shape[0] > 0:
                if scipy.sparse.issparse(condition_data):
                    condition_data = condition_data.toarray()
                
                condition_encoding = match_condition_with_genes(self.gene_names, condition)
                
                for sample in condition_data:
                    X_train.append(condition_encoding)
                    y_train.append(sample)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"Training multi-output linear regression for {len(self.gene_names)} genes...")
        self.model.fit(X_train, y_train)
        print("Linear regression training completed!")
    
    def predict(self, target_conditions, sample_size):
        predictions = {}
        
        for condition in target_conditions:
            condition_encoding = match_condition_with_genes(self.gene_names, condition)
            
            predicted_expr = self.model.predict([condition_encoding])[0]
            
            predictions[condition] = np.tile(predicted_expr, (sample_size, 1))
        
        return predictions


def sample_test_conditions_task1(adata, test_conditions, sample_size):
    condition_dict = {}
    
    for condition in test_conditions:
        condition_data = adata[adata.obs['condition'] == condition].X
        if condition_data.shape[0] > 0:
            if condition_data.shape[0] <= sample_size:
                indices = np.random.choice(condition_data.shape[0], sample_size, replace=True)
            else:
                indices = np.random.choice(condition_data.shape[0], sample_size, replace=False)
            
            if scipy.sparse.issparse(condition_data):
                condition_dict[condition] = condition_data[indices].toarray()
            else:
                condition_dict[condition] = condition_data[indices]
        else:
            print(f"Warning: No data found for test condition {condition}.")
            condition_dict[condition] = np.zeros((sample_size, adata.shape[1]))
    
    return condition_dict


def run_task1_baselines(data_path, picklefile, picklefile_cond, output_base_path):
    print("Running Task1 baselines...")
    adata = sc.read_h5ad(data_path)
    gene_names = adata.var_names
    split = pd.read_pickle(picklefile)
    sample_size = Counter(adata.obs['condition'] == 'ctrl')[1]

    val_cond = pd.read_pickle(picklefile_cond)
    test_conditions = sum(val_cond['test_subgroup'].values(), [])
    print(f"Test conditions: {test_conditions}")

    test_condition_dict = sample_test_conditions_task1(adata, test_conditions, sample_size)
    

    train_conditions = split['train']
    if 'ctrl' not in train_conditions:
        train_conditions.append('ctrl') 
    print(f"Train conditions: {train_conditions}")
    
    baselines = {}
    
    # Context Mean Baseline
    print("Training Context Mean Baseline...")
    context_mean_baseline = Task1ContextMeanBaseline()
    context_mean_baseline.fit(adata, train_conditions)
    baselines['context_mean'] = context_mean_baseline
    
    # Linear Regression Baseline
    print("Training Linear Regression Baseline...")
    linear_baseline = Task1LinearBaseline(gene_names)
    linear_baseline.fit(adata, train_conditions)
    baselines['linear_regression'] = linear_baseline

    for test_condition in test_conditions:
        print(f"Processing condition: {test_condition}")

        if test_condition in test_condition_dict:
            true_data = test_condition_dict[test_condition]
        else:
            print(f"Warning: No test data for {test_condition}. Skipping.")
            continue
        
        for baseline_name, baseline_model in baselines.items():
            predictions = baseline_model.predict([test_condition], sample_size)
            predicted_data = predictions[test_condition]
            
            baseline_output_path = os.path.join(output_base_path, baseline_name)
            os.makedirs(baseline_output_path, exist_ok=True)
            
            if "+ctrl" in test_condition:
                gene_name = test_condition.replace("+ctrl", "")
                true_file = os.path.join(baseline_output_path, f"{gene_name}_ctrl_true_values.npy")
                pred_file = os.path.join(baseline_output_path, f"{gene_name}_ctrl_predicted_values.npy")
            else:
                clean_name = test_condition.replace("+", "_").replace("-", "_")
                true_file = os.path.join(baseline_output_path, f"{clean_name}_true_values.npy")
                pred_file = os.path.join(baseline_output_path, f"{clean_name}_predicted_values.npy")
            
            np.save(true_file, true_data)
            np.save(pred_file, predicted_data)
            
            print(f"{baseline_name} - True: {true_file}")
            print(f"{baseline_name} - Pred: {pred_file}")
    
    for baseline_name in baselines.keys():
        baseline_output_path = os.path.join(output_base_path, baseline_name)
        gene_names_file = os.path.join(baseline_output_path, "gene_names.npy")
        np.save(gene_names_file, gene_names)


if __name__ == "__main__":
    dat_list=['datlingerbock2017','datlingerbock2021',"perturb_processed_adamson","fengzhang2023",'frangiehizar2021_rna','papalexisatija2021_eccite_rna','replogleweissman2022_rpe1','tiankampmann2021_crispra','tiankampmann2021_crispri',"junyue_cao_filtered",'vcc_train_filtered',"dixitregev2016","normanweissman2019_filtered","sunshine2023_crispri_sarscov2","arce_mm_crispri_filtered"]
    output_base_path = "/data2/lanxiang/perturb_benchmark_v2/model/Baseline_review/Task1/"
    dat_path = '/data2/yue_data/pert/data'
    
    for dat in dat_list:
        print(f"Processing dataset: {dat}")
        dataset = dat
        data_path = f'{dat_path}/{dat}/perturb_processed.h5ad'
        pk1 = f'{dat_path}/{dat}/splits/{dat}_simulation_1_0.75.pkl'
        pk2 = f'{dat_path}/{dat}/splits/{dat}_simulation_1_0.75_subgroup.pkl'
        
        run_task1_baselines(data_path, pk1, pk2, os.path.join(output_base_path, dataset))