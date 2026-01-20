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


def parse_task2_conditions(conditions):
    parsed_conditions = []
    for cond in conditions:
        if "+" in cond and not cond.endswith("+ctrl"):
            parts = cond.split("+")
            if len(parts) == 2:  
                A, B = parts[0].strip(), parts[1].strip()
                parsed_conditions.append((A, B))
    return parsed_conditions


def sample_conditions(adata, condition_list, sample_size=32):
    condition_dict = {}
    
    for A, B in condition_list:
        A_ctrl = f"{A}+ctrl"
        B_ctrl = f"{B}+ctrl"
        AB_comb = f"{A}+{B}"
        
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
            condition_dict[B_ctrl] = np.zeros((sample_size, adata.shape[1]))

        condition_AB_data = adata[adata.obs['condition'] == AB_comb].X
        if condition_AB_data.shape[0] > 0:
            if condition_AB_data.shape[0] <= sample_size:
                indices = np.random.choice(condition_AB_data.shape[0], sample_size, replace=True)
            else:
                indices = np.random.choice(condition_AB_data.shape[0], sample_size, replace=False)
            if scipy.sparse.issparse(condition_AB_data):
                condition_dict[AB_comb] = condition_AB_data[indices].toarray()
            else:
                condition_dict[AB_comb] = condition_AB_data[indices]
        else:
            condition_dict[AB_comb] = np.zeros((sample_size, adata.shape[1]))

    return condition_dict


def match_condition_with_genes(gene_names, condition):
    if "+" in condition and not condition.endswith("+ctrl"):
        genes_in_condition = [gene.strip() for gene in condition.split("+")]
        match_vector = np.array([1 if gene in genes_in_condition else 0 for gene in gene_names], dtype=np.float32)
    elif condition.endswith("+ctrl"):
        target_gene = condition.replace("+ctrl", "").strip()
        match_vector = np.array([1 if gene == target_gene else 0 for gene in gene_names], dtype=np.float32)
    else:

        target_gene = condition.strip()
        match_vector = np.array([1 if gene == target_gene else 0 for gene in gene_names], dtype=np.float32)
    

    matched_count = np.sum(match_vector)
    if matched_count == 0:
        print(f"Warning: No match found for condition '{condition}'")
    
    return match_vector


class Task2ContextMeanBaseline:
    
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


class Task2AdditivePerturbationMeanBaseline:
    """Task2 Additive Perturbation-mean baseline: X̂_A+B = X_ctrl + Δ_A + Δ_B"""
    
    def __init__(self):
        self.ctrl_mean = None
        self.single_effects = {} 
    
    def fit(self, adata, train_conditions):
        ctrl_data = adata[adata.obs['condition'] == 'ctrl'].X
        if scipy.sparse.issparse(ctrl_data):
            ctrl_data = ctrl_data.toarray()
        self.ctrl_mean = np.mean(ctrl_data, axis=0)
        
        for condition in train_conditions:
            if condition == 'ctrl':
                continue
            if condition.endswith('+ctrl'):
                pert_name = condition.replace('+ctrl', '')
                condition_data = adata[adata.obs['condition'] == condition].X
                if condition_data.shape[0] > 0:
                    if scipy.sparse.issparse(condition_data):
                        condition_data = condition_data.toarray()
                    pert_mean = np.mean(condition_data, axis=0)
                    self.single_effects[pert_name] = pert_mean - self.ctrl_mean
    
    def predict(self, target_conditions, sample_size):
        predictions = {}
        
        for condition in target_conditions:
            if '+' in condition and not condition.endswith('+ctrl'):
                parts = condition.split('+')
                if len(parts) == 2:
                    A, B = parts
                    A, B = A.strip(), B.strip()
                    
                    predicted_expr = self.ctrl_mean.copy()
                    
                    if A in self.single_effects:
                        predicted_expr += self.single_effects[A]
                    else:
                        print(f"Warning: No effect found for perturbation {A}")
                    
                    if B in self.single_effects:
                        predicted_expr += self.single_effects[B]
                    else:
                        print(f"Warning: No effect found for perturbation {B}")
                    
                    predictions[condition] = np.tile(predicted_expr, (sample_size, 1))
                else:
                    predictions[condition] = np.tile(self.ctrl_mean, (sample_size, 1))
            else:
                predictions[condition] = np.tile(self.ctrl_mean, (sample_size, 1))
        
        return predictions


class Task2LinearBaseline:
    
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


def run_task2_baselines(data_path, picklefile, picklefile_cond, output_base_path):
    print("Running Task2 baselines...")
    
    adata = sc.read_h5ad(data_path)
    gene_names = adata.var_names
    split = pd.read_pickle(picklefile)
    sample_size = Counter(adata.obs['condition'] == 'ctrl')[1]

    val_cond = pd.read_pickle(picklefile_cond)
    all_test_conditions = sum(val_cond['test_subgroup'].values(), [])
    cond_to_predict = parse_task2_conditions(all_test_conditions)
    
    print(f"All test conditions: {len(all_test_conditions)}")
    print(f"Task2 combination conditions: {len(cond_to_predict)}")
    print(f"Task2 conditions: {cond_to_predict}")
    
    if not cond_to_predict:
        print("No Task2 combination conditions found. Exiting.")
        return
    
    cond_dict_pred = sample_conditions(adata, cond_to_predict, sample_size)
    

    train_conditions = split['train']
    train_conditions.append('ctrl')  

    baselines = {}
    
    # Context Mean Baseline
    print("Training Context Mean Baseline...")
    context_mean_baseline = Task2ContextMeanBaseline()
    context_mean_baseline.fit(adata, train_conditions)
    baselines['context_mean'] = context_mean_baseline
    
    # Additive Perturbation Mean Baseline
    print("Training Additive Perturbation Mean Baseline...")
    additive_baseline = Task2AdditivePerturbationMeanBaseline()
    additive_baseline.fit(adata, train_conditions)
    baselines['additive_perturbation_mean'] = additive_baseline
    
    # Linear Regression Baseline
    print("Training Linear Regression Baseline...")
    linear_baseline = Task2LinearBaseline(gene_names)
    linear_baseline.fit(adata, train_conditions)
    baselines['linear_regression'] = linear_baseline

    for A, B in cond_to_predict:
        condition_name = f"{A}+{B}" 
        print(f"Processing condition: {condition_name}")
        

        true_data = cond_dict_pred[condition_name]
        
        for baseline_name, baseline_model in baselines.items():
            predictions = baseline_model.predict([condition_name], sample_size)
            predicted_data = predictions[condition_name]
   
            baseline_output_path = os.path.join(output_base_path, baseline_name)
            os.makedirs(baseline_output_path, exist_ok=True)

            true_file = os.path.join(baseline_output_path, f"{A}_{B}_true_values.npy")
            pred_file = os.path.join(baseline_output_path, f"{A}_{B}_predicted_values.npy")
            
            np.save(true_file, true_data)
            np.save(pred_file, predicted_data)
            
            print(f"{baseline_name} - True: {true_file}")
            print(f"{baseline_name} - Pred: {pred_file}")

    for baseline_name in baselines.keys():
        baseline_output_path = os.path.join(output_base_path, baseline_name)
        gene_names_file = os.path.join(baseline_output_path, "gene_names.npy")
        np.save(gene_names_file, gene_names)


if __name__ == "__main__":
    dat_list=["dixitregev2016","normanweissman2019_filtered","sunshine2023_crispri_sarscov2","arce_mm_crispri_filtered"]
    output_base_path = "/data2/lanxiang/perturb_benchmark_v2/SA_review/Round1/Baseline_review/Task2/"
    dat_path = '/data2/yue_data/pert/data'
    
    for dat in dat_list:
        print(f"Processing dataset: {dat}")
        dataset = dat
        data_path = f'{dat_path}/{dat}/perturb_processed.h5ad'
        pk1 = f'{dat_path}/{dat}/splits/{dat}_simulation_1_0.75.pkl'
        pk2 = f'{dat_path}/{dat}/splits/{dat}_simulation_1_0.75_subgroup.pkl'
        
        run_task2_baselines(data_path, pk1, pk2, os.path.join(output_base_path, dataset))