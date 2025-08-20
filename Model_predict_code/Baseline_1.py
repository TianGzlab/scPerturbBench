import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import scanpy as sc
import os
from typing import Counter


# Separate A and B from combinations
def parse_conditions(conditions):
    parsed_conditions = []
    for cond in conditions:
        if "+" in cond:
            A, B = cond.split("+")
            A, B = A.strip(), B.strip()
            parsed_conditions.append((A, B))
    return parsed_conditions


# Encode conditions and sample gene expression
def sample_conditions(adata, condition_list, sample_size=32):
    condition_dict = {}
    
    # Iterate through each condition pair A and B
    for A, B in condition_list:
        # Generate condition names with 'ctrl'
        A_ctrl = f"{A}+ctrl"
        B_ctrl = f"{B}+ctrl"
        
        # Check condition A+ctrl and sample
        condition_A_data = adata[adata.obs['condition'] == A_ctrl].X
        if condition_A_data.shape[0] > 0:
            if condition_A_data.shape[0] <= sample_size:
                indices = np.random.choice(condition_A_data.shape[0], sample_size, replace=True)
            else:
                indices = np.random.choice(condition_A_data.shape[0], sample_size, replace=False)
            condition_dict[A_ctrl] = condition_A_data[indices].toarray()
        else:
            print(f"Warning: No data found for condition {A_ctrl}. Filling with zeros.")
            condition_dict[A_ctrl] = np.zeros((sample_size, adata.shape[1])) 

        # Check condition B+ctrl and sample
        condition_B_data = adata[adata.obs['condition'] == B_ctrl].X
        if condition_B_data.shape[0] > 0:
            if condition_B_data.shape[0] <= sample_size:
                indices = np.random.choice(condition_B_data.shape[0], sample_size, replace=True)
            else:
                indices = np.random.choice(condition_B_data.shape[0], sample_size, replace=False)
            condition_dict[B_ctrl] = condition_B_data[indices].toarray()
        else:
            print(f"Warning: No data found for condition {B_ctrl}. Filling with zeros.")
            condition_dict[B_ctrl] = np.zeros((sample_size, adata.shape[1]))

        # Check condition A+B and sample
        condition_AB_data = adata[adata.obs['condition'] == f"{A}+{B}"].X
        if condition_AB_data.shape[0] > 0:
            if condition_AB_data.shape[0] <= sample_size:
                indices = np.random.choice(condition_AB_data.shape[0], sample_size, replace=True)
            else:
                indices = np.random.choice(condition_AB_data.shape[0], sample_size, replace=False)
            condition_dict[f"{A}+{B}"] = condition_AB_data[indices].toarray()
        else:
            print(f"Warning: No data found for condition {A}+{B}. Filling with zeros.")
            condition_dict[f"{A}+{B}"] = np.zeros((sample_size, adata.shape[1])) 

    return condition_dict
# Model definition
class GeneExpressionModel(nn.Module):
    def __init__(self, gene_dim, embedding_dim):
        super(GeneExpressionModel, self).__init__()
        self.ctrl_embedding = nn.Sequential(
            nn.Linear(gene_dim, embedding_dim),
            nn.ReLU()
        )
        self.condition_embedding = nn.Sequential(
            nn.Linear(gene_dim, embedding_dim),  # Gene_dim -> embedding_dim
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, gene_dim)
        )

    def forward(self, gene_expr, condition):
        z1 = self.ctrl_embedding(gene_expr)
        z2 = self.condition_embedding(condition)
        z = z1 + z2
        return self.predictor(z)
    

# One-hot encoding function: gene dimension
def match_condition_with_genes(gene_names, condition):
    """
    Generate a matching vector based on condition and gene_names.
    If a gene_name appears in condition, set corresponding position to 1, else 0.
    """
    match_vector = np.array([1 if gene in condition else 0 for gene in gene_names], dtype=np.float32)
    return match_vector
    
def run_split(data_path,picklefile, picklefile_cond, output_base_path):

    adata = sc.read_h5ad(data_path)
    gene_names = adata.var_names 
    split=pd.read_pickle(picklefile)
    # Define sample size
    sample_size = Counter(adata.obs['condition'] == 'ctrl')[1]


    val_cond=pd.read_pickle(picklefile_cond)
    val_cond=sum(val_cond['test_subgroup'].values(), [])
    cond_to_predict=parse_conditions(val_cond)
    cond_dict_pred= sample_conditions(adata, cond_to_predict, sample_size)
    print(cond_dict_pred)
    parsed_conditions=parse_conditions(split['train'])
    condition_dict = sample_conditions(adata, parsed_conditions, sample_size)
    
    # Parameters
    gene_dim = adata.shape[1]
    embedding_dim = 128
    epochs = 10
    

    model = GeneExpressionModel(gene_dim, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    ctrl_data = adata[adata.obs['condition'] == 'ctrl'].X
    ctrl_train = torch.tensor(ctrl_data[np.random.choice(ctrl_data.shape[0], sample_size, replace=False)].toarray(), dtype=torch.float32)

    for epoch in range(epochs):
            model.train()
            for cond in condition_dict:
                perturb_condition = torch.tensor(match_condition_with_genes(gene_names, cond), dtype=torch.float32).unsqueeze(0)
                perturb_train=condition_dict[cond]
                perturb_train = torch.tensor(perturb_train, dtype=torch.float32)

                optimizer.zero_grad()
                output_A = model(ctrl_train, perturb_condition)
                loss_A = criterion(output_A, perturb_train)
                loss_A.backward()
                optimizer.step()
        
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_A.item()}")
    
    # Train and test each combination
    for A, B in cond_to_predict:
        print(f"Processing conditions: A+B={A}+{B}")

        perturbAB_condition = torch.tensor(match_condition_with_genes(gene_names, f"{A}+{B}"), dtype=torch.float32).unsqueeze(0)

        # Test
        model.eval()
        with torch.no_grad():
            predictions = model(ctrl_train, perturbAB_condition)

        os.makedirs(output_base_path, exist_ok=True)

        # Save perturbAB_test matrix
        perturbAB_data = cond_dict_pred[f"{A}+{B}"]
        perturbAB_test = torch.tensor(perturbAB_data, dtype=torch.float32)

        perturbAB_test_file = os.path.join(output_base_path, f"{A}_{B}_true_values.npy")
        np.save(perturbAB_test_file, perturbAB_data)

        # Save predictions matrix
        predictions_file = os.path.join(output_base_path, f"{A}_{B}_predicted_values.npy")
        np.save(predictions_file, predictions)
    
        print(f"True matrix saved to {perturbAB_test_file}")
        print(f"Predicted matrix saved to {predictions_file}")



    gene_names_file = os.path.join(output_base_path, "gene_names.npy")
    np.save(gene_names_file, gene_names)





#task1
dat_list=["dixitregev2016","arce_mm_crispri_filtered","normanweissman2019_filtered","sunshine2023_crispri_sarscov2",'datlingerbock2017','datlingerbock2021','frangiehizar2021_rna','papalexisatija2021_eccite_rna','replogleweissman2022_rpe1','tiankampmann2021_crispra','tiankampmann2021_crispri', 'fengzhang2023', 'vcc_train_filtered',"perturb_processed_adamson","junyue_cao_filtered"]
output_base_path="/data2/yue_data/pert/baseline/task1/"
dat_path='/data2/yue_data/pert/data'
for dat in dat_list:
    print(dat)
    dataset=dat
    data_path=f'{dat_path}/{dat}/perturb_processed.h5ad'
    pk1=f'{dat_path}/{dat}/splits/{dat}_simulation_1_0.75.pkl'
    pk2=f'{dat_path}/{dat}/splits/{dat}_simulation_1_0.75_subgroup.pkl'
    run_split(data_path, pk1,pk2,os.path.join(output_base_path,dataset))


