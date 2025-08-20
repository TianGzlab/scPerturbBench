import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse



# MLP model definition
class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x


# Define the complete model
class Model(nn.Module):
        def __init__(self, gene_input_dim, condition_input_dim, hidden_dim, output_dim):
            super(Model, self).__init__()
            self.mlp_gene = MLP(gene_input_dim, hidden_dim, hidden_dim)
            self.mlp_condition = MLP(condition_input_dim, hidden_dim, hidden_dim)
            self.mlp_final = MLP(hidden_dim, hidden_dim, output_dim)

        def forward(self, gene_data, condition_onehot):
            z1 = self.mlp_gene(gene_data)  # (batch_size, hidden_dim)
            z2 = self.mlp_condition(condition_onehot)  # (batch_size, hidden_dim)

            z_combined = z1+z2  
            output = self.mlp_final(z_combined)
            return output


# Training function
def train_model(model, train_data, train_condition, true_expr, optimizer, loss_fn):
        model.train()
        optimizer.zero_grad()
        predictions = model(train_data, train_condition)
        loss = loss_fn(predictions, true_expr)
        loss.backward()
        optimizer.step()
        return loss.item()


# Prediction function
def predict(model, gene_data, condition_onehot):
        model.eval()
        with torch.no_grad():
            return model(gene_data, condition_onehot)





dataset_paths = [
    "/data2/lanxiang/data/Task3_data/Hagai.h5ad",
    "/data2/lanxiang/data/Task3_data/Kang.h5ad",
    "/data2/lanxiang/data/Task3_data/Weinreb_time.h5ad",
     "/data2/lanxiang/data/Task3_data/Haber.h5ad",

    "/data2/lanxiang/data/Task3_data/Burkhardt_sub10.h5ad",
    "/data2/lanxiang/data/Task3_data/Perturb_cmo_V1_sub10.h5ad",
    "/data2/lanxiang/data/Task3_data/Perturb_KHP_sub10.h5ad",
    "/data2/lanxiang/data/Task3_data/Tahoe100_sub10.h5ad",
    "/data2/lanxiang/data/Task3_data/Srivatsan_sciplex3_sub10.h5ad",
    "/data2/lanxiang/data/Task3_data/Parse_10M_PBMC_sub10.h5ad"



]


for data_path in dataset_paths:
    adata = sc.read(data_path)

    # Check and standardize 'cell_type' and 'condition' column names
    if 'cell_type' not in adata.obs.columns:
        if 'species' in adata.obs.columns:
            adata.obs.rename(columns={'species': 'cell_type'}, inplace=True)
        elif 'celltype' in adata.obs.columns:
            adata.obs.rename(columns={'celltype': 'cell_type'}, inplace=True)

    if 'cell_type' not in adata.obs or 'condition' not in adata.obs:
        print(f"Dataset {data_path} is missing 'cell_type' or 'condition' columns. Skipping this dataset.")
        continue

    # Standardize the 'condition' column values
    adata.obs['condition'] = adata.obs['condition'].replace({'control': 'ctrl', 'Control': 'ctrl', 'unst': 'ctrl'})

    cell_types = adata.obs['cell_type'].unique()  
    conditions = adata.obs['condition'].unique()  
    conditions = [cond for cond in conditions if cond != 'ctrl']

    gene_names = adata.var_names 

    # One-hot encoding for conditions
    unique_conditions = conditions
    condition_map = {condition: idx for idx, condition in enumerate(unique_conditions)}

    
    # Initialize model, optimizer, and loss function
    gene_input_dim = adata.raw.X.shape[1] if adata.raw is not None else adata.X.shape[1]  
    condition_input_dim = len(unique_conditions)  
    hidden_dim = 128  
    output_dim = gene_input_dim  


    dataset_name = os.path.basename(data_path).split('.')[0]  
    print(dataset_name)

    for condition in unique_conditions:
        print(condition)

        condition_samples = adata[adata.obs['condition'] == condition]

        for target_cell in cell_types:
            print(target_cell)
            # other cell contorl
            other_cells_ctrl = adata.X[(adata.obs['cell_type'] != target_cell) & (adata.obs['condition'] == 'ctrl')]       
            # target cell control
            target_cell_data_ctrl = adata.X[(adata.obs['cell_type'] == target_cell) & (adata.obs['condition'] == 'ctrl')]      

            #target cell perturb
            target_cell_data_true = condition_samples.X[(condition_samples.obs['cell_type'] == target_cell)]      
            # other cell perturb
            other_cells_data = condition_samples.X[(condition_samples.obs['cell_type'] != target_cell) ]   
                
            select_num = min(target_cell_data_ctrl.shape[0],target_cell_data_true.shape[0])
            indices_ctrl = np.random.choice(target_cell_data_ctrl.shape[0], select_num, replace=False)
            indices_true = np.random.choice(target_cell_data_true.shape[0], select_num, replace=False)
            

            other_cells_ctrl=other_cells_ctrl[np.random.choice(other_cells_ctrl.shape[0], select_num, replace=True)]
            other_cells_data=other_cells_data[np.random.choice(other_cells_data.shape[0], select_num, replace=True)]
            target_cell_data_ctrl = target_cell_data_ctrl[indices_ctrl]
            target_cell_data_true = target_cell_data_true[indices_true]
            


            condition_onehot = torch.tensor(np.array([np.eye(len(unique_conditions))[condition_map[condition]]]), dtype=torch.float32)
        
            model = Model(gene_input_dim, condition_input_dim, hidden_dim, output_dim)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()  

            for epoch in range(30):  
                other_cells_data = other_cells_data.toarray() if scipy.sparse.issparse(other_cells_data) else other_cells_data
                other_cells_ctrl = other_cells_ctrl.toarray() if scipy.sparse.issparse(other_cells_ctrl) else other_cells_ctrl
                target_cell_data_ctrl = target_cell_data_ctrl.toarray() if scipy.sparse.issparse(target_cell_data_ctrl) else target_cell_data_ctrl
                target_cell_data_true = target_cell_data_true.toarray() if scipy.sparse.issparse(target_cell_data_ctrl) else target_cell_data_ctrl

                loss = train_model(
                    model,
                    torch.tensor(other_cells_ctrl, dtype=torch.float32),
                    condition_onehot,
                    torch.tensor(other_cells_data, dtype=torch.float32),
                    optimizer,
                    loss_fn
                )
                print(f"Epoch [{epoch+1}/10], Loss: {loss:.4f}")


            predictions_target_cell = predict(model, torch.tensor(target_cell_data_ctrl, dtype=torch.float32), condition_onehot)

            save_dir = f"/data2/yue_data/pert/baseline/task3/{dataset_name}/{target_cell}"
            os.makedirs(save_dir, exist_ok=True)


            perturb_test_file = os.path.join(save_dir, f"{condition}_true_values.npy")
            np.save(perturb_test_file, predictions_target_cell)
            print(predictions_target_cell.shape)

            predictions_file = os.path.join(save_dir, f"{condition}_predicted_values.npy")
            np.save(predictions_file, target_cell_data_true)
    
            print(f"True values saved to {perturb_test_file}")
            print(f"Predicted values saved to {predictions_file}")
