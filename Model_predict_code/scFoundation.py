import json
import os
from re import M
import time
import copy
import torch
import numpy as np
import argparse
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import logging

from gears import PertData
from os.path import join as pjoin
import pandas as pd
from pathlib import Path
from load import *
from finetune_model_pert import MaeAutobin_pert
from gears.utils import create_cell_graph_dataset_for_prediction
from torch_geometric.loader import DataLoader
import glob

# set seed 
set_seed(222222)


def setup_logger(log_file_path):
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    
    # Create a file handler to save logs to file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler to output logs to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def denormalize(data):
        data_nonzero = torch.where(data == 0, torch.tensor(1e9, device=data.device), data)
        min_values_y, _ = torch.min(data_nonzero, dim=1, keepdim=True)
        data = data / min_values_y
        return(data)

def load_model(model_file):
    model_data = torch.load(model_file)
    key='gene'
    model_data = model_data[key]
    model_data = convertconfig(model_data)

    if not model_data.__contains__('config'):
        print('***** No config *****')
        config={}
        config['model_type']='flash_all'
    else:
        config=model_data['config']
        print(config)
    if not config.__contains__('qv_dim'):
        if config['model'] != 'mae_autobin':
            if config.__contains__('dim_head'):
                config['qv_dim']=config['dim_head']
            else:
                print('***** No qv_dim ***** set 64')
                config['qv_dim']= 64
    if not config.__contains__('ppi_edge'):
        config['ppi_edge']=None

    model= MaeAutobin_pert(config)
    model_state_dict = model_data['model_state_dict']    

    return model, model_state_dict

def predict_all_subgroups(model, pert_data, data_name, save_dir, device, batch_size=32):
    """
    Unified prediction function that handles all subgroups.
    """
    os_gene = pd.read_csv('/data/yy_data/sc_model/scFoundation-main/OS_scRNA_gene_index.19264.tsv', sep='\t')
    ref_gene_ids = os_gene['gene_name'].tolist()
    
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.adata.var['gene_name'].tolist()
    
    results_dir = save_dir.joinpath('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    var_names_compressed_path = results_dir.joinpath('gene_names.csv.gz')
    pd.DataFrame(pert_data.adata.var['gene_name']).to_csv(var_names_compressed_path, index=False, compression='gzip')
    
    model.eval()
    
    subgroups = ['unseen_single', 'combo_seen0', 'combo_seen1', 'combo_seen2']
    
    with torch.no_grad():
        for subgroup in subgroups:
            if subgroup in pert_data.subgroup['test_subgroup']:
                if subgroup in ['combo_seen1', 'combo_seen2']:
                    pert_lists = pert_data.subgroup['test_subgroup']['combo_seen1'] + pert_data.subgroup['test_subgroup']['combo_seen2']
                else:
                    pert_lists = pert_data.subgroup['test_subgroup'][subgroup]
                
                print(f"Processing {subgroup} with {len(pert_lists)} perturbations")
                
                for query in pert_lists:
                    print(f"Predicting {query}")
                    
                    if query.split("+")[1] == "ctrl":
                        pert_list = [[query.split("+")[0]]]
                        key = query.split("+")[0]
                    elif query.split("+")[0] == "ctrl":
                        pert_list = [[query.split("+")[1]]]
                        key = query.split("+")[1]
                    else:
                        pert_list = [query.split("+")]
                        key = "_".join(query.split("+"))
                    
                    for pert in pert_list:
                        for i in pert:
                            if i not in gene_list:
                                raise ValueError(f"Gene {i} not in perturbation graph!")
                    
                    cell_graphs = create_cell_graph_dataset_for_prediction(
                        pert_list[0], ctrl_adata, gene_list, device, num_samples=pool_size
                    )
                    loader = DataLoader(cell_graphs, batch_size=batch_size, shuffle=False)
                    preds = []
                    
                    for batch_data in loader:
                        pred_gene_values = model.pred_perturb(
                            batch_data,
                            gene_ids=gene_list,
                            ref_gene_ids=ref_gene_ids,
                            pad_token_id=model.config['pad_token_id'],
                            seq_len=model.config['seq_len'],
                            amp=False
                        )
                        preds.append(pred_gene_values)
                    
                    preds = torch.cat(preds, dim=0)
                    pred_compressed_path = results_dir.joinpath(f'{query}.npy.gz')
                    np.savez_compressed(pred_compressed_path, pred=preds.cpu())
    
    print(f"Predictions saved to {results_dir}")

# 
def train(model: nn.Module, 
          train_loader: torch.utils.data.DataLoader,
          device,max_seq_len,
          gene_ids,scaler,optimizer,scheduler,logger,epoch) -> None:
    """
    Train the model for one epoch.
    """
    log_interval = 100

    model.train()
    total_loss, total_mse = 0.0, 0.0
    pad_token_id=model.config['pad_token_id']
    seq_len=model.config['seq_len']
    os_gene = pd.read_csv('/data/yy_data/sc_model/scFoundation-main/OS_scRNA_gene_index.19264.tsv', sep='\t')
    ref_gene_ids= os_gene['gene_name'].tolist()
    start_time = time.time()


    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.pert_idx)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        
        n_genes=int(len(x)/batch_size)
        ori_gene_values = x.view(batch_size, n_genes).to(device)
        #x_query=denormalize(ori_gene_values)
        x_query=ori_gene_values
        x_query=reorder_matrix_by_ref(x_query, gene_ids, ref_gene_ids)
        
        target_values = batch_data.y.view(batch_size, n_genes).to(device)
        #target_values=denormalize(target_values)
        target_values=reorder_matrix_by_ref(target_values, gene_ids, ref_gene_ids)

        pert_flags=torch.zeros_like(x_query).long().to(device)
        for idx, i in enumerate(batch_data.pert_idx):
            for j in i:
                 if  j != -1:
                    j=get_ref_indices_from_gene_indices(ref_gene_ids, gene_ids, [j])
                    pert_flags[idx, j[0]] = 1

        input_gene_ids = torch.arange(19264, device=device, dtype=torch.long).to(device)
        input_full_ids = torch.arange(19264+2, device=device, dtype=torch.long).to(device)
        # sample input_gene_id
        
        if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
     

        x_query = x_query[:, input_gene_ids]
        pert_flags = pert_flags[:, input_gene_ids]
        target_values = target_values[:, input_gene_ids]

        
        total_counts=torch.tensor(np.log10(np.array(batch_data.obs_counts)+1)).to(device)
        x_query=torch.cat((x_query, total_counts.unsqueeze(1),
                           total_counts.unsqueeze(1)), dim=1).to(device, dtype=torch.float32)
        
        pert_flags = torch.cat((pert_flags, torch.zeros((pert_flags.size(0), 1)).to(device),
                                torch.zeros((pert_flags.size(0), 1)).to(device)), dim=1)            

        
        input_gene_ids = torch.cat((input_gene_ids, torch.tensor([19264], device=device),
                                    torch.tensor([19264+1], device=device)), dim=0)
        
        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, input_full_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)


        #retain only expressed genes
        encoder_data_labels = x_query > 0
        
        encoder_data, encoder_data_padding = safe_gatherData(x_query, encoder_data_labels,pad_token_id)
        encoder_position_gene_ids, _ = safe_gatherData(mapped_input_gene_ids, encoder_data_labels,pad_token_id)

        decoder_position_gene_ids = mapped_input_gene_ids
        decoder_data_padding = torch.full_like(x_query, False, dtype=torch.bool).to(device)

        encoder_position_gene_ids[encoder_data_padding] = seq_len
        decoder_position_gene_ids[decoder_data_padding] = seq_len

        pert_position_gene_ids, _ = safe_gatherData(pert_flags.long(), encoder_data_labels,0)

        with torch.cuda.amp.autocast(enabled=True):
            output = model(x=encoder_data, padding_label=encoder_data_padding,
                        encoder_position_gene_ids=encoder_position_gene_ids,
                        encoder_perturb_label= pert_position_gene_ids,
                        encoder_labels=encoder_data_labels,
                        decoder_data=x_query,
                        mask_gene_name=False,
                        mask_labels=None,
                        decoder_position_gene_ids=decoder_position_gene_ids,
                        decoder_data_padding_labels=decoder_data_padding,
                        )     
            #print(output)
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("Model output contains NaN or Inf!")
                print(f"Input stats: min={encoder_data.min()}, max={encoder_data.max()}")
                print(f"Output stats: min={output.min()}, max={output.max()}")
                continue  # 跳过这个batch
            
            masked_positions = torch.ones_like(target_values, dtype=torch.bool)
            loss = loss_mse = masked_mse_loss(output[:,:-2], target_values, masked_positions)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("Loss is NaN or Inf!")
                continue
        model.zero_grad()
        scaler.scale(loss).backward()
        print("Before unscale_")
        scaler.unscale_(optimizer)
        print("After unscale_")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not check_gradients(model):
            print("Skipping optimizer step due to bad gradients")
            scaler.update()
            continue
        scaler.step(optimizer)
        scaler.update()
        print("After step and update")

        torch.cuda.empty_cache()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()
 
def evaluate(model: nn.Module, val_loader: torch.utils.data.DataLoader,
             device,max_seq_len,gene_ids
             ) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    pad_token_id=model.config['pad_token_id']
    seq_len=model.config['seq_len']
    total_loss = 0.0
    os_gene = pd.read_csv('/data/yy_data/sc_model/scFoundation-main/OS_scRNA_gene_index.19264.tsv', sep='\t')
    ref_gene_ids= os_gene['gene_name'].tolist()
    
    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_size = len(batch_data.pert_idx)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        
            n_genes=int(len(x)/batch_size)
            ori_gene_values = x.view(batch_size, n_genes).to(device)
            #x_query=denormalize(ori_gene_values)
            x_query=ori_gene_values
            x_query=reorder_matrix_by_ref(x_query, gene_ids, ref_gene_ids)
        
            target_values = batch_data.y.view(batch_size, n_genes).to(device)
            #target_values=denormalize(target_values)
            target_values=reorder_matrix_by_ref(target_values, gene_ids, ref_gene_ids)

            pert_flags=torch.zeros_like(x_query).long().to(device)
            for idx, i in enumerate(batch_data.pert_idx):
                for j in i:
                    if  j != -1:
                        j=get_ref_indices_from_gene_indices(ref_gene_ids, gene_ids, [j])
                        pert_flags[idx, j[0]] = 1

            input_gene_ids = torch.arange(19264, device=device, dtype=torch.long).to(device)
            input_full_ids = torch.arange(19264+2, device=device, dtype=torch.long).to(device)
            # sample input_gene_id
            
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
       

            x_query = x_query[:, input_gene_ids]
            pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_values[:, input_gene_ids]

            total_counts=torch.tensor(np.log10(np.array(batch_data.obs_counts)+1)).to(device)
            x_query=torch.cat((x_query, total_counts.unsqueeze(1),
                           total_counts.unsqueeze(1)), dim=1).to(device, dtype=torch.float32)
        
            pert_flags = torch.cat((pert_flags, torch.zeros((pert_flags.size(0), 1)).to(device),
                                torch.zeros((pert_flags.size(0), 1)).to(device)), dim=1)            

        
            input_gene_ids = torch.cat((input_gene_ids, torch.tensor([19264], device=device),
                                    torch.tensor([19264+1], device=device)), dim=0)
        
            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, input_full_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            encoder_data_labels = x_query > 0

            encoder_data, encoder_data_padding = safe_gatherData(x_query, encoder_data_labels,pad_token_id)
            encoder_position_gene_ids, _ = safe_gatherData(mapped_input_gene_ids, encoder_data_labels,pad_token_id)

            decoder_position_gene_ids = mapped_input_gene_ids
            decoder_data_padding = torch.full_like(x_query, False, dtype=torch.bool).to(device)

            encoder_position_gene_ids[encoder_data_padding] = seq_len
            decoder_position_gene_ids[decoder_data_padding] = seq_len

            pert_position_gene_ids, _ = safe_gatherData(pert_flags.long(), encoder_data_labels,0)
           
            with torch.cuda.amp.autocast(enabled=True):
                output = model(x=encoder_data, padding_label=encoder_data_padding,
                        encoder_position_gene_ids=encoder_position_gene_ids,
                        encoder_perturb_label= pert_position_gene_ids,
                        encoder_labels=encoder_data_labels,
                        decoder_data=x_query,
                        mask_gene_name=False,
                        mask_labels=None,
                        decoder_position_gene_ids=decoder_position_gene_ids,
                        decoder_data_padding_labels=decoder_data_padding,
                        )     

                masked_positions = torch.ones_like(
                    target_values, dtype=torch.bool
                )  # Use all
                loss = loss_mse = masked_mse_loss(output[:,:-2], 
                                              target_values, masked_positions)

            total_loss += loss.item()
    return total_loss / len(val_loader)



def train_the_model(model,
                    pert_data,
                    save_dir,logger,
                    gene_ids,device,
                    epochs=5,
                    max_seq_len=2000):
    early_stop=2
    best_val_loss = float("inf")

    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    #scaler used in training process

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]
        print(device)
        train(
            model,
            train_loader,
            device,
            max_seq_len,
            gene_ids,
            scaler,
            optimizer,scheduler,logger,epoch
        )
        val_loss = evaluate(
            model,
            valid_loader,
            device,
            max_seq_len,
            gene_ids
        )
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} |"     

        )
        logger.info("-" * 89)
        print(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"Best model with score {best_val_loss:5.4f}")
       

        torch.save(
            model.state_dict(),
            save_dir / f"model_{epoch}.pt",
        )

        scheduler.step()



def main(parser):
    args = parser.parse_args()
    save_dir = Path(f"/data1/yy_data/pert/scfoundation/save/dev_perturb_{args.data_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving to {save_dir}")


    
    logger = setup_logger(save_dir / "run.log")
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # get data
    pert_data = PertData(args.data_dir)
    #load the data processed
    pert_data.load(data_path=pjoin(args.data_dir,args.data_name.lower()))
    pert_data.prepare_split(split = 'simulation', seed = 1, train_gene_set_size=0.75)
    pert_data.get_dataloader(batch_size = args.batch_size, test_batch_size = args.batch_size)
    
    # set up a model
    '''
    os_gene = pd.read_csv('/data/yy_data/sc_model/scFoundation-main/OS_scRNA_gene_index.19264.tsv', sep='\t')
    os_gene_dict = {}
    for index, row in os_gene.iterrows():
        os_gene_dict[row['gene_name']] = row['index']
    os_gene_dict["<pad>"]=19266 #model.config['seq_len']
    os_gene_dict["<tc>"]=19265

    #check for genes
    genes = pert_data.adata.var["gene_name"].tolist()
    gene_ids = np.array(
    [os_gene_dict[gene] if gene in os_gene_dict else os_gene_dict["<pad>"] for gene in genes], dtype=int
)

    tc_value = np.array([os_gene_dict["<tc>"]], dtype=int)
    gene_ids = np.concatenate((gene_ids, tc_value))    # model initialize and load pretrained weights
    '''

    gene_ids= pert_data.adata.var['gene_name'].tolist()

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path=pjoin(args.model_dir,'models/models.ckpt')
    model,model_state_dict = load_model(ckpt_path)
    model.load_state_dict(model_state_dict,strict=False)
    
    
    model.to(device)

    #train and save the model
    max_seq_len=args.max_seq_len
    train_the_model(model,epochs=args.epochs,
                    pert_data=pert_data,
                    save_dir=save_dir,
                    logger=logger,
                    max_seq_len=max_seq_len,
                    gene_ids=gene_ids,
                    device=device)
    
    # Load best model and run predictions
    weight_dir_pattern = pjoin(save_dir, 'model_*.pt')
    weight_files = glob.glob(weight_dir_pattern)
    
    if weight_files:
        # Find the best model (assuming model_2.pt pattern from original code)
        weight_file = None
        for file in weight_files:
            if '_2' in file:
                weight_file = file
        
        if not weight_file and weight_files:
            # If no _2 model, use the last epoch
            weight_file = sorted(weight_files)[-1]
        
        if weight_file:
            logger.info(f"Loading weights from {weight_file} for prediction")
            model_weights = torch.load(weight_file, map_location=device)
            model.load_state_dict(model_weights)
            
            # Run predictions on all subgroups
            predict_all_subgroups(model, pert_data, args.data_name, save_dir, device, args.batch_size)
        else:
            logger.warning("No suitable weight file found for prediction")
    else:
        logger.warning("No weight files found for prediction")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scfoundation')

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--data_name', type=str, default='norman')
    parser.add_argument('--model_dir', type=str, default='/data/yy_data/sc_model/scFoundation-main/model')
    parser.add_argument('--max_seq_len', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)

    main(parser)