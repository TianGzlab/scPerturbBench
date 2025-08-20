import json
import os
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

import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import masked_mse_loss
from scgpt.tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id

from gears import PertData, GEARS
from os.path import join as pjoin
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader
from gears.utils import create_cell_graph_dataset_for_prediction
from typing import Dict

# set seed 
set_seed(222)


def load_model(model_config_file,vocab):
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
    ntokens = len(vocab)  # size of vocabulary

    dropout = 0.2 
    MVC = False  # Masked value prediction for cell embedding
    cell_emb_style = "cls"
    mvc_decoder_style = "inner product, detach"
    use_fast_transformer = True  # whether to use fast transformer
    pad_token = "<pad>"
    #pad_value = 0  # for padding values
    pert_pad_id = 0

    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=vocab['<pad>'],
        pert_pad_id=pert_pad_id,
        do_mvc=MVC,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        use_fast_transformer=use_fast_transformer,
    )
    return model

def predict(model: TransformerGenerator, pert_list, pert_data, gene_ids, batch_size=64) -> Dict:
    """
    Predict the gene expression values for the given perturbations.
    """
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    pool_size = len(ctrl_adata.obs)
    include_zero_gene = 'all'
    gene_list = pert_data.adata.var['gene_name'].tolist()
    amp = True
    
    for pert in pert_list:
        for i in pert:
            if i not in gene_list:
                raise ValueError(
                    "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                )
   
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        results_pred = {}
        for pert in pert_list:
            print(pert)
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )            

            loader = DataLoader(cell_graphs, batch_size=batch_size, shuffle=False)

            preds = []
            for batch_data in loader:
                pred_gene_values = model.pred_perturb(
                    batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = preds

    return results_pred

def predict_all_subgroups(model, pert_data, gene_ids, model_dir, data_name, batch_size=32):
    """
    Unified prediction function that handles all subgroups.
    """
    save_dir = Path('/data1/yy_data/pert/scgpt/save')
    results_dir = save_dir.joinpath(f'dev_perturb_{data_name}', 'results_unified')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    var_names_compressed_path = results_dir.joinpath('gene_names.csv.gz')
    pd.DataFrame(pert_data.adata.var["gene_name"]).to_csv(var_names_compressed_path, index=False, compression='gzip')
    
    # Get all subgroups
    subgroups = {
        'unseen_single': pert_data.subgroup['test_subgroup']['unseen_single'],
        'combo_seen0': pert_data.subgroup['test_subgroup']['combo_seen0'],
        'combo_seen1': pert_data.subgroup['test_subgroup']['combo_seen1'],
        'combo_seen2': pert_data.subgroup['test_subgroup']['combo_seen2']
    }
    
    # Combine all queries
    all_queries = []
    for subgroup_name, queries in subgroups.items():
        all_queries.extend(queries)
    
    for query in all_queries:
        print(f"Processing: {query}")
        
        if query.split("+")[1] == "ctrl":
            pred = predict(model, [[query.split("+")[0]]], pert_data, gene_ids, batch_size)
            pred_result = pred[query.split("+")[0]]
        elif query.split("+")[0] == "ctrl":
            pred = predict(model, [[query.split("+")[1]]], pert_data, gene_ids, batch_size)
            pred_result = pred[query.split("+")[1]]
        else:
            pred = predict(model, [query.split("+")], pert_data, gene_ids, batch_size)
            pred_result = pred["_".join(query.split("+"))]
        
        pred_compressed_path = results_dir.joinpath(f'{query}.npy.gz')
        np.savez_compressed(pred_compressed_path, pred=pred_result.cpu())
    
    print(f"All predictions saved to {results_dir}")

def load_pretrained(model,model_file,logger):
    load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
    ]
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if any([k.startswith(prefix) for prefix in load_param_prefixs])
    }
    for k, v in pretrained_dict.items():
        logger.info(f"Loading params {k} with shape {v.shape}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
# 
def train(model: nn.Module, 
          train_loader: torch.utils.data.DataLoader,
          device,max_seq_len,
          gene_ids,scaler,optimizer,scheduler,logger,epoch) -> None:
    """
    Train the model for one epoch.
    """
    CLS = False  # celltype classification objective
    CCE = False  # Contrastive cell embedding objective
    MVC = False  # Masked value prediction for cell embedding
    ECS = False
    log_interval = 100

    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.pert_idx)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        n_genes=int(len(x)/batch_size)
        ori_gene_values = x.view(batch_size, n_genes).to(device)
        target_gene_values = batch_data.y.view(batch_size, n_genes).to(device)


        #need to remove last column
        pert_flags=torch.zeros_like(ori_gene_values).long().to(device)
        for i, idx in enumerate(batch_data.pert_idx):
                if idx!=([-1]):
                    pert_flags[i, idx] = 1

        input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long).to(device)

        # sample input_gene_id
        if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]
        target_values = target_gene_values[:, input_gene_ids]

        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

        # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
        src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )
        with torch.cuda.amp.autocast(enabled=True):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = masked_mse_loss(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
 
        scaler.step(optimizer)
        scaler.update()

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
             device,max_seq_len,gene_ids,
             ) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    CLS = False  # celltype classification objective
    CCE = False  # Contrastive cell embedding objective
    MVC = False  # Masked value prediction for cell embedding
    ECS = False
    model.eval()

    total_loss = 0.0
    total_loss_noz = 0.0
    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_size = len(batch_data.pert_idx)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            n_genes=int(len(x)/batch_size)
            ori_gene_values = x.view(batch_size, n_genes).to(device)
            target_gene_values = batch_data.y.view(batch_size, n_genes).to(device)

            #need to remove last column 
            pert_flags=torch.zeros_like(ori_gene_values).long().to(device)
            for i, idx in enumerate(batch_data.pert_idx):
                if idx!=([-1]):
                    pert_flags[i, idx] = 1

            input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long).to(device)

            # sample input_gene_id
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
                )
        
            with torch.cuda.amp.autocast(enabled=True):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=True,
                )
                output_values = output_dict["mlm_output"]
                

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
                loss  = masked_mse_loss(output_values, target_values, masked_positions)
                masked_no_zeros = torch.zeros_like(input_values, dtype=torch.bool,device=input_values.device)
                masked_no_zeros[target_values!=0] = True
                loss_noz  = masked_mse_loss (output_values, target_values, masked_no_zeros)

            total_loss += loss.item()
            total_loss_noz+=loss_noz.item()
    return total_loss / len(val_loader), total_loss_noz/len(val_loader)



def train_the_model(model,
                    pert_data,
                    save_dir,logger,
                    gene_ids,device,
                    epochs=5,
                    max_seq_len=2000):
    early_stop=5
    best_val_loss = float("inf")
    patience = 0
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    #scaler used in training process

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]

        train(
            model,
            train_loader,
            device,
            max_seq_len,
            gene_ids,
            scaler,
            optimizer,scheduler,logger,epoch
        )
        val_loss, val_loss_noz = evaluate(
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
            f"valid no zero loss/mse {val_loss_noz:5.4f} |"  

        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"Best model with score {best_val_loss:5.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                logger.info(f"Early stop at epoch {epoch}")
                break

        scheduler.step()
    torch.save(
            model.state_dict(),
            save_dir / f"model_{epoch}.pt",
        )
    logger.info("Training complete. Final model saved.")



def main(parser):
    args = parser.parse_args()
    save_dir = Path(f"/data1/yy_data/pert/scgpt/save/dev_perturb_{args.data_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving to {save_dir}")


    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    # log running date 
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")


    # get data
    pert_data = PertData(args.data_dir)
    #load the data processed
    pert_data.load(data_path=pjoin(args.data_dir,args.data_name.lower()))
    pert_data.prepare_split(split = 'simulation', seed = 1, train_gene_set_size=0.75)
    pert_data.get_dataloader(batch_size = args.batch_size, test_batch_size = args.batch_size)
    
    # set up a model
    model_config_file = pjoin(args.model_dir , "args.json")
    model_file = pjoin(args.model_dir , "best_model.pt")
    vocab_file = pjoin(args.model_dir , "vocab.json")

    #check for genes
    vocab = GeneVocab.from_file(vocab_file)
    genes = pert_data.adata.var["gene_name"].tolist()
    gene_ids = np.array(
    [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
)

    # model initialize
    device=args.device
    model = load_model(model_config_file,vocab)
    model.to(device)
    # load pretrained weights
    model=load_pretrained(model,model_file,logger=logger)

    #train and save the model
    max_seq_len=args.max_seq_len
    train_the_model(model,epochs=args.epochs,
                    pert_data=pert_data,
                    save_dir=save_dir,
                    logger=logger,
                    max_seq_len=max_seq_len,
                    gene_ids=gene_ids,
                    device=device)
    
    # Run predictions if requested
    if args.run_prediction:
        logger.info("Starting prediction on all subgroups...")
        # Load the final trained model
        final_model_path = save_dir / f"model_{args.epochs}.pt"
        if final_model_path.exists():
            model_weights = torch.load(final_model_path, map_location=device)
            model.load_state_dict(model_weights)
            logger.info(f"Loaded trained model from {final_model_path}")
        
        predict_all_subgroups(model, pert_data, gene_ids, args.model_dir, args.data_name, args.batch_size)
        logger.info("Prediction completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scGPT')

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--data_name', type=str, default='norman')
    parser.add_argument('--model_dir', type=str, default='/data/yy_data/sc_model/scGPT/save/scGPT_human')
    parser.add_argument('--max_seq_len', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--run_prediction', action='store_true', help='Run prediction on all subgroups after training')

    main(parser)