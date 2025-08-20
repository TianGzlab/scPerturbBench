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
import random

from gears import PertData, GEARS
from os.path import join as pjoin
import pandas as pd
from pathlib import Path
import pickle

# set seed 
def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(222)

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

def get_gene_embed(pert_data):
    with open("/data/yy_data/sc_model/scELMo/pkl/ensem_emb_gpt3.5all_new.pickle", "rb") as fp:
        GPT_3_5_gene_embeddings = pickle.load(fp)
    gene_names= list(pert_data.adata.var['gene_name'].values)
    count_missing = 0
    EMBED_DIM = 1536 # embedding dim from GPT-3.5
    lookup_embed = np.zeros(shape=(len(gene_names),EMBED_DIM))
    for i, gene in enumerate(gene_names):
        if gene in GPT_3_5_gene_embeddings:
            lookup_embed[i,:] = GPT_3_5_gene_embeddings[gene].flatten()
        else:
            count_missing+=1  
    print('missed genes in GPT:')
    print(count_missing)
    return lookup_embed
  


def main(parser):
    args = parser.parse_args()
 
     # get data
    pert_data = PertData(args.data_dir)
    #load the data processed
    pert_data.load(data_path=pjoin(args.data_dir,args.data_name.lower()))
    pert_data.prepare_split(split = 'simulation', seed = 1, train_gene_set_size=0.75)
    pert_data.get_dataloader(batch_size = args.batch_size, test_batch_size = args.batch_size)


    #get gpt embed
    gene_emb=get_gene_embed(pert_data=pert_data)
    #load and train the gears model
    if args.use_gene_emb:
        save_dir = Path(f"/data1/yy_data/pert/scelmo/save2/dev_perturb_{args.data_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"saving to {save_dir}")
        gears_model = GEARS(pert_data, device = args.device, gene_emb = gene_emb)

    else:
        save_dir = Path(f"/data1/yy_data/pert/gears/save2/dev_perturb_{args.data_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"saving to {save_dir}")
        gears_model = GEARS(pert_data, device = args.device, gene_emb = None)



    logger = setup_logger(save_dir / "run.log")
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    epochs=args.epochs
    gears_model.model_initialize(hidden_size = 64)
    gears_model.train(epochs = epochs,logger=logger)
    gears_model.save_model(save_dir)
    
    logger.info("Training completed. Starting prediction on test subgroups...")
    if args.use_gene_emb:
        predict_save_dir = "/data1/yy_data/pert/scelmo/save2"
    else:
        predict_save_dir = "/data1/yy_data/pert/gears/save2"
    
    predict_all_subgroups(args.data_dir, args.data_name, predict_save_dir, args.device, args.batch_size)
    logger.info("Prediction completed.")



def predict_all_subgroups(data_dir, data_name, save_dir, device, batch_size=256):
    pert_data = PertData(data_dir)
    pert_data.load(data_path=pjoin(data_dir, data_name.lower()))
    pert_data.prepare_split(split='simulation', seed=1, train_gene_set_size=0.75)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=batch_size)
    
    gears_model = GEARS(pert_data, device=device, gene_emb=None)
    save_path = pjoin(save_dir, f'dev_perturb_{data_name}')
    gears_model.load_pretrained(save_path, gene_emb=None)
    
    results_dir = Path(save_dir).joinpath(f'dev_perturb_{data_name}', 'results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    subgroups = ['unseen_single', 'combo_seen0', 'combo_seen1', 'combo_seen2']
    
    for subgroup in subgroups:
        if subgroup in pert_data.subgroup['test_subgroup']:
            pert_lists = pert_data.subgroup['test_subgroup'][subgroup]
            
            if subgroup == 'combo_seen1' or subgroup == 'combo_seen2':
                combo_seen1 = pert_data.subgroup['test_subgroup'].get('combo_seen1', [])
                combo_seen2 = pert_data.subgroup['test_subgroup'].get('combo_seen2', [])
                pert_lists = combo_seen1 + combo_seen2
            
            if pert_lists:
                processed_list = []
                for pert in pert_lists:
                    parts = pert.split('+')
                    filtered_parts = [part for part in parts if part != 'ctrl']
                    processed_list.append(filtered_parts)
                
                if processed_list:
                    pred_results = gears_model.predict(processed_list)
                    
                    for key in pred_results.keys():
                        pred_compressed_path = results_dir.joinpath(f'{subgroup}_{key}.npz')
                        pred_array = pred_results[key].cpu().numpy()
                        np.savez_compressed(pred_compressed_path, pred=pred_array)
                    
                    print(f"Completed predictions for {subgroup}: {len(processed_list)} perturbations")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scGPT Training')

    # 原 bash 里设置的默认参数
    parser.add_argument('--device_id', type=int, default=3)
    parser.add_argument('--data_dir', type=str, default='/data2/yue_data/pert/data/')
    parser.add_argument('--model_dir', type=str, default='/data/yy_data/pert/scELMO')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--max_seq_len', type=int, default=5000)
    parser.add_argument('--use_gene_emb', type=str, default='True')
    args = parser.parse_args()
    main(args)