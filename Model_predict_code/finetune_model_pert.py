import sys 
import numpy as np
import torch
from torch import nn
sys.path.append("/data/yy_data/sc_model/scFoundation-main")
from load import *
from modules.encoders import *
from modules.mae_autobin import AutoDiscretizationEmbedding2
from modules.performer_module import PerformerModule
from modules.transformer import pytorchTransformerModule
from torch.nn import functional as F
import pandas as pd
from typing import Iterable, List, Tuple, Dict, Union, Optional


def denormalize(data):

    data_nonzero = torch.where(data == 0, torch.tensor(1e-8, device=data.device), data)
    min_values_y, _ = torch.min(data_nonzero, dim=1, keepdim=True)
    min_values_y = torch.clamp(min_values_y, min=1e-8) 
    data = data / min_values_y
    return data

def exists(val):
    return val is not None

class MaeAutobin_pert(nn.Module):
    def __init__(
            self,
            config,
            *,
            #num_tokens,  # num of tokens
            #max_seq_len,  # max length of sequence
            #embed_dim,  # encoder dim of tokens
            #decoder_embed_dim,
            tie_embed=False,
            bin_alpha = 1.0,
            bin_num = 10,
            pad_token_id = None,
            mask_token_id = None,
    ):
        super(MaeAutobin_pert, self).__init__()
        self.config=config
        model_type = self.config["model_type"]        
        max_seq_len = self.config['seq_len']
        self.max_seq_len=max_seq_len
        num_tokens = self.config['n_class']
        pad_token_id = self.config['pad_token_id']
        mask_token_id = self.config['mask_token_id']
        embed_dim= self.config['encoder']['hidden_dim'] 
        decoder_embed_dim= self.config['decoder']['hidden_dim']
        bin_num= self.config['bin_num']

        encoder_config = self.config['encoder']
        decoder_config = self.config['decoder']
        encoder = select_module(self.config, encoder_config, self.config['encoder']['module_type'])
        decoder = select_module(self.config, decoder_config, self.config['decoder']['module_type'])

        # encoder
        self.token_emb = AutoDiscretizationEmbedding2(embed_dim, max_seq_len, bin_num=bin_num, bin_alpha=bin_alpha, pad_token_id=pad_token_id, mask_token_id=mask_token_id)
        self.pos_emb = nn.Embedding(max_seq_len+1, embed_dim)  #RandomPositionalEmbedding(embed_dim, max_seq_len)
        #self.pert_emb = nn.Embedding(max_seq_len+1, embed_dim)  
        self.pert_emb = nn.Embedding(3, embed_dim)

        # ## DEBUG
        self.encoder = encoder

        ##### decoder
        self.decoder = decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.to_final = nn.Linear(decoder_embed_dim, 1)

    def forward(self, x, padding_label, encoder_position_gene_ids, encoder_labels, decoder_data,encoder_perturb_label,
                decoder_position_gene_ids, decoder_data_padding_labels=None,
                mask_gene_name=False, mask_labels=None, 
                output_attentions=False,
                decoder_perturb_label=None, perturb_emb=None):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embedding
        x = self.token_emb(torch.unsqueeze(x, 2), output_weight = 0)

        if output_attentions:
            x.requires_grad_()  # used for attn_map output

        position_emb = self.pos_emb(encoder_position_gene_ids)
        pert_emb = self.pert_emb(encoder_perturb_label)

        x += position_emb
        x += pert_emb
        x = self.encoder(x, padding_mask=padding_label)

        decoder_data = self.token_emb(torch.unsqueeze(decoder_data, 2))
        position_emb = self.pos_emb(decoder_position_gene_ids)
        if mask_gene_name:
            # todo
            # mask gene_name
            print('mask_gene_name not done')
            exit(0)
        batch_idx, gen_idx = (encoder_labels == True).nonzero(as_tuple=True)
        decoder_data[batch_idx, gen_idx] = x[~padding_label].to(decoder_data.dtype)

        decoder_data += position_emb
        #decoder_data += pert_emb

        decoder_data = self.decoder_embed(decoder_data)
        x = self.decoder(decoder_data, padding_mask=decoder_data_padding_labels)

        # print("x0",x.shape) 
        x = self.norm(x)
        # print("x1",x.shape) 
        if exists(self.to_final):
            x = self.to_final(x)
            return x.squeeze(2) 
        else:
            return x
        
    def pred_perturb(self, batch_data,
                     gene_ids, ref_gene_ids,pad_token_id,seq_len,amp):
        
        self.eval()
        device = next(self.parameters()).device

        batch_size = len(batch_data.pert_idx)

        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        n_genes=int(len(x)/batch_size)-1
        ori_gene_values = x.view(batch_size, n_genes+1).to(device)
        obs_counts = ori_gene_values[:,-1].unsqueeze(1)

        ori_gene_values = ori_gene_values[:,:-1]
        


        x_query=denormalize(ori_gene_values)
        x_query=reorder_matrix_by_ref(x_query, gene_ids, ref_gene_ids)
        
        pert_flags=torch.zeros_like(x_query).long().to(device)
        for idx, i in enumerate(batch_data.pert_idx):
                for j in i:
                    if  j != -1:
                        j=get_ref_indices_from_gene_indices(ref_gene_ids, gene_ids, [j])
                        pert_flags[idx, j[0]] = 1

        input_gene_ids = torch.arange(19264, device=device, dtype=torch.long).to(device)
        input_full_ids = torch.arange(19264+2, device=device, dtype=torch.long).to(device)
            
        total_counts=torch.log10(obs_counts).to(device)
      
        x_query=torch.cat((x_query, total_counts,
                           total_counts), dim=1).to(device, dtype=torch.float32)
        pert_flags = torch.cat((pert_flags, torch.zeros((pert_flags.size(0), 1)).to(device),
                                torch.zeros((pert_flags.size(0), 1)).to(device)), dim=1)            
        input_gene_ids = torch.cat((input_gene_ids, torch.tensor([19264], device=device),
                                    torch.tensor([19264+1], device=device)), dim=0)
        #(input_gene_ids.shape)
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

        with torch.cuda.amp.autocast(enabled=amp):
            output = self(x=encoder_data, padding_label=encoder_data_padding,
                        encoder_position_gene_ids=encoder_position_gene_ids,
                        encoder_perturb_label= pert_position_gene_ids,
                        encoder_labels=encoder_data_labels,
                        decoder_data=x_query,
                        mask_gene_name=False,
                        mask_labels=None,
                        decoder_position_gene_ids=decoder_position_gene_ids,
                        decoder_data_padding_labels=decoder_data_padding,
                        )   
        pred_gene_values=output[:,:-2]
        return pred_gene_values
 

def masked_mse_loss(input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    mask_sum = mask.sum()
    if mask_sum == 0:
        return torch.tensor(0.0, device=input.device, requires_grad=True)
    return loss / mask_sum


def map_raw_id_to_pretrained_id(
    raw_ids: Union[np.ndarray, torch.Tensor],
    gene_ids: np.ndarray,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Map some raw ids which are indices of the raw gene names to the indices of the

    Args:
        raw_ids: the raw ids to map
        gene_ids: the gene ids to map to
    """
    if isinstance(raw_ids, torch.Tensor):
        device = raw_ids.device
        dtype = raw_ids.dtype
        return_pt = True
        raw_ids = raw_ids.cpu().numpy()
    elif isinstance(raw_ids, np.ndarray):
        return_pt = False
        dtype = raw_ids.dtype
    else:
        raise ValueError(f"raw_ids must be either torch.Tensor or np.ndarray.")

    if raw_ids.ndim != 1:
        raise ValueError(f"raw_ids must be 1d, got {raw_ids.ndim}d.")

    if gene_ids.ndim != 1:
        raise ValueError(f"gene_ids must be 1d, got {gene_ids.ndim}d.")

    mapped_ids: np.ndarray = gene_ids[raw_ids]
    assert mapped_ids.shape == raw_ids.shape
    if return_pt:
        return torch.from_numpy(mapped_ids).type(dtype).to(device)
    return mapped_ids.astype(dtype)



def getData(data, data_raw, pad_token_id, pert_flags):
    """
    """

    decoder_data = data.clone().detach()
    decoder_data_padding = torch.full_like(data, False, dtype=torch.bool).to(data.device)


    encoder_data_labels = data_raw > 0
    encoder_data, encoder_data_padding = safe_gatherData(decoder_data, encoder_data_labels,
                                                    config['pad_token_id'])
    data_gene_ids = torch.arange(data.shape[1], device=data.device).repeat(data.shape[0], 1)
    encoder_position_gene_ids, _ = safe_gatherData(data_gene_ids, encoder_data_labels,
                                                config['pad_token_id'])
    
    #bool_tensor = pert_flags != 0
    pert_position_gene_ids, _ = safe_gatherData(pert_flags.long(), encoder_data_labels,
                                                config['pad_token_id'])

    decoder_position_gene_ids = data_gene_ids
    data_mask_labels = None

    encoder_position_gene_ids[encoder_data_padding] = config["seq_len"]
    decoder_position_gene_ids[decoder_data_padding] = config["seq_len"]
    pert_position_gene_ids[encoder_data_padding] = config["seq_len"]

    return encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_data_labels, decoder_data, decoder_data_padding,  data_mask_labels, decoder_position_gene_ids,pert_position_gene_ids

def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()
