import argparse
import os
from collections import OrderedDict

import torch
import pandas as pd

from torch.utils.data import DataLoader

from models.datasets import MicroInferDataset
from models.utils import (
    get_logger,
    convert_to_bert_config,
    kv_args
)
from models.models import BertWithoutSegEmb
from my_config import BaseConfig
from train_tokenizer import get_tokenizer
from train_encoder_simcse import get_default_config


def get_dataloader(config: BaseConfig):
    # load tokenizer
    tokenizer, tknz_cfg = get_tokenizer(config.tokenizer, train=False)
    if config.tokenizer.truncate:
        tokenizer.enable_truncation(config.tokenizer.max_length)
    special_tokens = {k: v['id'] for k, v in tknz_cfg['special_tokens'].items()}
    # load data
    data_df = pd.read_csv(config.dataset, sep='\t', usecols=['idx', config.text])
    data_set = MicroInferDataset(
        data_df['idx'].tolist(),
        data_df[config.text].tolist(),
        tokenizer, 
        special_tokens,
        tknz_cfg['vocab_size']
    )
    loader = DataLoader(
        data_set,
        collate_fn=data_set.collate_fn,
        **config.loader
    )
    n_ent = max(data_df['idx']) + 1
    return loader, n_ent

def main(config):
    if not os.path.exists(os.path.dirname(config.save_path)):
        os.makedirs(os.path.dirname(config.save_path))

    logger = get_logger('Infer with Encoder SimCSE', os.path.join(os.path.dirname(config.save_path), 'infer.log'))

    if config.gpu < 0 or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = f"cuda:{config.gpu}"
    
    logger.info("Building datasets.")
    loader, n_ent = get_dataloader(config)
    embeddings_dict = {}

    # build model
    pretrain_cfg_path = os.path.join(
        os.path.dirname(config.pretrain_model_path),
        'configs.yml'
    )
    pretrain_cfg = get_default_config(pretrain_cfg_path)
    model_cfg = pretrain_cfg.model
    if model_cfg.vocab_size is None:
        model_cfg.vocab_size = loader.dataset.vocab_size
    model_cfg = convert_to_bert_config(model_cfg)
    model = BertWithoutSegEmb(model_cfg)
    model.to(device)
    logger.info(model)
    # load pre-trained weights
    ckpt = torch.load(config.pretrain_model_path, map_location=device)
    mismatch = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    logger.info(f"Missing keys: {mismatch.missing_keys}")
    logger.info(f"Unexpected keys: {mismatch.unexpected_keys}")

    n_dim = None
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            idx_list = batch["sample_indices"]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            last_hidden_state = outputs.last_hidden_state
            cls_embeddings = last_hidden_state[:, 0, :].cpu()
            n_dim = cls_embeddings.shape[1]
            for idx_val, emb in zip(idx_list, cls_embeddings):
                embeddings_dict[idx_val] = emb
    
    embedding_matrix = torch.zeros((n_ent, n_dim), dtype=torch.float32)
    for idx, emb in embeddings_dict.items():
        embedding_matrix[idx] = emb
    torch.save(embedding_matrix, config.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Infer with AA/SMILES encoder")
    parser.add_argument("config", type=str, help="config path")
    parser.add_argument("-u", "--update", type=kv_args, nargs='*', help="path.to.config=config_value")
    args = parser.parse_args()

    config = get_default_config(args.config)
    if args.update is not None:
        for k, v in args.update:
            config.set_config_via_path(k, v)

    main(config)
