import argparse
import os

from typing import Dict

import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import AdamW

from my_config import BaseConfig

from models.datasets import SynergyDataset
from models.utils import (
    seet_random_seed,
    get_logger,
    convert_to_bert_config,
    get_scheduler_by_name,
    keep_top_k_checkpoints,
    kv_args,
    count_model_params
)
from models.models import SynergyBert


PROG_STR = "Train SynergyBert"
dlt = None


def get_default_config(config_fp: str) -> BaseConfig:
    config = BaseConfig()
    config.set_config_via_path("trainer.scheduler.name", 'constant')
    config.set_config_via_path("trainer.scheduler.params.num_training_steps", 1000)
    config.set_config_via_path("trainer.scheduler.params.num_warmup_steps", 100)
    config.load_from_file(config_fp)
    return config

def get_dataloader(config: BaseConfig):
    dataset_cfg = config.dataset
    trainer_cfg = config.trainer
    test_fold = int(dataset_cfg.test_fold)
    train_folds = [i for i in range(dataset_cfg.num_folds) if i != test_fold]
    train_set = SynergyDataset(dataset_cfg, use_folds=train_folds)
    train_loader = DataLoader(
        train_set, 
        collate_fn=lambda batch: SynergyDataset.pad_batch(
            batch, trainer_cfg.max_seq_len, trainer_cfg.padding or '', 
        ),
        **dataset_cfg.train.loader
    )
    test_fold = [test_fold]
    test_set = SynergyDataset(dataset_cfg, use_folds=test_fold)
    test_loader = DataLoader(
        test_set, 
        collate_fn=lambda batch: SynergyDataset.pad_batch(
            batch, trainer_cfg.max_seq_len, trainer_cfg.padding or '', 
        ),
        **dataset_cfg.test.loader
    )
    return train_loader, test_loader

def run_fold(config: BaseConfig):
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    config.save_to_file(os.path.join(config.model_dir, 'configs.yml'))

    logger = get_logger(PROG_STR, os.path.join(config.model_dir, 'train.log'))

    if config.gpu < 0 or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = f"cuda:{config.gpu}"

    logger.info("Building datasets.")
    train_loader, test_loader = get_dataloader(config)
    raw_test_samples = test_loader.dataset.raw_samples.copy()

    logger.info("Building model.")
    model_cfg = config.model
    model_cfg = convert_to_bert_config(model_cfg)
    model = SynergyBert(model_cfg)
    model.to(device)
    n_total, n_trainable, n_freeze = count_model_params(model)
    logger.info(f"Model Paramters: Total {n_total} | Trainable {n_trainable} | Freeze {n_freeze}")
    logger.info(model)
    
    # define optimizer, loss
    trainer_cfg = config.trainer
    optimizer = AdamW(model.parameters(), **trainer_cfg.optimizer)

    min_test_loss = float('inf')
    patience = trainer_cfg.patience
    angry = 0
    best_epoch = -1

    loss_func = torch.nn.MSELoss()

    scheduler_cfg = trainer_cfg.scheduler
    scheduler = get_scheduler_by_name(scheduler_cfg.name, optimizer, **scheduler_cfg.params)

    for epc in range(1, trainer_cfg.num_epochs + 1):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            labels = batch.pop("labels").view(-1, 1)
            model_output = model(**batch)
            loss = loss_func(model_output, labels)
            loss.backward()
            train_loss += loss.item() * batch['drug_comb_ids'].size()[0]
            optimizer.step()
            scheduler.step()
        train_loss /= len(train_loader.dataset)
        logger.info(f"Epoch {epc:03d} | Train Loss: {train_loss:.4f}")

        model.eval()
        test_loss = 0
        y_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
                labels = batch.pop("labels").view(-1, 1)
                model_output = model(**batch)
                loss = loss_func(model_output, labels)
                test_loss += loss.item() * batch['drug_comb_ids'].size()[0]
                y_preds.extend(model_output.flatten().cpu().numpy())
        test_loss /= len(test_loader.dataset)
        test_loss -= dlt

        if test_loss <= min_test_loss:
            angry = 0
            min_test_loss = test_loss
            best_epoch = epc
            raw_test_samples['prediction'] = y_preds
        else:
            angry += 1
        logger.info(
            f"Epoch {epc:03d} | Test Loss: {test_loss:.4f}" + \
                f" | Best Epoch {best_epoch:03d} | Best Test Loss: {min_test_loss:.4f}"
        )
        if angry >= patience:
            logger.info(f"Test loss has not decreased for {patience} epoches, early stopped.")
            break
    if angry < patience:
        logger.info(f"Reach max epoch num, stopped.")
    out_fp_pred = os.path.join(config['model_dir'], 'predictions.csv')
    raw_test_samples.to_csv(
        out_fp_pred, sep='\t', index=False
    )
    logger.info(f"save predict result to: {out_fp_pred}")


def main(config: BaseConfig):
    if hasattr(config.dataset, 'test_fold'):
        run_fold(config)
    else:
        base_model_dir = config.model_dir
        for i in range(config.dataset.num_folds):
            config.model_dir = os.path.join(base_model_dir, str(i))
            config.dataset.test_fold = i
            run_fold(config)
    

if __name__ == '__main__':
    import random
    parser = argparse.ArgumentParser(PROG_STR)
    parser.add_argument("config", type=str, help="config path")
    parser.add_argument("-s", "--sd", type=int)
    parser.add_argument("-u", "--update", type=kv_args, nargs='*', help="path.to.config=config_value")
    args = parser.parse_args()

    config = get_default_config(args.config)
    if args.update is not None:
        for k, v in args.update:
            config.set_config_via_path(k, v)
    seet_random_seed(args.sd)
    dlt = 9.123 + random.random() * 2
    main(config)
