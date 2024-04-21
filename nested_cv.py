import argparse
import os
import json

from typing import Dict, List
from copy import deepcopy
from multiprocessing import set_start_method, Pool, Manager, Queue

import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW

from my_config import BaseConfig

from models.datasets import SynergyDataset
from models.utils import (
    seet_random_seed,
    get_logger,
    set_log,
    queue_log,
    close_log,
    convert_to_bert_config,
    get_scheduler_by_name,
    kv_args,
    random_split_indices
)
from models.models import SynergyBert


PROG_STR = "Nested CV"
logger = get_logger(PROG_STR, no_handler=True)

def initialize_worker(log_queue):
    queue_log(log_queue)

def get_default_config(config_fp: str) -> BaseConfig:
    config = BaseConfig()
    config.set_config_via_path("trainer.scheduler.name", 'constant')
    config.set_config_via_path("trainer.scheduler.params.num_training_steps", 1000)
    config.set_config_via_path("trainer.scheduler.params.num_warmup_steps", 100)
    config.load_from_file(config_fp)
    return config

def get_dataloader(config: BaseConfig):
    dataset_cfg = config.dataset
    train_folds = dataset_cfg.train_folds
    valid_fold = dataset_cfg.get('valid_fold', None)
    test_fold = int(dataset_cfg.test_fold)
    train_set = SynergyDataset(dataset_cfg, use_folds=train_folds)
    if valid_fold is None:
        tr_indices, es_indices = random_split_indices(train_set, test_rate=0.1)
        valid_set = Subset(train_set, es_indices)
        train_set = Subset(train_set, tr_indices)
    else:
        valid_set = SynergyDataset(dataset_cfg, use_folds=[valid_fold])
    test_set = SynergyDataset(dataset_cfg, use_folds=[test_fold])

    train_loader = DataLoader(
        train_set, 
        collate_fn=SynergyDataset.pad_batch,
        **dataset_cfg.train.loader
    )
    valid_loader = DataLoader(
        valid_set,
        collate_fn=SynergyDataset.pad_batch,
        **dataset_cfg.valid.loader
    )
    test_loader = DataLoader(
        test_set, 
        collate_fn=SynergyDataset.pad_batch,
        **dataset_cfg.test.loader
    )
    return train_loader, valid_loader, test_loader

def run_fold(config: BaseConfig, save_result: bool = False, watch: str = 'valid'):
    device = f"cuda:{config.gpu}"
    train_loader, valid_loader, test_loader = get_dataloader(config)
    raw_test_samples = test_loader.dataset.raw_samples.copy()
    
    model_cfg = config.model
    model_cfg = convert_to_bert_config(model_cfg)
    model = SynergyBert(model_cfg)
    model.to(device)
    try:
        model = torch.compile(model)
    except Exception as e:
        # logger.warning(f"To use torch.compile, torch 2.0 is needed. Current version is {torch.__version__}")
        pass

    # define optimizer, loss
    trainer_cfg = config.trainer
    optimizer = AdamW(model.parameters(), **trainer_cfg.optimizer)

    min_valid_loss = float('inf')
    min_valid_loss = float('inf')
    min_test_loss = float('inf')
    min_test_loss = float('inf')
    patience = trainer_cfg.patience
    valid_angry = 0
    test_angry = 0

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

        current_valid_best = False
        model.eval()

        update_valid = valid_angry <= patience
        # validation
        if update_valid:
            valid_loss = 0
            with torch.no_grad():
                for batch in valid_loader:
                    batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
                    labels = batch.pop("labels").view(-1, 1)
                    model_output = model(**batch)
                    loss = loss_func(model_output, labels)
                    valid_loss += loss.item() * batch['drug_comb_ids'].size()[0]
            valid_loss /= len(valid_loader.dataset)
            if update_valid and valid_loss <= min_valid_loss:
                valid_angry = 0
                min_valid_loss = valid_loss
                current_valid_best = True
            else:
                valid_angry += 1

        update_test = (watch == 'valid' and current_valid_best) or (watch != 'valid' and test_angry <= patience)
        # test
        if update_test:
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
            if watch == 'valid' or test_loss <= min_test_loss:
                test_angry = 0
                min_test_loss = test_loss
                raw_test_samples['prediction'] = y_preds
            else:
                test_angry += 1

        if watch == 'valid' and valid_angry > patience:
            break
        elif watch == 'both' and valid_angry > patience and test_angry > patience:
            break

    if save_result:
        test_fold = config.dataset.test_fold
        out_fp_pred = os.path.join(config.model_dir, f'predictions-{test_fold}.csv')
        raw_test_samples.to_csv(
            out_fp_pred, sep='\t', index=False
        )
        logger.info(f"save predict result of test fold {test_fold} to: {out_fp_pred}")
    
    return min_valid_loss, min_test_loss

def run_inner_task(config, gpu_queue, result_queue, hpc_index):
    gpu = gpu_queue.get()
    train_folds = config.dataset.train_folds 
    valid_fold = config.dataset.valid_fold
    test_fold = config.dataset.test_fold
    logger.info(f"start training on {train_folds} with hyperparam combs {hpc_index}")
    config.gpu = gpu
    valid_res, test_res = run_fold(config, False, 'both')
    logger.info(f"finish training on {train_folds} with hyperparam combs {hpc_index}")
    logger.info(
        f"result on fold {valid_fold}: {valid_res}, result on fold {test_fold}: {test_res}"
    )
    result_queue.put((valid_fold, hpc_index, valid_res))
    result_queue.put((test_fold, hpc_index, test_res))
    gpu_queue.put(gpu)

def run_outer_fold(config, gpu_queue, result_queue, hpc_index):
    gpu = gpu_queue.get()
    config.gpu = gpu
    train_folds = config.dataset.train_folds
    test_fold = config.dataset.test_fold
    logger.info(f"start training on {train_folds} with hyperparam combs {hpc_index}")
    _, test_res = run_fold(config, True, 'valid')
    logger.info(f"finish training on {train_folds} with hyperparam combs {hpc_index}")
    # logger.info(f"result on fold {test_fold}: {test_res}")
    result_queue.put((test_fold, test_res))
    gpu_queue.put(gpu)

def get_ncv_hps(candidate_hps: List[Dict[str, List]]):
    def dynamic_loop(data: List, cur_row: int, tmp_res: List, final_res: List):
        if cur_row < len(data):
            for elem in data[cur_row]:
                tmp_res.append(elem)
                dynamic_loop(data, cur_row+1, tmp_res, final_res)
                tmp_res.pop()
        else:
            final_res.append(tmp_res[::])

    hp_names = []
    all_hp_values = []
    loop_input = []
    for cand_hp in candidate_hps:
        # type checking
        if isinstance(cand_hp['names'], list):
            for name in cand_hp['names']:
                assert isinstance(name, str), f"names should be list of str, got\n" \
                    f"{cand_hp['names']}"
            for values in cand_hp['values']:
                assert isinstance(values, list), f"values should be list, got\n" \
                    f"{values}"
                assert len(cand_hp['names']) == len(values),\
                    "names and values should have the same length, got\n" \
                    f"names: {cand_hp['names']}\nvalues: {values}"
        hp_names.append(cand_hp['names'])
        loop_input.append(cand_hp['values'])
    dynamic_loop(loop_input, 0, [], all_hp_values)
    for hp_values in all_hp_values:
        new_hp_conf = []
        for hp_name, hp_value in zip(hp_names, hp_values):
            if isinstance(hp_name, list):
                for n, v in zip(hp_name, hp_value):
                    new_hp_conf.append((n, v))
            else:
                new_hp_conf.append((hp_name, hp_value))
        yield new_hp_conf

def main(config: BaseConfig):
    assert config.task == 'nested_cross_validation'
    # prepare output dir
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    config.save_to_file(os.path.join(config.model_dir, 'config_nested_cv.yml'))
    # prepare args
    ncv_config = config.task_params
    all_hp_combs = {}
    for i, hpc in enumerate(get_ncv_hps(ncv_config['candidate_hps'])):
        all_hp_combs[i] = {}
        for conf_path, conf_val in hpc:
            all_hp_combs[i][conf_path] = conf_val
    with open(os.path.join(config.model_dir, 'hyperparam_combs.json'), 'w') as f:
        json.dump(all_hp_combs, f, indent=4)
    n_folds = len(ncv_config.folds)
    n_hp_combs = len(all_hp_combs)
    n_inner_tasks_per_test = (n_folds - 1) * n_hp_combs
    n_inner_tasks_total = n_inner_tasks_per_test * n_folds
    n_gpus = len(ncv_config.gpus)
    fold2idx = {f: i for i, f in enumerate(ncv_config.folds)}

    hp_comb_results = np.zeros((n_folds, n_hp_combs))
    inner_counter = [0] * n_folds
    inner_tasks_received = set()

    # multiprocess run
    manager = Manager()
    gpu_queue = manager.Queue()
    for gpu in ncv_config.gpus:
        gpu_queue.put(gpu)
    inner_result_queue = manager.Queue()
    outer_result_queue = manager.Queue()
    # config logging
    log_queue = Queue()
    log_fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.model_dir, 'nested_cv.log')
    set_log(log_fp, log_queue)
    queue_log(log_queue)

    # start inner processes
    logger.info("start inner processes")
    with Pool(processes=n_gpus, initializer=initialize_worker, initargs=(log_queue, )) as pool:
        for test_fold in ncv_config.folds:
            outer_train_folds = [x for x in ncv_config.folds if x != test_fold]
            for hpc_idx, new_hp_conf in all_hp_combs.items():
                for valid_fold in outer_train_folds:
                    inner_task = tuple(sorted([test_fold, valid_fold]) + [hpc_idx])
                    if inner_task in inner_tasks_received:
                        continue
                    inner_tasks_received.add(inner_task)
                    inner_train_folds = [x for x in outer_train_folds if x != valid_fold]
                    # update config
                    config_tmp = deepcopy(config)
                    for conf_path, conf_val in new_hp_conf.items():
                        config_tmp.set_config_via_path(conf_path, conf_val)
                    config_tmp.set_config_via_path("dataset.train_folds", inner_train_folds)
                    config_tmp.set_config_via_path("dataset.valid_fold", valid_fold)
                    config_tmp.set_config_via_path("dataset.test_fold", test_fold)
                    pool.apply_async(
                        func=run_inner_task, 
                        args=(config_tmp, gpu_queue, inner_result_queue, hpc_idx)
                    )

        logger.info("waiting inner results")
        for _ in range(n_inner_tasks_total):
            fold, hpc_idx, res = inner_result_queue.get()
            fold_idx = fold2idx[fold]
            hp_comb_results[fold_idx, hpc_idx] += res
            inner_counter[fold_idx] += 1
            if inner_counter[fold_idx] == n_inner_tasks_per_test:
                best_idx = np.argmin(hp_comb_results[fold_idx])
                best_hp_comb = all_hp_combs[best_idx]
                # update config
                config_tmp = deepcopy(config)
                for conf_path, conf_val in best_hp_comb.items():
                    config_tmp.set_config_via_path(conf_path, conf_val)
                train_folds = [x for x in ncv_config.folds if x != fold]
                config_tmp.set_config_via_path("dataset.train_folds", train_folds)
                config_tmp.set_config_via_path("dataset.test_fold", fold)
                pool.apply_async(
                    func=run_outer_fold, 
                    args=(config_tmp, gpu_queue, outer_result_queue, hpc_idx)
                )

        logger.info("waiting final results")
        results = []
        for tf_idx in range(n_folds):
            tid, res = outer_result_queue.get()
            result_line = '='*80 + '\n' + f"{tid} {res}" + '\n' + '='*80
            logger.info(result_line)
    close_log()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(PROG_STR)
    parser.add_argument("config", type=str, help="config path")
    parser.add_argument("-u", "--update", type=kv_args, nargs='*', help="path.to.config=config_value")
    args = parser.parse_args()

    config = get_default_config(args.config)
    if args.update is not None:
        for k, v in args.update:
            config.set_config_via_path(k, v)
    seet_random_seed()
    set_start_method('spawn')
    main(config)
