import logging
import os
import re
import random

from typing import List, Tuple, Union, Any, Dict

from my_config import BaseConfig
from logging.handlers import QueueHandler, QueueListener
from multiprocessing.queues import Queue

import torch
import numpy as np

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.trainer_utils import SchedulerType
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
from transformers.models.bert import BertConfig

CHECKPOINT_RECORDS = Tuple[str, Union[int, float]]

_LISENER = None

def seet_random_seed(seed=18):
    torch.manual_seed(seed)            # set seed for CPU
    torch.cuda.manual_seed(seed)       # set seed for current GPU
    torch.cuda.manual_seed_all(seed)   # set seed for all GPU
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_logger(logger_name: str, logging_file: str = None, no_handler=False) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '[%Y-%m-%d %H:%M:%S]')
    if not no_handler:
        if logging_file is not None:
            handler = logging.FileHandler(logging_file)
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(fmt=formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def set_log(logging_file: str, queue: Queue):
    if isinstance(queue, Queue):
        global _LISENER
        if not _LISENER:
            handler = logging.FileHandler(logging_file)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)

            _LISENER = QueueListener(queue, handler)
            _LISENER.start()

def queue_log(queue: Queue):
    if isinstance(queue, Queue):
        logger = logging.getLogger()
        if logger.hasHandlers():
            print('return')
            return
        handler = QueueHandler(queue)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            '[%Y-%m-%d %H:%M:%S]'
        )
        handler.setFormatter(formatter)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

def close_log():
    if _LISENER:
        _LISENER.stop()

def convert_to_bert_config(model_config: BaseConfig) -> BertConfig:
    bert_cfg = BertConfig.from_dict(model_config)
    bert_cfg.is_decoder = False
    bert_cfg.use_cache = False
    return bert_cfg

def get_scheduler_by_name(
    name: str, 
    optimizer: Optimizer,
    **kwargs
) -> LambdaLR:
    """
    https://github.com/huggingface/transformers/blob/04ab5605fbb4ef207b10bf2772d88c53fc242e83/src/transformers/optimization.py#L315
    """
    def _update_kw(k):
        if k in kwargs:
            kw[k] = kwargs[k]

    name = SchedulerType(name)
    scheduler_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    kw = {}
    _update_kw('last_epoch')

    if name == SchedulerType.CONSTANT:
        return scheduler_func(optimizer, **kw)

    # All other schedulers require `num_warmup_steps`
    if 'num_warmup_steps' not in kwargs:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")
    _update_kw('num_warmup_steps')

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return scheduler_func(optimizer, **kw)
    if name == SchedulerType.INVERSE_SQRT:
        _update_kw('timescale')
        return scheduler_func(optimizer, **kw)

    # All other schedulers require `num_training_steps`
    if 'num_training_steps' not in kwargs:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")
    _update_kw('num_training_steps')

    if name == SchedulerType.LINEAR:
        return scheduler_func(optimizer, **kw)
    
    if name == SchedulerType.POLYNOMIAL:
        _update_kw('lr_end')
        _update_kw('power')
        return scheduler_func(optimizer, **kw)

    _update_kw('num_cycles')
    if name == SchedulerType.COSINE:
        return scheduler_func(optimizer, **kw)
    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return scheduler_func(optimizer, **kw)

def remove_files(files: List[str]):
    files_removed = []
    for file in files:
        if os.path.exists(file):
            files_removed.append(file)
            os.remove(file)
    return files_removed

def keep_top_k_checkpoints(
    checkpoints: List[CHECKPOINT_RECORDS],
    k: int,
    cmp: str = 'max'
) -> List[CHECKPOINT_RECORDS]:
    assert cmp in ('min', 'max')
    assert k > 0
    reverse_ = cmp == 'max'
    ckpts_sorted = sorted(checkpoints, key=lambda x: x[1], reverse=reverse_)
    top_k_ckpts = set(ckpts_sorted[:k])
    ckpts_expired = []
    ckpts_keeped = []
    for ckpt in checkpoints:
        if ckpt in top_k_ckpts:
            ckpts_keeped.append(ckpt)
        else:
            ckpts_expired.append(ckpt[0])
    ckpts_expired = remove_files(ckpts_expired)
    return ckpts_keeped, ckpts_expired

def kv_args(arg: str) -> Tuple[str, Any]:
    k, v = arg.split('=')
    if v.lower() in ('true', 'false'):
        v = v[0].lower() == 't'
    elif re.match("^\d+$", v):
        v = int(v)
    elif re.match(r"^\d+(\.\d*)?e\-?\d+$", v):
        v = float(v)
    elif re.match(r"^\d+\.\d+$", v):
        v = float(v)
    elif '[' in v or '{' in v:
        v = eval(v)
    return (k, v)

def count_model_params(model):
    n_total = 0
    n_trainable = 0
    for param in model.parameters():
        ps = param.size()
        n = ps[0]
        for v in ps[1:]:
            n *= v
        n_total += n
        if param.requires_grad:
            n_trainable += n
    n_freeze = n_total - n_trainable
    return n_total, n_trainable, n_freeze


def random_split_indices(dataset, test_size: int = None, test_rate: float = None):
    u_keys = list(set(dataset.keys[:]))
    n_keys = len(u_keys)
    if test_size is None and test_rate is None:
        raise ValueError("Either train_rate or test_rate should be given.")
    elif test_size is not None:
        if test_size <= 0:
            raise ValueError("test size should be larger than 0, found {}".format(test_size))
        train_size = n_keys - test_size
    elif test_rate is not None:
        if test_rate < 0 or test_rate > 1:
            raise ValueError("test rate should be in [0, 1], found {}".format(test_rate))
        train_size = int(n_keys * (1 - test_rate))
    else:
        Warning("Both test_size and test_rate are given, use test_size.")
        train_size = n_keys - test_size

    random.shuffle(u_keys)
    k2f = dict()
    for k in u_keys[:train_size]:
        k2f[k] = 0
    for k in u_keys[train_size:]:
        k2f[k] = 1
    train_indices = []
    test_indices = []
    for i, key in enumerate(dataset.keys):
        if k2f[key] == 0:
            train_indices.append(i)
        else:
            test_indices.append(i)
    return train_indices, test_indices
