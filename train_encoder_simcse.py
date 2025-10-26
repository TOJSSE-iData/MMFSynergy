import argparse
import os

from collections import OrderedDict

import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW

from models.datasets import TextDatasetForSimCSE
from models.utils import (
    seet_random_seed,
    get_logger,
    convert_to_bert_config,
    get_scheduler_by_name,
    keep_top_k_checkpoints,
    kv_args
)
from models.models import BertWithoutSegEmbForSimCSE
from my_config import BaseConfig
from train_tokenizer import get_tokenizer


def get_default_config(config_fp: str) -> BaseConfig:
    config = BaseConfig()
    config.set_config_via_path("tokenizer.truncate", True)
    config.set_config_via_path("tokenizer.max_length", 512)
    config.set_config_via_path("trainer.scheduler.name", 'constant')
    config.set_config_via_path("trainer.scheduler.params.num_training_steps", 1000)
    config.set_config_via_path("trainer.scheduler.params.num_warmup_steps", 100)
    config.load_from_file(config_fp)

    return config

def get_dataloader(config: BaseConfig):
    def _get_dataloader(usage):
        data_set = TextDatasetForSimCSE(
            config.dataset[usage].files, tokenizer, special_tokens,
            tknz_cfg['vocab_size']
        )
        loader = DataLoader(
            data_set, 
            collate_fn=data_set.collate_fn,
            **config.dataset[usage].loader
        )
        return loader
    # load tokenizer
    tokenizer, tknz_cfg = get_tokenizer(config.tokenizer, train=False)
    if config.tokenizer.truncate:
        tokenizer.enable_truncation(config.tokenizer.max_length)
    special_tokens = {k: v['id'] for k, v in tknz_cfg['special_tokens'].items()}
    train_loader = _get_dataloader('train')
    # valid_loader = _get_dataloader('valid')
    # test_loader = _get_dataloader('test')
    # return train_loader, valid_loader, test_loader
    return train_loader

def main(config: BaseConfig):
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    logger = get_logger('Train Encoder SimCSE', os.path.join(config.model_dir, 'train.log'))

    if config.gpu < 0 or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = f"cuda:{config.gpu}"
    
    logger.info("Building datasets.")
    train_loader = get_dataloader(config)

    # build model
    model_cfg = config.model
    if model_cfg.vocab_size is None:
        model_cfg.vocab_size = train_loader.dataset.vocab_size
    model_cfg = convert_to_bert_config(model_cfg)
    model = BertWithoutSegEmbForSimCSE(model_cfg)
    model.to(device)
    logger.info(model)
    if hasattr(config, 'pretrain_model_path'):
        pretrain_model_path = config.pretrain_model_path
        # load pre-trained weights
        ckpt = torch.load(pretrain_model_path, map_location=device)
        mismatch = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        logger.info(f"Missing keys: {mismatch.missing_keys}")
        logger.info(f"Unexpected keys: {mismatch.unexpected_keys}")
    # define optimizer, loss
    trainer_cfg = config.trainer
    optimizer = AdamW(model.parameters(), **trainer_cfg.optimizer)

    scheduler_cfg = trainer_cfg.scheduler
    scheduler = get_scheduler_by_name(scheduler_cfg.name, optimizer, **scheduler_cfg.params)
    patience = trainer_cfg.patience
    angry = 0
    # run training
    cur_step = 0
    min_loss = float('inf')
    ckpts_by_step = []
    ckpts_by_loss = []
    config.save_to_file(os.path.join(config.model_dir, 'configs.yml'))
    logger.info("Start training.")
    for epc in range(1, trainer_cfg.num_epochs + 1):
        for batch in train_loader:
            model.train()
            cur_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            model_output = model(**batch)
            model_output.loss.backward()
            train_loss = model_output.loss.item()
            optimizer.step()
            scheduler.step()
            if cur_step % trainer_cfg.print_per_steps == 0:
                logger.info(f"Epoch {epc:03d} | Step {cur_step:07d} | Train Loss: {train_loss:.4f}")
            if cur_step % trainer_cfg.save_per_steps == 0:
                ckpt_for_saving = {
                    'epoch': epc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }
                ckpt_path = os.path.join(config.model_dir, f"model_{cur_step}.pt")
                torch.save(ckpt_for_saving, ckpt_path)
                ckpts_by_step.append((ckpt_path, cur_step))
                logger.info(f"Save model at Epoch {epc:03d} Step {cur_step:07d} | Train Loss: {train_loss:.4f}")
                ckpts_by_step, ckpts_expired = keep_top_k_checkpoints(
                    ckpts_by_step, trainer_cfg.max_keep, cmp='max'
                )
                for ckpt_expired in ckpts_expired:
                    logger.info(f"Remove model at {ckpt_expired}")
        #     if cur_step % trainer_cfg.eval_per_steps == 0:
        #         with torch.no_grad():
        #             model.eval()
        #             valid_loss = 0
        #             for valid_batch in valid_loader:
        #                 valid_batch = {k: v.to(device) for k, v in valid_batch.items()}
        #                 model_output = model(**valid_batch)
        #                 valid_loss += model_output.loss.item() * valid_batch['labels'].size()[0]
        #             valid_loss /= len(valid_loader.dataset)
        #             if min_loss > valid_loss:
        #                 angry = 0
        #                 ckpt_for_saving = {
        #                     'epoch': epc,
        #                     'model_state_dict': model.state_dict(),
        #                     'optimizer_state_dict': optimizer.state_dict(),
        #                     'loss': valid_loss,
        #                 }
        #                 min_loss = valid_loss
        #                 ckpt_path = os.path.join(config.model_dir, f"model_{cur_step}_{min_loss:.4f}.pt")
        #                 torch.save(ckpt_for_saving, ckpt_path)
        #                 ckpts_by_loss.append((ckpt_path, min_loss))
        #                 logger.info(f"Save best model at Epoch {epc:03d} Step {cur_step:07d} | Valid Loss: {valid_loss:.4f}")
        #                 ckpts_by_loss, ckpts_expired = keep_top_k_checkpoints(
        #                     ckpts_by_loss, trainer_cfg.max_keep_best, cmp='min'
        #                 )
        #                 for ckpt_expired in ckpts_expired:
        #                     logger.info(f"Remove model at {ckpt_expired}")
        #             else:
        #                 angry += 1
        #                 if angry >= patience:
        #                     logger.info(f"Valid loss has not decreased for {patience} evals, early stopped.")
        #                     break
        # if angry >= patience:
        #     break
        logger.info(f"Epoch {epc:03d} | Train Loss: {train_loss:.4f}")
    logger.info("Finish training.")
    
    # # load best model
    # best_model_ckpt = torch.load(ckpts_by_loss[0][0])
    # model.load_state_dict(best_model_ckpt['model_state_dict'])

    # # run testing
    # logger.info("Start testing.")
    # with torch.no_grad():
    #     model.eval()
    #     test_loss = 0
    #     for test_batch in test_loader:
    #         test_batch = {k: v.to(device) for k, v in test_batch.items()}
    #         model_output = model(**test_batch)
    #         test_loss += model_output.loss.item() * test_batch['labels'].size()[0]
    #     test_loss /= len(test_loader.dataset)
    # logger.info(f"Test Loss: {test_loss:.4f}")
    # logger.info("Finish testing.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train AA encoder with SimCSE task.")
    parser.add_argument("config", type=str, help="config path")
    parser.add_argument("-u", "--update", type=kv_args, nargs='*', help="path.to.config=config_value")
    args = parser.parse_args()

    config = get_default_config(args.config)
    if args.update is not None:
        for k, v in args.update:
            config.set_config_via_path(k, v)
    seet_random_seed()
    main(config)
