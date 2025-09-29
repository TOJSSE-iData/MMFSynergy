import argparse
import os
import json

from typing import List, Tuple

from tokenizers import Tokenizer
from tokenizers.models import (
    BPE, Unigram, WordPiece, WordLevel,
    Model as TokenizerModel
)
from tokenizers.trainers import (
    BpeTrainer, UnigramTrainer, WordPieceTrainer, WordLevelTrainer,
    Trainer as TokenizerTrainer
)
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing

from my_config import BaseConfig


def get_default_config(config_fp: str) -> BaseConfig:
    """
    Get default config for training AA tokenizer.
    """
    config = BaseConfig()
    config.load_from_file(config_fp)
    return config

def get_special_tokens(tknz_cfg: BaseConfig) -> List[str]:
    """
    Get tokeinzer special tokens.
    """
    special_tokens = [
        tknz_cfg.unk_token, tknz_cfg.cls_token,
        tknz_cfg.sep_token, tknz_cfg.pad_token,
        tknz_cfg.mask_token
    ]
    return special_tokens

def prepare_tokenizer_trainer(tknz_cfg: BaseConfig) -> Tuple[TokenizerModel, TokenizerTrainer]:
    """
    Get tokenizer Model and Trainer according to given type.
    """
    special_tokens = get_special_tokens(tknz_cfg)
    unk_token = special_tokens[0]
    tknz_type = tknz_cfg.type
    if tknz_type == "BPE":
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        trainer = BpeTrainer(special_tokens=special_tokens)
    elif tknz_type == "Unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token=unk_token, special_tokens=special_tokens)
    elif tknz_type == "WordPiece":
        tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        trainer = WordPieceTrainer(special_tokens=special_tokens)
    elif tknz_type == "WordLevel":
        tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
        trainer = WordLevelTrainer(special_tokens=special_tokens)
    else:
        raise NotImplementedError(f"Unsupport tokenizer type: {tknz_type}")
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = BertProcessing(
        sep=(special_tokens[2], 2), cls=(special_tokens[1], 1)
    )
    return tokenizer, trainer

def get_tokenizer(config: BaseConfig, train=True) -> Tokenizer:
    """
    Get tokenizer by training or loading from path.
    """
    tknz_fp = os.path.join(config.model_dir, 'tokenizer.json')
    tknz_cfg_fp = os.path.join(config.model_dir, 'config.json')
    if train:
        tokenizer, trainer = prepare_tokenizer_trainer(config.tokenizer)
        tokenizer.train(config.dataset.train.files, trainer)
        if not os.path.exists(config.model_dir):
            os.makedirs(config.model_dir)
        tokenizer.save(tknz_fp)
        # save some config manually
        cfg = {}
        with open(tknz_fp, 'r') as f:
            tknz_cfg = json.load(f)
        cfg['type'] = config.tokenizer.type
        cfg['vocab_size'] = len(tknz_cfg['model']['vocab'])
        cfg['special_tokens'] = {}
        for sp_token_type in ['unk', 'cls', 'sep', 'pad', 'mask']:
            token = config.tokenizer[f'{sp_token_type}_token']
            tk_id = tokenizer.token_to_id(token)
            cfg['special_tokens'][f"{sp_token_type.upper()}_TOKEN"] = {
                "token": token,
                "id": tk_id
            }
        with open(tknz_cfg_fp, 'w') as f:
            json.dump(cfg, f)

    tokenizer = Tokenizer.from_file(tknz_fp)
    with open(tknz_cfg_fp, 'r') as f:
        cfg = json.load(f)
    return tokenizer, cfg

def demo(config: BaseConfig, tokenizer: Tokenizer):
    with open(config.dataset.train.files[0], 'r') as f:
        sequence = next(f).strip()
    print("sequence:", sequence)
    enc_res = tokenizer.encode(sequence)
    print(enc_res)
    print("ids:", enc_res.ids)
    print("tokens:", enc_res.tokens)
    print("attention_mask:", enc_res.attention_mask)
    print("special_tokens_mask:", enc_res.special_tokens_mask)
    print("overflowing:", enc_res.overflowing)

def demo_batch(config: BaseConfig, tokenizer: Tokenizer):
    sequences = []
    with open(config.dataset.train.files[0], 'r') as f:
        for _ in range(2):
            sequence = next(f).strip()
            sequences.append(sequence)
    print("sequences:", sequences)
    tokenizer.enable_truncation(32)
    enc_res = tokenizer.encode_batch(sequences)
    print(enc_res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train Tokenizer")
    parser.add_argument("config", type=str, help="config filepath")
    parser.add_argument("--model_dir", type=str, default=None, help="model dir")
    parser.add_argument("--train", action="store_true", help="train tokenizer")
    parser.add_argument("--demo", action="store_true", help="demo tokenizer result")
    parser.add_argument("--demo_batch", action="store_true", help="demo batch tokenizer result")
    args = parser.parse_args()

    config = get_default_config(args.config)
    if args.model_dir is not None:
        config.model_dir = args.model_dir
    tokenizer, tokenizer_cfg = get_tokenizer(config, args.train)
    if args.demo:
        demo(config, tokenizer)
    if args.demo_batch:
        demo_batch(config, tokenizer)
