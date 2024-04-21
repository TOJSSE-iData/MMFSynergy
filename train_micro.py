import argparse
import yaml
import pandas as pd

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from datasets import IterableDataset

from torch.utils.data import DataLoader


def load_tokenizer(tokenizer_cfg):
    tknz = Tokenizer.from_file(tokenizer_cfg['tokenizer_file'])
    tknz = PreTrainedTokenizerFast(tokenizer_object=tknz)
    tknz.pad_token = '[PAD]'
    return tknz


def generate_sample(triplets, i2s):
    for anchor, pos, neg in triplets:
        yield {
            "anchor": i2s[anchor],
            "positive": i2s[pos],
            "negative": i2s[neg]
        }


def encode(sample, tokenizer, **kwargs):
    ret = {}
    kwargs['return_tensors'] = 'pt'
    for src in ['anchor', 'positive', 'negative']:
        inputs = tokenizer(
            sample[src], **kwargs
        )
        for k, v in inputs.items():
            ret[f'{src}_'+k] = v
    return ret


def load_micro_dataset(cfg, usage, tokenizer=None):
    tokenizer_cfg = cfg['tokenizer']
    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_cfg)

    dataset_cfg = cfg['dataset']
    df = pd.read_csv(dataset_cfg[usage], sep='\t')
    df[dataset_cfg['pos_col']] = df[dataset_cfg['pos_col']].apply(lambda x: x.split(','))
    df[dataset_cfg['neg_col']] = df[dataset_cfg['neg_col']].apply(lambda x: x.split(','))
    df = df.explode(dataset_cfg['pos_col']).explode(dataset_cfg['neg_col'])
    t_a = df[dataset_cfg['id_col']].tolist()
    t_p = df[dataset_cfg['pos_col']].tolist()
    t_n = df[dataset_cfg['neg_col']].tolist()
    triplets = list(zip(t_a, t_p, t_n))
    id2seq = pd.read_csv(dataset_cfg['id2seq'], sep='\t', index_col=dataset_cfg['id_col'])
    id2seq = id2seq[dataset_cfg['seq_col']].to_dict()
    gen_kwargs = {"triplets": triplets, "i2s": id2seq}
    micro_dataset = IterableDataset.from_generator(generate_sample, gen_kwargs=gen_kwargs)
    encode_kwargs = {k: v for k, v in tokenizer_cfg.items() if k in ['max_length', 'truncation', 'padding']}
    micro_dataset = micro_dataset.map(
        lambda x: encode(x, tokenizer, **encode_kwargs),
        batched=True,
        remove_columns=['anchor', 'positive', 'negative']
    )
    return micro_dataset, tokenizer


def train_micro(config):
    train_set, tokenizer = load_micro_dataset(config, 'train')
    valid_set, _ = load_micro_dataset(config, 'valid', tokenizer)
    test_set, _ = load_micro_dataset(config, 'test', tokenizer)
    train_loader = DataLoader(train_set, batch_size=64)
    valid_loader = DataLoader(valid_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="config file path")
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='UTF-8') as yaml_file:
        config = yaml.safe_load(yaml_file)
    # update gpu config
    if 'gpu' not in config.keys():
        config['gpu'] = args.gpu
    elif args.gpu >= 0:
        config['gpu'] = args.gpu

    # print(config)
    train_micro(config)