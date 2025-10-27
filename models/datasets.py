import random
import os
import logging

from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import dgl

from torch.utils.data import Dataset
from tokenizers import Tokenizer
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info

from models.utils import get_logger

os.environ['DGLBACKEND'] = 'pytorch'


def _encode_text(
    files: List[str],
    tokenizer: Tokenizer,
    chunk_size: int
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    raw_input_ids_list = []
    attention_mask_list = []
    special_token_positions_list = []
    chunk = []
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                chunk.append(line.strip())
                if len(chunk) == chunk_size:
                    enc_results = tokenizer.encode_batch(chunk)
                    for enc_res in enc_results:
                        raw_input_ids_list.append(enc_res.ids)
                        special_token_positions_list.append(set())
                        for pos, sp_msk in enumerate(enc_res.special_tokens_mask):
                            if sp_msk == 1:
                                special_token_positions_list[-1].add(pos)
                        attention_mask_list.append(enc_res.attention_mask)
                    chunk = []
    if len(chunk) > 0:
        enc_results = tokenizer.encode_batch(chunk)
        for enc_res in enc_results:
            raw_input_ids_list.append(enc_res.ids)
            special_token_positions_list.append(set())
            for pos, sp_msk in enumerate(enc_res.special_tokens_mask):
                if sp_msk == 1:
                    special_token_positions_list[-1].add(pos)
            attention_mask_list.append(enc_res.attention_mask)
    return raw_input_ids_list, attention_mask_list, special_token_positions_list


class SynergyDataset(Dataset):

    def __init__(
        self, config: Dict[str, Any], use_folds: List[int],
        key_cols: List[str] = ['drug_row_idx', 'drug_col_idx']
    ) -> None:
        super().__init__()
        # read samples
        samples = pd.read_csv(
            config.samples, sep='\t', 
            usecols=[
                'drug_row_idx', 'drug_col_idx', 'cell_line_idx', 'fold', f'synergy_{config.synergy_type}',
            ],
            dtype={
                'drug_row_idx': int,
                'drug_col_idx': int,
                'cell_line_idx': int,
                'fold': int,
                f'synergy_{config.synergy_type}': float
            }
        )
        samples = samples[samples['fold'].isin(use_folds)]
        self.raw_samples = samples.reset_index(drop=True)
        self.samples = []
        self.labels = []
        self.keys = []
        for _, row in samples.iterrows():
            self.samples.append((row['drug_row_idx'], row['drug_col_idx'], row['cell_line_idx']))
            self.labels.append(row[f'synergy_{config.synergy_type}'])
            key = sorted([row[k] for k in key_cols])
            self.keys.append(tuple(key))
    
        # read cell-proteins
        cell_protein_board = pd.read_csv(
            config.cell_protein_association, sep='\t', usecols=['cell_line_idx', 'weight', 'protein_idx'],
        )
        self.cell2weights = defaultdict(list)
        self.cell2proteins = defaultdict(list)
        for _, row in cell_protein_board.iterrows():
            self.cell2proteins[row['cell_line_idx']].append(row['protein_idx'])
            self.cell2weights[row['cell_line_idx']].append(row['weight'])
        
        # sort protein with weights
        for c in self.cell2proteins:
            pw = [(p, w) for p, w in zip(self.cell2proteins[c], self.cell2weights[c])]
            pw = sorted(pw, key=lambda x: abs(x[1]), reverse=True)
            self.cell2proteins[c] = [x[0] for x in pw]
            self.cell2weights[c] = [x[1] for x in pw]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        a, b, c = self.samples[index]
        proteins = self.cell2proteins[c]
        weights = self.cell2weights[c]
        scores = self.labels[index]
        return a, b, proteins, weights, scores

    @staticmethod
    def pad_batch(
        batch: List[Tuple],
    ) -> Tuple[torch.Tensor]:
        drug_comb_ids = []
        protein_ids = []
        weights = []
        labels = []
        attention_masks = []
        max_len = 256
        max_n_prots = min(max([len(x[2]) for x in batch]), max_len - 2)
        for a, b, p, w, s in batch:
            drug_comb_ids.append((a, b))
            attn_mask = [1] * (max_n_prots)
            if len(p) > max_n_prots:
                p = p[:max_n_prots]
                w = w[:max_n_prots]
            elif len(p) < max_n_prots:
                n_padding = max_n_prots - len(p)
                for i in range(n_padding):
                    attn_mask[len(p) + i] = 0
                p = p + [0] * n_padding
                w = w + [0.0] * n_padding
            attn_mask = [1] * 2 + attn_mask  # drug, drug, proteins
            protein_ids.append(p)
            weights.append(w)
            attention_masks.append(attn_mask)
            labels.append(s)
        ret_dict = {
            'drug_comb_ids': torch.LongTensor(drug_comb_ids),
            'protein_ids': torch.LongTensor(protein_ids),
            'weights': torch.FloatTensor(weights),
            'attention_mask': torch.LongTensor(attention_masks),
            'labels': torch.FloatTensor(labels),
        }
        return ret_dict


class TextDatasetForMLM(Dataset):

    def __init__(
        self, 
        files: List[str], 
        tokenizer: Tokenizer,
        special_tokens: Dict[str, int],
        vocab_size: int,
        mask_rate: float = 0.15,
        mask_token_rate: float = 0.8,
        random_token_rate: float = 0.1,
        chunk_size: int = 1000
    ) -> None:
        super().__init__()
        assert 0 < mask_rate < 1, \
            f"mask_rate should be in (0, 1), got {mask_rate}"
        assert 0 < mask_token_rate < 1, \
            f"mask_token_rate should be in (0, 1), got {mask_token_rate}"
        assert 0 < random_token_rate < 1, \
            f"random_token_rate should be in (0, 1), got {random_token_rate}"
        assert mask_token_rate + random_token_rate < 1, \
            "mask_token_rate + random_token_rate should less than 1, got " +\
            f"{mask_token_rate + random_token_rate}"

        raw_input_ids_list, attention_mask_list, special_token_positions_list = \
            _encode_text(files, tokenizer, chunk_size)
        self.tokenizer = tokenizer
        self.raw_input_ids_list = raw_input_ids_list
        self.attention_mask_list = attention_mask_list
        self.special_token_positions_list = special_token_positions_list

        self.pad_token_id = special_tokens['PAD_TOKEN']
        self.mask_token_id = special_tokens['MASK_TOKEN']
        self.special_token_ids = special_tokens.values()
        
        self.vocab_size = vocab_size
        self.mask_rate = mask_rate
        self.mask_token_rate = mask_token_rate
        self.random_token_rate = random_token_rate
        self._random_token_rate = random_token_rate / (1 - self.mask_token_rate)

    def __len__(self):
        return len(self.raw_input_ids_list)
    
    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        input_ids = self.raw_input_ids_list[index]
        sp_token_positions = self.special_token_positions_list[index]
        attn_mask = self.attention_mask_list[index]
        masked_input_ids, mlm_labels = self.mask_input_ids(input_ids, sp_token_positions)
        return masked_input_ids, attn_mask, mlm_labels

    def mask_input_ids(
        self, 
        input_ids: List[int], 
        special_token_positions: Set[int]
    ):
        n_mask = int((len(input_ids) - len(special_token_positions)) * self.mask_rate)
        masked_token_ids = input_ids[:]
        label_pos = []
        candidate_positions = list(range(len(input_ids)))
        random.shuffle(candidate_positions)
        for pos in candidate_positions:
            if len(label_pos) == n_mask:
                break
            if pos in special_token_positions:
                continue
            masked_token_id = None
            # mask with mask token
            if random.random() < self.mask_token_rate:
                masked_token_id = self.mask_token_id
                label_pos.append(pos)
            else:
                if random.random() < self._random_token_rate:
                    masked_token_id = self.mask_token_id
                    while masked_token_id in self.special_token_ids:
                        masked_token_id = random.randint(0, self.vocab_size - 1)
                    label_pos.append(pos)
                else:
                    masked_token_id = masked_token_ids[pos]
            masked_token_ids[pos] = masked_token_id
        mlm_labels = [-100] * len(input_ids)  # label -100 will be masked when computing mlm loss
        for pos in label_pos:
            mlm_labels[pos] = input_ids[pos]
        return masked_token_ids, mlm_labels

    def collate_fn(self, batch: List[Tuple]) -> Tuple[torch.Tensor]:
        max_len = -1
        for masked_input_ids, _, _ in batch:
            max_len = max(max_len, len(masked_input_ids))
        batch_input_ids = []
        batch_attn_mask = []
        batch_mlm_labels = []
        for masked_input_ids, attn_mask, mlm_labels in batch:
            if len(masked_input_ids) < max_len:
                n_pad = (max_len - len(masked_input_ids))
                masked_input_ids = masked_input_ids + [self.pad_token_id] * n_pad
                attn_mask = attn_mask + [0] * n_pad
                mlm_labels = mlm_labels + [-100] * n_pad
            batch_input_ids.append(masked_input_ids)
            batch_attn_mask.append(attn_mask)
            batch_mlm_labels.append(mlm_labels)
        ret_dict = {
            "input_ids": torch.LongTensor(batch_input_ids),
            "attention_mask": torch.LongTensor(batch_attn_mask),
            "labels": torch.LongTensor(batch_mlm_labels),
        }
        return ret_dict

class TextDatasetForSimCSE(Dataset):

    def __init__(
        self, 
        files: List[str], 
        tokenizer: Tokenizer,
        special_tokens: Dict[str, int],
        vocab_size: int,
        chunk_size: int = 1000
    ) -> None:
        super().__init__()

        raw_input_ids_list, attention_mask_list, _ = \
            _encode_text(files, tokenizer, chunk_size)
        self.tokenizer = tokenizer
        self.raw_input_ids_list = raw_input_ids_list
        self.attention_mask_list = attention_mask_list

        self.pad_token_id = special_tokens['PAD_TOKEN']
        self.special_token_ids = special_tokens.values()
        
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.raw_input_ids_list)
    
    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        input_ids = self.raw_input_ids_list[index]
        attn_mask = self.attention_mask_list[index]
        return input_ids, attn_mask

    def collate_fn(self, batch: List[Tuple]) -> Tuple[torch.Tensor]:
        max_len = -1
        for input_ids, attn_mask in batch:
            max_len = max(max_len, len(input_ids))
        batch_input_ids = []
        batch_attn_mask = []
        for input_ids, attn_mask in batch:
            if len(input_ids) < max_len:
                n_pad = (max_len - len(input_ids))
                input_ids = input_ids + [self.pad_token_id] * n_pad
                attn_mask = attn_mask + [0] * n_pad
            batch_input_ids.append([input_ids, input_ids])
            batch_attn_mask.append([attn_mask, attn_mask])
        ret_dict = {
            "input_ids": torch.LongTensor(batch_input_ids),
            "attention_mask": torch.LongTensor(batch_attn_mask),
            "labels": torch.arange(len(batch_input_ids)).long(),
        }

        return ret_dict


class MacroNetDataset(DGLDataset):

    def __init__(
        self,
        name: str,
        raw_dir: str,
        save_dir: str = None,
        reverse: bool = True,
        force_reload: bool = False,
        verbose: bool = False,
        logger: logging.Logger = None,
    ):
        if save_dir is None:
            save_dir = raw_dir
        self._reverse = reverse
        if logger is not None:
            self.logger = logger
        else:
            self.logger = get_logger(__name__)
        super(MacroNetDataset, self).__init__(
            name=name,
            url=None,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose
        )


    def __getitem__(self, idx):
        # 通过idx得到与之对应的一个样本
        pass

    def __len__(self):
        # 数据样本的数量
        pass
    
    def download(self):
        assert os.path.exists(os.path.join(self.raw_dir, 'drug2idx.tsv'))
        assert os.path.exists(os.path.join(self.raw_dir, 'protein2idx.tsv'))
        assert os.path.exists(os.path.join(self.raw_dir, 'ddi.tsv'))
        assert os.path.exists(os.path.join(self.raw_dir, 'dti.tsv'))
        assert os.path.exists(os.path.join(self.raw_dir, 'ppi.tsv'))
        assert os.path.exists(os.path.join(self.raw_dir, 'drug_feat.npy'))
        assert os.path.exists(os.path.join(self.raw_dir, 'protein_feat.npy'))
    
    def process(self):
        def _load_idx_map(fn):
            x2i = {}
            with open(os.path.join(self.raw_dir, fn), 'r') as f:
                next(f)
                for line in f:
                    d, i = line.strip().split('\t')
                    x2i[d] = int(i)
            return x2i

        def _load_net(fn, x2i_1, x2i_2):
            with open(os.path.join(self.raw_dir, fn), 'r') as f:
                next(f)
                for line in f:
                    e1, e2, t = line.strip().split('\t')
                    yield x2i_1[e1], x2i_2[e2], int(t)

        ent_maps = {
            'drug': _load_idx_map('drug2idx.tsv'),
            'protein': _load_idx_map('protein2idx.tsv')
        }
        relation_fns = (
            ('drug', 'drug', 'ddi.tsv'),
            ('drug', 'protein', 'dti.tsv'),
            ('protein', 'protein', 'ppi.tsv'),
        )
        hg_data = {}
        hg_mask = {}
        for ent1, ent2, fn in relation_fns:
            rel = f'{ent1}2{ent2}'
            rel_key = (ent1, rel, ent2)
            if rel not in hg_data:
                hg_data[rel_key] = ([], [])
                hg_mask[rel_key] = []
            for eid1, eid2, ds_idx in _load_net(fn, ent_maps[ent1], ent_maps[ent2]):
                hg_data[rel_key][0].append(eid1)
                hg_data[rel_key][1].append(eid2)
                hg_mask[rel_key].append(ds_idx)
        if self._reverse:
            for ent1, ent2, fn in relation_fns:
                rel = f'{ent2}2{ent1}'
                rel_key = (ent2, rel, ent1)
                if rel not in hg_data:
                    hg_data[rel_key] = ([], [])
                    hg_mask[rel_key] = []
                for eid1, eid2, ds_idx in _load_net(fn, ent_maps[ent1], ent_maps[ent2]):
                    hg_data[rel_key][0].append(eid2)
                    hg_data[rel_key][1].append(eid1)
                    hg_mask[rel_key].append(ds_idx)

        for k, v in hg_data.items():
            hg_data[k] = (torch.tensor(v[0]), torch.tensor(v[1]))
            hg_mask[k] = torch.tensor(hg_mask[k], dtype=torch.int32)

        g = dgl.heterograph(hg_data, idtype=torch.int32)
        for rel_key, mask in hg_mask.items():
            g.edges[rel_key].data['train_mask'] = (mask == 0)
            g.edges[rel_key].data['valid_mask'] = (mask == 1)
            g.edges[rel_key].data['test_mask'] = (mask == 2)
        for ent in ['drug', 'protein']:
            feat = np.load(os.path.join(self.raw_dir, f'{ent}_feat.npy'))
            g.nodes[ent].data['feature'] = torch.from_numpy(feat).float()
            if self.verbose:
                self.logger.info(f"{ent} feat shape: {feat.shape}")
        self._g = g

    def __getitem__(self, idx):
        assert idx == 0, "This dataset contains only one graph"
        return self._g

    def __len__(self):
        return 1

    def save(self):
        graph_path = self.save_path + '.bin'
        save_graphs(graph_path, [self._g])

    def load(self):
        graph_path = self.save_path + '.bin'
        graphs, _ = load_graphs(graph_path)
        self._g = graphs[0]

    def has_cache(self):
        graph_path = self.save_path + '.bin'
        return os.path.exists(graph_path)
