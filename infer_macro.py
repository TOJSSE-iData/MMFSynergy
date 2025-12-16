import argparse
import os

from collections import defaultdict

import torch
import dgl

from torch.optim import AdamW

from models.datasets import MacroNetDataset
from models.utils import (
    get_logger,
    kv_args
)
from models.models import MacroEncoder
from my_config import BaseConfig


def get_dataset(config: BaseConfig, device):
    dataset = MacroNetDataset(**config.dataset)
    dataset.load()
    macro_graph, edge_splits = dataset[0]
    # train
    all_edges = {}
    for etype in dataset.pred_edge_types:
        u, v = macro_graph.edges(etype=etype)
        all_edges[etype] = {
            "all": [torch.tensor(u).to(device), torch.tensor(v).to(device)]
        }
    macro_graph = macro_graph.to(device)
    return macro_graph, all_edges
    

def main(config: BaseConfig):
    if not os.path.exists(os.path.dirname(config.save_path.drug)):
        os.makedirs(os.path.dirname(config.save_path.drug))
    if not os.path.exists(os.path.dirname(config.save_path.protein)):
        os.makedirs(os.path.dirname(config.save_path.protein))

    logger = get_logger('Infer with Macro Encoder', os.path.join(os.path.dirname(config.save_path.drug), 'infer.log'))

    if config.gpu < 0 or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = f"cuda:{config.gpu}"
    
    logger.info("Building datasets.")

    macro_graph, edge_splits = get_dataset(config, device)

    # build model
    model_cfg = config.model
    model = MacroEncoder(
        in_dims={x: macro_graph.nodes[x].data['feat'].shape[1] for x in macro_graph.ntypes},
        **model_cfg
    )
    model = model.to(device)
    # load best model
    best_model_ckpt = torch.load(config.pretrain_model_path, weights_only=True)
    model.load_state_dict(best_model_ckpt['model_state_dict'])
    logger.info(model)

    logger.info("Start infering.")
    with torch.no_grad():
        model.eval()
        model_output = model(macro_graph)
        # save embeddings
        for ntype in ['drug', 'protein']:
            emb = model_output[ntype]
            torch.save(emb, config.save_path[ntype])
    logger.info("Finish infering.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Infer macro encoder.")
    parser.add_argument("config", type=str, help="config path")
    parser.add_argument("-u", "--update", type=kv_args, nargs='*', help="path.to.config=config_value")
    args = parser.parse_args()

    config = BaseConfig()
    config.load_from_file(args.config)
    if args.update is not None:
        for k, v in args.update:
            config.set_config_via_path(k, v)
    main(config)
