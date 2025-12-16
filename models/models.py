from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Union

from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertPooler,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BertPreTrainedModel,
    BertOnlyMLMHead,
    SequenceClassifierOutput,
    MaskedLMOutput, 
    BertConfig
)
import torch.nn.functional as thfn
import dgl



from dgl.nn.pytorch import GATConv, HeteroGraphConv
from collections import defaultdict



class DrugProteinEmbeddingLayer(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()

        self.drug_embeddings = nn.Embedding(config.drug_size, config.drug_hidden_size)
        self.protein_embeddings = nn.Embedding(config.protein_size, config.protein_hidden_size)
        self.protein_weight_embedding = nn.Parameter(torch.ones((1, config.hidden_size)))

        self.drug_projector = nn.Linear(
            config.drug_hidden_size, config.hidden_size, bias=config.project_with_bias
        )
        self.protein_projector = nn.Linear(
            config.protein_hidden_size, config.hidden_size, bias=config.project_with_bias
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        drug_feats = np.load(config.drug_feature_file)
        self.reset_drug_embeddings(drug_feats, config.freeze_embedding)
        protein_feats = np.load(config.protein_feature_file)
        self.reset_protein_embeddings(protein_feats, config.freeze_embedding)

    def forward(
        self,
        drug_comb_ids: torch.LongTensor,
        protein_ids: torch.LongTensor,
        weights: torch.Tensor = None
    ):
        # N * 2 -> N * 2 * H, N = b_size, H = hidden_size
        drug_comb_embs = self.drug_projector(self.drug_embeddings(drug_comb_ids))
        # N * P -> N * P * H, P = p_size
        protein_embs = self.protein_projector(self.protein_embeddings(protein_ids))
        if weights is not None:
            if len(weights.size()) == 2:
                # N * P -> N * P * 1
                weights = weights.unsqueeze(-1)
            # (N * P * 1) * (1 * 1 * H) -> N * P * H
            weights_embeddings = weights * self.protein_weight_embedding.view(1, 1, -1)
            protein_embs += weights_embeddings  # N * P * H

        # N * (1 + 2 + P) * H
        embeddings = torch.concat([drug_comb_embs, protein_embs], dim=1)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def reset_drug_embeddings(
        self, drug_features: np.ndarray, freeze: bool = True
    ):
        drug_features = torch.from_numpy(drug_features).float()
        self.drug_embeddings = nn.Embedding.from_pretrained(drug_features, freeze)

    def reset_protein_embeddings(
        self, protein_features: np.ndarray, freeze: bool = True
    ):
        protein_features = torch.from_numpy(protein_features).float()
        self.protein_embeddings = nn.Embedding.from_pretrained(protein_features, freeze)


class BertHeadForSynergy(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.predictor = nn.Linear(config.hidden_size, 1)
    def forward(self, pooled_output):
        synergy_score = self.predictor(pooled_output)
        return synergy_score


class BertSynergyPooler(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = torch.mean(hidden_states[:, :2], dim=1)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SynergyBert(BertPreTrainedModel):

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)

        self.config = config
        self.embedding_layer = DrugProteinEmbeddingLayer(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertSynergyPooler(config) if config.add_pooler else None
        self.predictor = BertHeadForSynergy(config)

        self.post_init()

    def forward(
        self,
        drug_comb_ids: torch.LongTensor,
        protein_ids: torch.LongTensor,
        weights: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        r"""
        copying transformers.models.bert.modeling_bert
        """
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_cache = False

        input_shape = protein_ids.size()
        batch_size = input_shape[0]
        seq_length = input_shape[1] + 2
        device = drug_comb_ids.device

        # past_key_values_length
        past_key_values_length = 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)

        embedding_output = self.embedding_layer(drug_comb_ids, protein_ids, weights)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output)
        else:
            pooled_output = torch.mean(sequence_output[:, :2], dim=1)  # cls+drugA+drugB
        prediction = self.predictor(pooled_output)
        return prediction


class BertEmbeddingLayerWithoutSegEmb(nn.Module):

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.entry_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        _, seq_len = input_ids.size()
        inputs_embeddings = self.entry_embeddings(input_ids)
        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertWithoutSegEmb(BertPreTrainedModel):

    def __init__(self, config: BertConfig, add_pooler=True) -> None:
        super().__init__(config)

        self.config = config
        self.embedding_layer = BertEmbeddingLayerWithoutSegEmb(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooler else None

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        r"""
        copying transformers.models.bert.modeling_bert
        """
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_cache = False

        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device

        # past_key_values_length
        past_key_values_length = 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)

        embedding_output = self.embedding_layer(input_ids)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertWithoutSegEmbForMaskedLM(BertPreTrainedModel):

    def __init__(self, config: BertConfig, add_pooler=True) -> None:
        super().__init__(config)

        self.bert = BertWithoutSegEmb(config, add_pooler)
        self.cls = BertOnlyMLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SimCSEPooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    reference: https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError



class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    reference: https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class BertWithoutSegEmbForSimCSE(BertPreTrainedModel):

    def __init__(
        self, config: BertConfig
    ) -> None:
        super().__init__(config)

        self.bert = BertWithoutSegEmb(config, False)
        self.pooler = SimCSEPooler(config.pooler_type)
        if 'cls' in config.pooler_type:
            self.mlp = MLPLayer(config)
        self.sim = Similarity(config.temperature)
        self.loss_fct = nn.CrossEntropyLoss()

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)
        # Number of sentences in one instance
        # 2: pair instance; 3: pair instance with a hard negative
        num_sent = input_ids.size(1)

        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)

        outputs = self.bert(
            input_ids,
            attention_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        # Pooling
        pooler_output = self.pooler(attention_mask, outputs)
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
        if self.pooler.pooler_type == 'cls':
            pooler_output = self.mlp(pooler_output)
        elif self.pooler.pooler_type == 'cls_before_pooler' and self.training:
            pooler_output = self.mlp(pooler_output)

        # Separate representation
        z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
        # Hard negative
        if num_sent == 3:
            z3 = pooler_output[:, 2]
        # similarity
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        # Hard negative
        if num_sent >= 3:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        loss = None
        if labels is not None:
            loss = self.loss_fct(cos_sim, labels)

        if not return_dict:
            output = (cos_sim,) + outputs[2:]
            if loss is not None:
                output = (loss,) + output
            return output

        return SequenceClassifierOutput(
            loss=loss,
            logits=cos_sim,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def reset_linear(linear_layer: nn.Module):
    gain = nn.init.calculate_gain('relu')
    nn.init.xavier_normal_(linear_layer.weight, gain=gain)
    if linear_layer.bias is not None:
        nn.init.zeros_(linear_layer.bias)


def reset_linear_in_seq(seq_linear):
    for layer in seq_linear:
        if isinstance(layer, nn.Linear):
            reset_linear(layer)


class AutoEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
        )
        self.reconstruct_loss = nn.L1Loss(reduction='mean')

    def calc_loss(
        self,
        x: torch.Tensor,
        model_outs: Union[torch.Tensor, Tuple[torch.Tensor]]
    ) -> torch.Tensor:
        enc, dec = model_outs
        return self.reconstruct_loss(dec, x)

    def forward(self, x, ret_dec=True):
        enc = self.encoder(x)
        if ret_dec:
            dec = self.decoder(enc)
            return enc, dec
        return enc

class HANLayer(nn.Module):

    def __init__(
        self, 
        in_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.4
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.node_attn = HeteroGraphConv({
            "drug2drug": GATConv(in_dim, hidden_dim//num_heads, num_heads=num_heads, feat_drop=dropout),
            "drug2protein": GATConv(in_dim, hidden_dim//num_heads, num_heads=num_heads, feat_drop=dropout),
            "protein2drug": GATConv(in_dim, hidden_dim//num_heads, num_heads=num_heads, feat_drop=dropout),
            "protein2protein": GATConv(in_dim, hidden_dim//num_heads, num_heads=num_heads, feat_drop=dropout),
            "sideeffect2drug": GATConv(in_dim, hidden_dim//num_heads, num_heads=num_heads, feat_drop=dropout),
        }, aggregate="stack")

        self.semantic_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, g, h):
        node_feats = self.node_attn(g, h)  # {ntype: feat}, N, R, H, D
        out_h = {}
        for ntype, feats in node_feats.items():
            feats = feats.view(feats.size(0), feats.size(1), -1)
            sem_weights = self.semantic_attn(feats)  # N, R, 1
            aggregated = (feats * sem_weights).sum(dim=1) # N, H*D
            out_h[ntype] = aggregated
        return out_h

class MacroEncoder(nn.Module):

    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.4,
        etypes: List[str] = ['drug2drug', 'drug2protein', 'protein2protein', 'sideeffect2drug']
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 节点特征投影层（统一输入维度）
        self.proj_layers = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) 
            for ntype, in_dim in in_dims.items()
        })
        
        self.hans = nn.ModuleList([
            HANLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.link_pred_heads = nn.ModuleDict({
            etype: nn.Sequential(
                nn.Linear(2 * hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ) for etype in etypes
        })

    def forward(self, g):
        raw_h = {
            ntype: self.proj_layers[ntype](g.nodes[ntype].data["feat"]) 
            for ntype in g.ntypes
        }
        h = raw_h
        for han in self.hans:
            h  = han(g, h)
            for ntype, feat in raw_h.items():
                if ntype not in h:
                    h[ntype] = feat
        return h

    def link_pred_loss(self, h, edge_splits, etype, split="train"):
        u, v = edge_splits[etype][split]
        src_ntype, dst_ntype = etype.split('2')
        u_feat = h[src_ntype][u]
        v_feat = h[dst_ntype][v]
        concat_feat = torch.cat([u_feat, v_feat], dim=1)
        pred = self.link_pred_heads[etype](concat_feat) 

        pos_labels = torch.ones_like(pred)
        neg_u, neg_v = self._negative_sampling(u, v, h[src_ntype].shape[0], h[dst_ntype].shape[0])
        neg_u_feat = h[src_ntype][neg_u]
        neg_v_feat = h[dst_ntype][neg_v]
        neg_concat = torch.cat([neg_u_feat, neg_v_feat], dim=1)
        neg_pred = self.link_pred_heads[etype](neg_concat)
        neg_labels = torch.zeros_like(neg_pred)

        all_pred = torch.cat([pred, neg_pred], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)
        loss = thfn.binary_cross_entropy(all_pred, all_labels)
        return loss

    def _negative_sampling(self, u, v, num_src_nodes, num_dst_nodes, neg_ratio=1):
        num_pos = u.shape[0]
        num_neg = num_pos * neg_ratio

        neg_u = torch.randint(0, num_src_nodes, (num_neg,), device=u.device)
        neg_v = torch.randint(0, num_dst_nodes, (num_neg,), device=v.device)

        pos_set = set(zip(u.cpu().numpy(), v.cpu().numpy()))
        neg_set = set()
        for nu, nv in zip(neg_u.cpu().numpy(), neg_v.cpu().numpy()):
            if (nu, nv) not in pos_set:
                neg_set.add((nu, nv))
            if len(neg_set) == num_neg:
                break
        neg_u, neg_v = zip(*neg_set)
        return torch.tensor(neg_u, device=u.device), torch.tensor(neg_v, device=v.device)

class FusionModel(nn.Module):

    def __init__(self, micro_dim: int, macro_dim: int, hidden_size: int = 128, dropout: float = 0.4):
        super().__init__()
        self.micro_proj = nn.Linear(micro_dim, hidden_size)
        self.macro_proj = nn.Linear(macro_dim, hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, micro_feat: torch.Tensor, macro_feat: torch.Tensor, ret_prob: bool = True) -> torch.Tensor:
        micro_proj = self.micro_proj(micro_feat)
        macro_proj = self.macro_proj(macro_feat)
        fused_feat = torch.cat([micro_proj, macro_proj], dim=1)
        if ret_prob:
            prob = self.classifier(fused_feat)
            return fused_feat, prob
        return fused_feat, None
