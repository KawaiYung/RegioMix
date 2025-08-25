import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.relation_encoder import (
    ExplicitRelationEncoder,
    ImplicitRelationEncoder,
    FaissRelationEncoder,
)
from models.language_model import (
    WordEmbedding,
    QuestionEmbedding,
    QuestionSelfAttention,
)
from utils.mimic_utils import (
    torch_extract_position_embedding,
    torch_extract_position_matrix,
)
import time
from models.relation_encoder import q_expand_v_cat
import torchvision.models as tmodels


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super(SelfAttention, self).__init__()
        if cfg.model.change_detector.att_dim % cfg.model.change_detector.att_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)"
                % (
                    cfg.model.change_detector.att_dim,
                    cfg.model.change_detector.att_head,
                )
            )
        self.num_attention_heads = cfg.model.change_detector.att_head
        self.attention_head_size = int(
            cfg.model.change_detector.att_dim / cfg.model.change_detector.att_head
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)
        self.key = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)
        self.value = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(cfg.model.change_detector.att_dim, eps=1e-6)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        )  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer += query_states
        context_layer = self.layer_norm(context_layer)
        return context_layer


class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super(CrossAttention, self).__init__()
        if cfg.model.change_detector.att_dim % cfg.model.change_detector.att_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)"
                % (
                    cfg.model.change_detector.att_dim,
                    cfg.model.change_detector.att_head,
                )
            )
        self.num_attention_heads = cfg.model.change_detector.att_head
        self.attention_head_size = int(
            cfg.model.change_detector.att_dim / cfg.model.change_detector.att_head
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)
        self.key = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)
        self.value = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(cfg.model.change_detector.att_dim, eps=1e-6)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        )  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer += query_states
        context_layer = self.layer_norm(context_layer)
        return context_layer


class GatedCrossAttention(nn.Module):
    def __init__(self, dim, head):
        super(GatedCrossAttention, self).__init__()

        self.num_attention_heads = head
        self.attention_head_size = int(dim / head)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(dim, self.all_head_size)
        self.key = nn.Linear(dim, self.all_head_size)
        self.value = nn.Linear(dim, self.all_head_size)

        # Gating network layers
        self.gate = nn.Linear(dim, self.all_head_size)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_states, key_states, value_states):
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Gating mechanism
        gating_scores = self.gate(query_states)
        gating_scores = torch.sigmoid(self.transpose_for_scores(gating_scores))
        attention_scores = attention_scores * gating_scores

        # Normalize the attention scores to probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer += value_states
        context_layer = self.layer_norm(context_layer)
        return context_layer


class SelfAttention2(nn.Module):
    def __init__(self, cfg):
        super(SelfAttention2, self).__init__()
        if cfg.model.change_detector.att_dim % cfg.model.change_detector.att_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)"
                % (
                    cfg.model.change_detector.att_dim,
                    cfg.model.change_detector.att_head,
                )
            )
        self.num_attention_heads = cfg.model.change_detector.att_head
        self.attention_head_size = int(52 / cfg.model.change_detector.att_head)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(52, self.all_head_size)
        self.key = nn.Linear(52, self.all_head_size)
        self.value = nn.Linear(52, self.all_head_size)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(52, eps=1e-6)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        )  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer += query_states
        context_layer = self.layer_norm(context_layer)
        return context_layer


class ChangeDetector(nn.Module):
    def __init__(self, cfg, word_to_idx, args):
        super().__init__()
        self.input_dim = cfg.model.change_detector.input_dim
        self.dim = cfg.model.change_detector.dim
        self.feat_dim = cfg.model.change_detector.feat_dim - 2
        self.att_head = cfg.model.change_detector.att_head
        self.att_dim = cfg.model.change_detector.att_dim
        self.nongt_dim = cfg.model.change_detector.nongt_dim
        self.pos_emb_dim = cfg.model.change_detector.pos_emb_dim
        self.num_hidden_layers = cfg.model.change_detector.num_hidden_layers
        self.use_allign = args.use_allign
        self.img = nn.Linear(self.feat_dim, self.att_dim)

        self.SSRE = SelfAttention(cfg)

        self.context1 = nn.Linear(self.att_dim, self.att_dim, bias=False)
        self.context2 = nn.Linear(self.att_dim, self.att_dim)

        self.gate1 = nn.Linear(self.att_dim, self.att_dim, bias=False)
        self.gate2 = nn.Linear(self.att_dim, self.att_dim)

        self.dropout = nn.Dropout(0.5)

        self.embed = nn.Sequential(
            # nn.Conv2d(self.att_dim*3, self.dim, kernel_size=1, padding=0),
            nn.Linear(self.att_dim * 3, self.dim),
            # nn.GroupNorm(32, self.dim),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        # self.att = nn.Conv2d(self.dim, 1, kernel_size=1, padding=0)
        self.att = nn.Linear(self.dim, 1)
        self.fc1 = nn.Linear(self.att_dim, 6)

        self.coef_sem = cfg.model.change_detector.coef_sem
        self.coef_spa = cfg.model.change_detector.coef_spa

        self.rr = nn.ModuleList(
            [SelfAttention(cfg) for _ in range(self.num_hidden_layers)]
        )
        self.embed_fc = nn.Sequential(
            nn.Linear(self.att_dim * 2, self.feat_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
        )

        # self.rr_2 = nn.ModuleList(
        #     [SelfAttention2(cfg) for _ in range(self.num_hidden_layers)]
        # )
        # self.embed_fc_2 = nn.Sequential(
        #     nn.Linear(52 * 2, 52),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        # )
        assert self.coef_sem + self.coef_spa <= 1

        if cfg.train.setting == "mode2":
            if cfg.train.graph == "all" or cfg.train.graph == "semantic":
                self.semantic_relation = ExplicitRelationEncoder(
                    cfg.model.change_detector.att_dim,
                    cfg.model.speaker.embed_dim,
                    cfg.model.change_detector.att_dim,
                    cfg.model.change_detector.dir_num,
                    cfg.model.change_detector.sem_label_num,
                    num_heads=cfg.model.change_detector.att_head,
                    num_steps=1,
                    nongt_dim=cfg.model.change_detector.nongt_dim,
                    residual_connection=True,
                    label_bias=False,
                )
            if (
                cfg.train.graph == "all"
                or cfg.train.graph == "spatial"
                or cfg.train.graph == "i+s"
            ):
                self.spatial_relation = ExplicitRelationEncoder(
                    cfg.model.change_detector.att_dim,
                    cfg.model.speaker.embed_dim,
                    cfg.model.change_detector.att_dim,
                    cfg.model.change_detector.dir_num,
                    cfg.model.change_detector.spa_label_num,
                    num_heads=cfg.model.change_detector.att_head,
                    num_steps=1,
                    nongt_dim=cfg.model.change_detector.nongt_dim,
                    residual_connection=True,
                    label_bias=False,
                )
            if (
                cfg.train.graph == "all"
                or cfg.train.graph == "implicit"
                or cfg.train.graph == "i+s"
            ):
                self.imp_relation = ImplicitRelationEncoder(
                    cfg.model.change_detector.att_dim,
                    cfg.model.speaker.embed_dim,
                    cfg.model.change_detector.att_dim,
                    cfg.model.change_detector.dir_num,
                    64,
                    cfg.model.change_detector.nongt_dim,
                    num_heads=cfg.model.change_detector.att_head,
                    num_steps=1,
                    residual_connection=True,
                    label_bias=False,
                )
        if cfg.data.use_reaug:
            self.rag_spatial_relation = ExplicitRelationEncoder(
                cfg.model.change_detector.att_dim,
                cfg.model.speaker.embed_dim,
                cfg.model.change_detector.att_dim,
                cfg.model.change_detector.dir_num,
                cfg.model.change_detector.spa_label_num,
                num_heads=cfg.model.change_detector.att_head,
                num_steps=1,
                nongt_dim=cfg.model.change_detector.nongt_dim,
                residual_connection=True,
                label_bias=False,
            )

            self.rag_semantic_relation = ExplicitRelationEncoder(
                cfg.model.change_detector.att_dim,
                cfg.model.speaker.embed_dim,
                cfg.model.change_detector.att_dim,
                cfg.model.change_detector.dir_num,
                cfg.model.change_detector.sem_label_num,
                num_heads=cfg.model.change_detector.att_head,
                num_steps=1,
                nongt_dim=cfg.model.change_detector.nongt_dim,
                residual_connection=True,
                label_bias=False,
            )

            self.imp_faiss_relation = FaissRelationEncoder(
                cfg.model.change_detector.att_dim,
                cfg.model.speaker.embed_dim,
                cfg.model.change_detector.att_dim,
                cfg.model.change_detector.dir_num,
                64,
                64,
                cfg.model.change_detector.nongt_dim,
                num_heads=cfg.model.change_detector.att_head,
                num_steps=1,
                residual_connection=True,
                label_bias=False,
            )

        self.w_emb = WordEmbedding(len(word_to_idx), 300, 0.0, "c")
        self.aw_emb = WordEmbedding(len(word_to_idx), 300, 0.0, "c")
        self.q_emb = QuestionEmbedding(
            300 if "c" not in "c" else 600, cfg.model.speaker.embed_dim, 1, False, 0.0
        )
        self.a_emb = QuestionEmbedding(
            300 if "c" not in "c" else 600, cfg.model.speaker.embed_dim, 1, False, 0.0
        )
        self.q_att = QuestionSelfAttention(cfg.model.speaker.embed_dim, 0.2)
        self.q_a_gacattn = CrossAttention(cfg)
        self.rag_diff_gacattn = CrossAttention(cfg)
        self.rag_diff_gru = QuestionEmbedding(
            cfg.model.speaker.embed_dim, cfg.model.speaker.embed_dim, 1, False, 0.0
        )
        self.answer_allign_norm = nn.LayerNorm(cfg.model.speaker.embed_dim, eps=1e-6)
        self.answer_allign_norm2 = nn.LayerNorm(cfg.model.speaker.embed_dim, eps=1e-6)

        self.rag_att = QuestionSelfAttention(cfg.model.speaker.embed_dim, 0.2)
        # self.gated_support_linear = nn.Linear(
        #     cfg.model.speaker.embed_dim, cfg.model.speaker.embed_dim
        # )
        if cfg.data.use_reaug:
            # self.topk_linear = nn.Linear(cfg.faiss.reaug_k*600, 600)
            # self.sequence_linear = nn.Linear(86*600, 20*600)

            self.faiss_context1 = nn.Linear(self.att_dim, self.att_dim, bias=False)
            self.faiss_context2 = nn.Linear(self.att_dim, self.att_dim)
            self.faiss_gate1 = nn.Linear(self.att_dim, self.att_dim, bias=False)
            self.faiss_gate2 = nn.Linear(self.att_dim, self.att_dim)
            self.faiss_embed = nn.Sequential(
                # nn.Conv2d(self.att_dim*3, self.dim, kernel_size=1, padding=0),
                nn.Linear(self.att_dim * 3, self.dim),
                # nn.GroupNorm(32, self.dim),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

            self.faiss_att = nn.Linear(self.dim, 1)
            self.faiss_fc1 = nn.Linear(
                cfg.model.speaker.embed_dim * cfg.faiss.reaug_k,
                cfg.model.speaker.embed_dim,
            )

            self.bef_embed = nn.Linear(
                cfg.model.speaker.embed_dim * 2, cfg.model.speaker.embed_dim
            )
            self.aft_embed = nn.Linear(
                cfg.model.speaker.embed_dim * 2, cfg.model.speaker.embed_dim
            )
            self.diff_embed = nn.Linear(
                cfg.model.speaker.embed_dim * 2, cfg.model.speaker.embed_dim
            )
        self.cfg = cfg
        if cfg.data.feature_mode == "mode0":
            model = getattr(tmodels, "resnet101")(pretrained=True)
            feature_map = list(model.children())
            feature_map.pop()
            feature_map.pop()
            self.extractor = nn.Sequential(*feature_map)

            self.fc_reshape = nn.Linear(2048, self.att_dim)

    def position_emb(self, bb):
        pos_mat = torch_extract_position_matrix(bb, nongt_dim=self.nongt_dim)
        pos_emb = torch_extract_position_embedding(pos_mat, feat_dim=self.pos_emb_dim)
        return pos_emb

    def forward(
        self,
        input_1,
        input_2,
        d_adj_matrix,
        q_adj_matrix,
        d_sem_adj_matrix,
        q_sem_adj_matrix,
        d_bb,
        q_bb,
        question,
        d_reaug_features,
        q_reaug_features,
        reaug_state_list,
        d_reaug_bb,
        q_reaug_bb,
        rag_d_semantic_adj,
        rag_q_semantic_adj,
        rag_d_adj_matrix,
        rag_q_adj_matrix,
        labels,
        question_mask,
        setting="mode2",
        graph="all",
    ):
        if self.cfg.data.train.empty_image == True:
            input_1 = torch.ones(input_1.shape).to(input_1.device)
            input_2 = torch.ones(input_2.shape).to(input_2.device)
            d_adj_matrix = torch.ones(d_adj_matrix.shape).to(d_adj_matrix.device)
            q_adj_matrix = torch.ones(q_adj_matrix.shape).to(q_adj_matrix.device)
            d_sem_adj_matrix = torch.ones(d_sem_adj_matrix.shape).to(
                d_sem_adj_matrix.device
            )
            q_sem_adj_matrix = torch.ones(q_sem_adj_matrix.shape).to(
                q_sem_adj_matrix.device
            )
            d_bb = torch.ones(d_bb.shape).to(d_bb.device)
            q_bb = torch.ones(q_bb.shape).to(q_bb.device)
        if self.cfg.data.feature_mode == "mode0":
            input_1 = torch.cat(
                (input_1.unsqueeze(1), input_1.unsqueeze(1), input_1.unsqueeze(1)), 1
            )
            input_2 = torch.cat(
                (input_2.unsqueeze(1), input_2.unsqueeze(1), input_2.unsqueeze(1)), 1
            )
            input_1 = self.extractor(input_1.float())
            input_2 = self.extractor(input_2.float())
            # input_1 =  input_1.reshape(input_1.shape[0],input_1.shape[1])
            # input_2 =  input_2.reshape(input_2.shape[0],input_2.shape[1])
            input_1 = self.fc_reshape(input_1.permute(0, 2, 3, 1))
            input_2 = self.fc_reshape(input_2.permute(0, 2, 3, 1))
            input_1 = input_1.reshape(input_1.shape[0], -1, input_1.shape[-1])
            input_2 = input_2.reshape(input_2.shape[0], -1, input_2.shape[-1])

        batch_size, N, C = input_1.size()
        # input_1 = input_1.view(batch_size, C, -1).permute(0, 2, 1) # (128, 196, 1026) b, h*w, c
        # input_2 = input_2.view(batch_size, C, -1).permute(0, 2, 1)

        input_bef = self.img(input_1)  # (128,196, 512)
        input_aft = self.img(input_2)

        # question part
        # q = torch.zeros(input_1.shape[0], 10).to(torch.int64).to(input_1.device)
        q = question
        w_emb = self.w_emb(q)
        assert not q.isnan().any()
        assert not w_emb.isnan().any()
        q_emb_seq = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        assert not q_emb_seq.isnan().any()
        q_emb_self_att = self.q_att(q_emb_seq)
        assert not q_emb_self_att.isnan().any()
        if self.cfg.data.use_reaug:
            # features
            rag_q_emb_self_att = q_emb_self_att.detach().clone()

            d_reaug_features = self.img(d_reaug_features)
            q_reaug_features = self.img(q_reaug_features)
            bb, K = d_reaug_features.shape[1], d_reaug_features.shape[2]
            # pseudo change caption
            faiss_diff_state = self.aw_emb(reaug_state_list.to(torch.int64))
            faiss_diff_emb_seq = self.a_emb.forward_all(faiss_diff_state)

        aff_bef = []
        aff_aft = []
        if setting == "mode1":
            input_diff = input_aft - input_bef

            input_diff, aff = self.graph_relation.forward(
                input_diff, d_adj_matrix + q_adj_matrix, q_emb_self_att
            )
        elif setting == "mode2":
            if graph == "semantic" or graph == "all":
                input_bef1, aff_bef_sem = self.semantic_relation.forward(
                    input_bef, d_sem_adj_matrix, q_emb_self_att
                )
                input_aft1, aff_aft_sem = self.semantic_relation.forward(
                    input_aft, q_sem_adj_matrix, q_emb_self_att
                )

            if graph == "spatial" or graph == "all" or graph == "i+s":
                input_bef2, aff_bef_spa = self.spatial_relation.forward(
                    input_bef, d_adj_matrix, q_emb_self_att
                )
                input_aft2, aff_aft_spa = self.spatial_relation.forward(
                    input_aft, q_adj_matrix, q_emb_self_att
                )

            if graph == "implicit" or graph == "all" or graph == "i+s":
                bef_pos_emb = self.position_emb(d_bb)
                aft_pos_emb = self.position_emb(q_bb)
                input_bef3, aff_bef_imp = self.imp_relation.forward(
                    input_bef, bef_pos_emb, q_emb_self_att
                )
                input_aft3, aff_aft_imp = self.imp_relation.forward(
                    input_aft, aft_pos_emb, q_emb_self_att
                )

            if graph == "all":
                input_bef = (
                    self.coef_sem * input_bef1
                    + self.coef_spa * input_bef2
                    + (1 - self.coef_sem - self.coef_spa) * input_bef3
                )
                input_aft = (
                    self.coef_sem * input_aft1
                    + self.coef_spa * input_aft2
                    + (1 - self.coef_sem - self.coef_spa) * input_aft3
                )
            elif graph == "i+s":
                input_bef = (input_bef2 + input_bef3) / 2
                input_aft = (input_aft2 + input_aft3) / 2
            elif graph == "semantic":
                input_bef = input_bef1
                input_aft = input_aft1
            elif graph == "spatial":
                input_bef = input_bef2
                input_aft = input_aft2
            elif graph == "implicit":
                input_bef = input_bef3
                input_aft = input_aft3

            input_bef_list = []
            for layer_idx, layer_module in enumerate(self.rr):
                # input_1, input_2 = layer_module(input_1, input_2, input_2), layer_module(input_2, input_1, input_1)
                input_bef_list.append(input_bef)
                input_bef = layer_module(input_bef, input_aft, input_aft)
                input_aft = layer_module(
                    input_aft, input_bef_list[layer_idx], input_bef_list[layer_idx]
                )
            input_diff = self.embed_fc(torch.cat([input_bef, input_aft], -1))

        elif setting == "mode3":
            input_bef = self.SSRE(input_bef, input_bef, input_bef)
            input_aft = self.SSRE(input_aft, input_aft, input_aft)

            input_diff = input_aft - input_bef

            input_diff, aff = self.graph_relation.forward(
                input_diff, d_adj_matrix + q_adj_matrix, q_emb_self_att
            )
        elif setting == "mode4":
            input_bef = self.SSRE(input_bef, input_bef, input_bef)
            input_aft = self.SSRE(input_aft, input_aft, input_aft)

            input_bef, aff = self.graph_relation.forward(
                input_bef, d_adj_matrix, q_emb_self_att
            )
            input_aft, aff = self.graph_relation.forward(
                input_aft, q_adj_matrix, q_emb_self_att
            )

            input_diff = input_aft - input_bef
        elif setting == "mode0":
            input_bef2 = q_expand_v_cat(q_emb_self_att, input_bef)
            input_aft2 = q_expand_v_cat(q_emb_self_att, input_aft)

            input_bef = self.SSRE(input_bef2, input_bef2, input_bef2)
            input_aft = self.SSRE(input_aft2, input_aft2, input_aft2)

            input_diff = input_aft - input_bef

        input_bef_context = torch.tanh(
            self.context1(input_diff) + self.context2(input_bef)
        )
        input_bef_context = self.dropout(input_bef_context)
        input_bef_gate = torch.sigmoid(self.gate1(input_diff) + self.gate2(input_bef))
        input_bef_gate = self.dropout(input_bef_gate)
        input_befs = input_bef_gate * input_bef_context

        input_aft_context = torch.tanh(
            self.context1(input_diff) + self.context2(input_aft)
        )
        input_aft_context = self.dropout(input_aft_context)
        input_aft_gate = torch.sigmoid(self.gate1(input_diff) + self.gate2(input_aft))
        input_aft_gate = self.dropout(input_aft_gate)
        input_afts = input_aft_gate * input_aft_context

        input_bef = input_bef.permute(0, 2, 1).view(batch_size, self.att_dim, N)
        input_aft = input_aft.permute(0, 2, 1).view(batch_size, self.att_dim, N)

        input_befs = input_befs.permute(0, 2, 1).view(batch_size, self.att_dim, N)
        input_afts = input_afts.permute(0, 2, 1).view(batch_size, self.att_dim, N)
        input_diff = input_diff.permute(0, 2, 1).view(batch_size, self.att_dim, N)

        input_before = torch.cat([input_bef, input_diff, input_befs], 1)
        input_after = torch.cat([input_aft, input_diff, input_afts], 1)

        embed_before = self.embed(input_before.transpose(1, 2))
        embed_after = self.embed(input_after.transpose(1, 2))
        att_weight_before = torch.sigmoid(self.att(embed_before)).transpose(1, 2)
        att_weight_after = torch.sigmoid(self.att(embed_after)).transpose(1, 2)

        att_1_expand = att_weight_before.expand_as(input_bef)
        attended_1 = input_bef * att_1_expand
        att_2_expand = att_weight_after.expand_as(input_aft)
        attended_2 = input_aft * att_2_expand

        # input_bef_list = []

        # for layer_idx, layer_module in enumerate(self.rr_2):
        #     # input_1, input_2 = layer_module(input_1, input_2, input_2), layer_module(input_2, input_1, input_1)
        #     input_bef_list.append(attended_1)
        #     attended_1 = layer_module(attended_1, attended_2, attended_2)
        #     attended_2 = layer_module(
        #         attended_2, input_bef_list[layer_idx], input_bef_list[layer_idx]
        #     )
        # input_attended = self.embed_fc_2(torch.cat([attended_1, attended_2], -1))
        if self.use_allign:
            input_attended_full = attended_2 - attended_1
        attended_1 = attended_1.sum(2)  # (batch, dim) # summed over 52
        attended_2 = attended_2.sum(2)  # (batch, dim)

        # input_attended = input_diff.sum(2)  # (batch, dim)

        input_attended = attended_2 - attended_1
        pred = self.fc1(input_attended)

        if self.cfg.data.use_reaug:
            # faiss_input_bef = input_bef[:,:,d_reaug_features.size(1):,].permute(0,2,1).detach().clone()
            # faiss_input_aft = input_aft[:,:,q_reaug_features.size(1):].permute(0,2,1).detach().clone()
            # print('faiss_input_bef', faiss_input_bef.shape)
            B, P, K, L = d_reaug_features.shape
            faiss_bef_pos_emb = self.position_emb(d_reaug_bb.reshape(-1, P, 4))
            faiss_aft_pos_emb = self.position_emb(q_reaug_bb.reshape(-1, P, 4))
            d_reaug_features = d_reaug_features.permute(0, 2, 1, 3).reshape(-1, P, L)
            q_reaug_features = q_reaug_features.permute(0, 2, 1, 3).reshape(-1, P, L)
            rag_q_emb = rag_q_emb_self_att.repeat_interleave(K, 0)

            rag_input_bef1, _ = self.rag_semantic_relation.forward(
                d_reaug_features, rag_d_semantic_adj, rag_q_emb
            )
            rag_input_aft1, _ = self.rag_semantic_relation.forward(
                q_reaug_features, rag_q_semantic_adj, rag_q_emb
            )

            rag_input_bef2, _ = self.rag_spatial_relation.forward(
                d_reaug_features, rag_d_adj_matrix, rag_q_emb
            )
            rag_input_aft2, _ = self.rag_spatial_relation.forward(
                q_reaug_features, rag_q_adj_matrix, rag_q_emb
            )

            faiss_bef3, _ = self.imp_faiss_relation.forward(
                d_reaug_features,
                faiss_bef_pos_emb,
                rag_q_emb,
            )
            faiss_aft3, _ = self.imp_faiss_relation.forward(
                q_reaug_features,
                faiss_aft_pos_emb,
                rag_q_emb,
            )

            faiss_bef3 = (
                self.coef_sem * rag_input_bef1
                + self.coef_spa * rag_input_bef2
                + (1 - self.coef_sem - self.coef_spa) * faiss_bef3
            )
            faiss_aft3 = (
                self.coef_sem * rag_input_aft1
                + self.coef_spa * rag_input_aft2
                + (1 - self.coef_sem - self.coef_spa) * faiss_aft3
            )

            faiss_diff = faiss_aft3 - faiss_bef3

            faiss_input_bef_context = torch.tanh(
                self.faiss_context1(faiss_diff) + self.faiss_context2(faiss_bef3)
            )
            faiss_input_bef_context = self.dropout(faiss_input_bef_context)
            faiss_input_bef_gate = torch.sigmoid(
                self.faiss_gate1(faiss_diff) + self.faiss_gate2(faiss_bef3)
            )
            faiss_input_bef_gate = self.dropout(faiss_input_bef_gate)
            faiss_input_befs = faiss_input_bef_gate * faiss_input_bef_context

            faiss_input_aft_context = torch.tanh(
                self.faiss_context1(faiss_diff) + self.faiss_context2(faiss_aft3)
            )
            faiss_input_aft_context = self.dropout(faiss_input_aft_context)
            faiss_input_aft_gate = torch.sigmoid(
                self.faiss_gate1(faiss_diff) + self.faiss_gate2(faiss_aft3)
            )
            faiss_input_aft_gate = self.dropout(faiss_input_aft_gate)
            faiss_input_afts = faiss_input_aft_gate * faiss_input_aft_context

            faiss_bef3 = faiss_bef3.permute(0, 2, 1).view(-1, self.att_dim, N)
            faiss_aft3 = faiss_aft3.permute(0, 2, 1).view(-1, self.att_dim, N)

            faiss_input_befs = faiss_input_befs.permute(0, 2, 1).view(
                -1, self.att_dim, N
            )
            faiss_input_afts = faiss_input_afts.permute(0, 2, 1).view(
                -1, self.att_dim, N
            )
            faiss_diff = faiss_diff.permute(0, 2, 1).view(-1, self.att_dim, N)

            faiss_input_before = torch.cat(
                [faiss_bef3, faiss_diff, faiss_input_befs], 1
            )
            faiss_input_after = torch.cat([faiss_aft3, faiss_diff, faiss_input_afts], 1)

            faiss_embed_before = self.faiss_embed(faiss_input_before.transpose(1, 2))
            faiss_embed_after = self.faiss_embed(faiss_input_after.transpose(1, 2))
            faiss_att_weight_before = torch.sigmoid(
                self.faiss_att(faiss_embed_before)
            ).transpose(1, 2)
            faiss_att_weight_after = torch.sigmoid(
                self.faiss_att(faiss_embed_after)
            ).transpose(1, 2)

            faiss_att_1_expand = faiss_att_weight_before.expand_as(faiss_bef3)
            faiss_attended_1 = (faiss_bef3 * faiss_att_1_expand).sum(2)  # (batch, dim)
            faiss_att_2_expand = faiss_att_weight_after.expand_as(faiss_aft3)
            faiss_attended_2 = (faiss_aft3 * faiss_att_2_expand).sum(2)  # (batch, dim)
            faiss_attended_1 = self.faiss_fc1(faiss_attended_1.reshape(batch_size, -1))
            faiss_attended_2 = self.faiss_fc1(faiss_attended_2.reshape(batch_size, -1))

            faiss_input_attended = faiss_attended_2 - faiss_attended_1

            # attended_1 = self.bef_embed(torch.cat([attended_1, faiss_attended_1], -1))
            # attended_2 = self.aft_embed(torch.cat([attended_2, faiss_attended_2], -1))
            # input_attended = self.diff_embed(
            #     torch.cat([input_attended, faiss_input_attended], -1)
            # )
            rag_label_state = None
            rag_diff_state = None
            rag_label_state_out = None
            if self.use_allign:
                input_attended_allign = (
                    input_attended_full.clone().detach().transpose(1, 2)
                )
                rag2label = self.rag_diff_gacattn(
                    faiss_diff_emb_seq,
                    input_attended_allign,
                    input_attended_allign,
                )
                rag2label = self.answer_allign_norm(rag2label + faiss_diff_emb_seq)
                rag_label_state = self.rag_diff_gru.forward_all(rag2label)
                faiss_diff_emb_seq = self.answer_allign_norm2(
                    rag2label + rag_label_state
                )
                rag_label_state_out = faiss_diff_emb_seq[question_mask.bool(), :, :]
                faiss_diff_emb_seq = self.q_a_gacattn(
                    q_emb_seq, faiss_diff_emb_seq, faiss_diff_emb_seq
                )
            rag_diff_state = self.rag_att(faiss_diff_emb_seq)

        # return pred, aff_bef, aff_aft, attended_1, attended_2, input_attended
        return (
            pred,
            att_weight_before,
            att_weight_after,
            attended_1,
            attended_2,
            input_attended,
            faiss_attended_1,
            faiss_attended_2,
            faiss_input_attended,
            rag_diff_state,
            rag_label_state_out,
            None,
        )


class AddSpatialInfo(nn.Module):
    def _create_coord(self, img_feat):
        batch_size, _, h, w = img_feat.size()
        coord_map = img_feat.new_zeros(2, h, w)
        for i in range(h):
            for j in range(w):
                coord_map[0][i][j] = (j * 2.0 / w) - 1
                coord_map[1][i][j] = (i * 2.0 / h) - 1
        sequence = [coord_map] * batch_size
        coord_map_in_batch = torch.stack(sequence)
        return coord_map_in_batch

    def forward(self, img_feat):
        coord_map = self._create_coord(img_feat)
        img_feat_aug = torch.cat([img_feat, coord_map], dim=1)
        return img_feat_aug
