# -*- coding:utf-8 -*-
# ###########################
# File Name: cccf.py
# Author: geekinglcq
# Mail: lcqgeek@live.com
# Created Time: 2021-01-01 16:10:50
# ###########################

import torch
import logging
import numpy as np
import torch.nn as nn
from .layers import MLPLayers
from .base import HModel
from torch.nn.init import normal_
from collections import Counter, defaultdict


class CCCFNet(HModel):
    """CCCFNet
    CCCFNet: A Content-Boosted Collaborative Filtering Neural Network for Cross Domain Recommender Systems
    """
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.logger = logging.getLogger()

        self.LABEL = dataset.config['LABEL_FIELD']

        self.user_emb_size = config['user_emb_size']
        self.item_emb_size = config['item_emb_size']
        self.token_emb_size = config['token_emb_size']

        self.user_cf_embedding = nn.Embedding(self.n_users, self.user_emb_size)
        self.item_cf_embedding = nn.Embedding(self.n_items, self.item_emb_size)

        self.P = len(dataset.config['item_feat_path'])
        self.item_size = dataset.item_nums

        self.device = config['device']
        self.user_hidden_size_list = config['user_hidden_size_list']
        self.item_hidden_size_list = config['item_hidden_size_list']

        assert self.user_hidden_size_list[-1] == self.item_hidden_size_list[-1]

        self.item_nn_dict = nn.ModuleDict()

        for item_type, item_feats in dataset.item_feat_fields.items():
            item_feat_type_count = Counter(
                [dataset.field2type[i] for i in item_feats])
            input_dim = (item_feat_type_count['token'] + 1) * self.token_emb_size + \
                item_feat_type_count['float']
            self.item_nn_dict[item_type] = MLPLayers(
                [input_dim + self.user_emb_size] + self.item_hidden_size_list,
                activation='tanh').to(self.device)

        self.user_fc_layers = MLPLayers([self.user_emb_size] +
                                        self.user_hidden_size_list).to(
                                            self.device)

        self.bce_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        # Save the item embedding before dot product layer to speed up evaluation
        self.i_embedding = None

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)

    def agg_item_feature(self, item_type, item_data):

        token_embeddings = []
        float_feats = []
        for feat_name, feat_value in item_data.items():
            if feat_name in self.token_embedding_table and feat_name != self.USER_ID:
                emb = self.token_embedding_table[feat_name](feat_value.long())
                token_embeddings.append(emb)
            if feat_name in self.float_field_names:
                float_feat = feat_value.float()
                if float_feat.dim() == 1:
                    float_feat = float_feat.unsqueeze(-1)
                float_feats.append(float_feat)
        all_emb = torch.cat(token_embeddings + float_feats, dim=-1)
        return all_emb

    def forward(self, item_type, data):

        user = data[self.USER_ID]
        item_id = data[self.ITEM_ID]
        user_emb = self.user_cf_embedding(user)
        item_cf_emb = self.item_cf_embedding(item_id)

        item_layer = self.item_nn_dict[item_type]
        item_content_emb = self.agg_item_feature(item_type, data)
        item_emb = torch.cat([item_cf_emb, item_content_emb], dim=-1)
        item_emb = item_layer(item_emb)

        user_emb = self.user_fc_layers(user_emb)

        vector = torch.mul(user_emb, item_emb).sum(dim=1)
        vector = self.sigmoid(vector)
        return vector

    def calculate_loss(self, data):
        losses = []
        losses_dict = defaultdict(int)

        for item_type, item_data in data.items():

            output = self.forward(item_type, item_data)

            label = item_data[self.LABEL].float()
            tmp_loss = self.bce_loss(output, label)
            losses.append(tmp_loss)
            losses_dict['total'] += tmp_loss.item()

        loss = torch.sum(torch.stack(losses))
        return loss, losses_dict

    def predict(self, h, data):
        return self.forward(h, data)
