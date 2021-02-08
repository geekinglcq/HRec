# -*- coding:utf-8 -*-
# ###########################
# File Name: ddtcdr.py
# Author: geekinglcq
# Mail: lcqgeek@live.com
# Created Time: 2021-02-03 01:38:58
# ###########################

import torch
import logging
import torch.nn as nn
from .base import HModel
from collections import Counter


class DDTCDR(HModel):
    """ DDTCDR
    DDTCDR: Deep Dual Transfer Cross Domain Recommendation.
    """
    def __init__(self, config, dataset, item_type):
        super().__init__(config, dataset)
        self.logger = logging.getLogger()

        self.LABEL = dataset.config['LABEL_FIELD']
        # self.RATING = dataset.config['RATING_FIELD']

        self.user_emb_size = config['latent_dim']
        self.item_emb_size = config['latent_dim']

        self.layers = config['layers']
        self.token_emb_size = config['token_emb_size']
        self.user_cf_embedding = nn.Embedding(self.n_users, self.user_emb_size)
        self.item_cf_embedding = nn.Embedding(self.n_items, self.item_emb_size)

        self.latent_dim = config['latent_dim']
        self.fc_layers = torch.nn.ModuleList()

        item_feats = dataset.item_feat_fields[item_type]
        item_feat_type_count = Counter(
            [dataset.field2type[i] for i in item_feats])
        input_dim = (item_feat_type_count['token'] + 1) * self.token_emb_size + \
            item_feat_type_count['float'] + self.user_emb_size + self.item_emb_size

        self.layers.insert(0, input_dim)
        for idx, (in_size,
                  out_size) in enumerate(zip(self.layers[:-1],
                                             self.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1],
                                             out_features=1)
        self.logistic = torch.nn.Sigmoid()
        self.bridge = torch.nn.Linear(config['latent_dim'],
                                      config['latent_dim'])
        torch.nn.init.orthogonal_(self.bridge.weight)

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

    def forward(self, item_type, data, dual=False):
        user = data[self.USER_ID]
        item_id = data[self.ITEM_ID]
        user_emb = self.user_cf_embedding(user)
        if dual:
            user_emb = self.bridge(user_emb)
        item_cf_emb = self.item_cf_embedding(item_id)

        item_content_emb = self.agg_item_feature(item_type, data)
        item_emb = torch.cat([item_cf_emb, item_content_emb], dim=-1)
        vector = torch.cat([user_emb, item_emb], dim=-1)
        vector = vector.float()

        for fc in self.fc_layers:
            vector = fc(vector)
            vector = torch.nn.Dropout(p=0.1)(vector)
            vector = torch.nn.ReLU()(vector)
        rating = self.affine_output(vector)
        rating = self.logistic(rating)
        return rating

    def calculate_loss(self):
        pass

    def predict(self, h, data):
        return self.forward(h, data)
