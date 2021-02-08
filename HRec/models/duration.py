# -*- coding:utf-8 -*-
# ###########################
# File Name: duration.py
# Author: geekinglcq
# Mail: lcqgeek@live.com
# Created Time: 2021-01-07 15:28:53
# ###########################

import torch
import random
import logging
import numpy as np
import torch.nn as nn
from .layers import MLPLayers, set_kernel_layer
from .base import HModel
from torch.nn.init import normal_
from collections import Counter, defaultdict
from itertools import combinations_with_replacement


class DURation(HModel):
    """ Deep Unified Representation for Heterogeneous Recommendation"""
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.logger = logging.getLogger()

        self.LABEL = dataset.config['LABEL_FIELD']
        # self.RATING = dataset.config['RATING_FIELD']

        self.user_emb_size = config['user_emb_size']
        self.item_emb_size = config['item_emb_size']
        self.token_emb_size = config['token_emb_size']

        # The number of item types
        self.P = len(dataset.config['item_feat_path'])
        self.item_size = dataset.item_nums

        self.device = config['device']
        self.user_hidden_size_list = config['user_hidden_size_list']
        self.item_hidden_size_list = config['item_hidden_size_list']
        self.item_map_hidden_size_list = config['item_map_hidden_size_list']
        self.kernel = set_kernel_layer(config.get('kernel', 'gaussian'))

        assert self.user_hidden_size_list[-1] == self.item_hidden_size_list[-1]
        self.inter_matrix_type = dataset.config['inter_matrix_type']

        # generate intermediate data
        if self.inter_matrix_type == '01':
            self.history_user_id, self.history_user_value, _ = dataset.history_user_matrix(
            )
            self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix(
            )
            self.interaction_matrix = dataset.inter_matrix(form='csr').astype(
                np.float32)
        elif self.inter_matrix_type == 'rating':
            self.history_user_id, self.history_user_value, _ = dataset.history_user_matrix(
                value_field=self.RATING)
            self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix(
                value_field=self.RATING)
            self.interaction_matrix = dataset.inter_matrix(
                form='csr', value_field=self.RATING).astype(np.float32)

        self.max_rating = self.history_user_value.max()
        # tensor of shape [n_items, H] where H is max length of history interaction.

        # Keep the user matrix in cpu to save gpu mem
        # self.history_user_id = self.history_user_id.to(self.device)
        # self.history_user_value = self.history_user_value.to(self.device)

        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)

        # define layers
        self.user_linear = nn.Linear(in_features=self.n_items,
                                     out_features=self.user_emb_size,
                                     bias=False)

        self.map_func_dict = nn.ModuleDict()

        self.pdist = nn.PairwiseDistance(p=2)

        for item_type, item_feats in dataset.item_feat_fields.items():

            item_feat_type_count = Counter(
                [dataset.field2type[i] for i in item_feats])
            input_dim = (item_feat_type_count['token'] + 1) * self.token_emb_size + \
                item_feat_type_count['float']
            self.map_func_dict[item_type] = MLPLayers(
                [input_dim] + self.item_map_hidden_size_list).to(self.device)

        self.item_linear = nn.Linear(in_features=self.n_users,
                                     out_features=self.item_emb_size,
                                     bias=False)
        self.user_fc_layers = MLPLayers([self.user_emb_size] +
                                        self.user_hidden_size_list).to(
                                            self.device)
        self.item_fc_layers = MLPLayers(
            [self.item_map_hidden_size_list[-1] + self.item_emb_size] +
            self.item_hidden_size_list).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()

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

    def get_item_embedding(self, item_type, data):
        item_id = data[self.ITEM_ID].long()

        # Following lines construct tensor of shape [B,n_users] using the tensor of shape [B,H]
        row_indices = torch.arange(item_id.shape[0],
                                   device=self.device).repeat_interleave(
                                       self.history_user_id.shape[1], dim=0)
        col_indices = self.history_user_id[item_id].flatten().to(self.device)
        matrix_01 = torch.zeros(1, device=self.device).repeat(
            item_id.shape[0], self.n_users)
        matrix_01.index_put_(
            (row_indices, col_indices),
            self.history_user_value[item_id].flatten().to(self.device))
        item_inter_feat = self.item_linear(matrix_01)

        map_layers = self.map_func_dict[item_type]
        item_emb = self.agg_item_feature(item_type, data)
        item_transformed_emb = map_layers(item_emb)
        item_feat = torch.cat([item_inter_feat, item_transformed_emb], dim=-1)

        return item_emb, item_transformed_emb, item_feat

    def forward(self, item_type, data, return_item_emb=False):

        # Interaction-related features
        user = data[self.USER_ID]
        item_id = data[self.ITEM_ID]
        user = self.get_user_embedding(user)

        # Following lines construct tensor of shape [B,n_users] using the tensor of shape [B,H]
        col_indices = self.history_user_id[item_id].flatten().to(self.device)
        row_indices = torch.arange(item_id.shape[0],
                                   device=self.device).repeat_interleave(
                                       self.history_user_id.shape[1], dim=0)
        matrix_01 = torch.zeros(1, device=self.device).repeat(
            item_id.shape[0], self.n_users)
        matrix_01.index_put_(
            (row_indices, col_indices),
            self.history_user_value[item_id].flatten().to(self.device))
        item_inter_feat = self.item_linear(matrix_01)

        # Context-related features

        # Map heterogeneous raw feature to unified feature space

        map_layers = self.map_func_dict[item_type]
        item_emb = self.agg_item_feature(item_type, data)
        item_transformed_emb = map_layers(item_emb)
        item_feat = torch.cat([item_inter_feat, item_transformed_emb], dim=-1)

        user = self.user_fc_layers(user)
        item = self.item_fc_layers(item_feat)

        vector = torch.mul(user, item).sum(dim=1)
        vector = self.sigmoid(vector)

        if return_item_emb:
            return vector, item_emb, item_transformed_emb
        else:
            return vector

    def calculate_topo_loss(self, raw_emb, emb):
        """
        Calculate the topology loss, for every pair of items sampled from given batch,
        calculate the
            |x_i, x_j|^2 * W(r_i, r_j)
        x is the tranformed representation
        r is the raw representation
        W is the similarity function

        Input:
            raw_emb: [bs, dim] raw features
            emb: [bs, new_dim] embedding in transformed feature space

        """

        d = emb.shape[0]
        n_r = raw_emb.shape[1]
        n_x = emb.shape[1]
        # r_one = torch.ones((1, n_r), device=self.device) @ raw_emb.T
        # x_one = torch.ones((1, n_x), device=self.device) @ emb.T
        c_r = 1 / (n_r - 1) * torch.matmul(raw_emb, raw_emb.T)
        # (1 / n_r) * torch.matmul(r_one.T, r_one))
        c_x = 1 / (n_x - 1) * torch.matmul(emb, emb.T)
        # (1 / n_x) * torch.matmul(x_one.T, x_one))

        loss = 1 / (4 * d**2) * (c_r - c_x).pow(2).sum().sqrt()
        return loss

    def calculate_align_loss(self, data):
        """
        Calculate the alignment loss. For each batch, sample a number of pairs to minimize
        the alignment loss.
        """
        size = 128
        item_size = self.item_size

        min_size = min([i.shape[0] for i in data.values()])
        if min_size < size:
            return None
        losses = []
        for type_i, type_j in combinations_with_replacement(data.keys(), r=2):

            if type_i == type_j:
                factor = (self.P - 1) / (self.P**2 * item_size[type_i]**2)
            else:
                factor = -1 / (self.P**2 * item_size[type_i] *
                               item_size[type_j])
            indice_i = random.sample(range(min_size), size)
            indice_i = torch.tensor(indice_i, device=self.device)
            sample_i = data[type_i][indice_i]
            indice_j = random.sample(range(min_size), size)
            indice_j = torch.tensor(indice_j, device=self.device)
            sample_j = data[type_j][indice_j]

            res = self.kernel(sample_i, sample_j)
            loss = factor * res
            losses.append(loss)

        align_loss = torch.sum(torch.stack(losses))
        return align_loss

    def calculate_loss(self, data):
        # when starting a new epoch, the item embedding we saved must be cleared.
        # The
        if self.training:
            self.i_embedding = None

        losses = []
        losses_dict = defaultdict(int)

        item_emb_dict = {}
        for item_type, item_data in data.items():

            if self.inter_matrix_type == '01':
                label = item_data[self.LABEL].float()
            elif self.inter_matrix_type == 'rating':
                label = item_data[self.RATING] * item_data[self.LABEL]

            output, item_raw_emb, item_emb = self.forward(item_type,
                                                          item_data,
                                                          return_item_emb=True)

            item_emb_dict[item_type] = item_emb

            topo_loss = 0.001 * self.calculate_topo_loss(
                item_raw_emb, item_emb)
            losses_dict['topo'] += topo_loss.item()
            losses.append(topo_loss)

            label = label / self.max_rating  # normalize the label to calculate BCE loss.

            cls_loss = self.bce_loss(output, label)
            losses_dict['cls'] += cls_loss.item()

            losses.append(cls_loss)

        align_loss = 5e8 * self.calculate_align_loss(item_emb_dict)
        if align_loss is not None:
            losses.append(align_loss)
            losses_dict['align'] += align_loss.item()

        loss = torch.sum(torch.stack(losses))
        return loss, losses_dict

    def predict(self, h, data):
        return self.forward(h, data)

    def get_user_embedding(self, user):
        r"""Get a batch of user's embedding with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, emb_size]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(user.shape[0],
                                   device=self.device).repeat_interleave(
                                       self.history_item_id.shape[1], dim=0)
        matrix_01 = torch.zeros(1, device=self.device).repeat(
            user.shape[0], self.n_items)
        matrix_01.index_put_((row_indices, col_indices),
                             self.history_item_value[user].flatten())
        user = self.user_linear(matrix_01)

        return user


if __name__ == '__main__':

    model = DURation()
