# -*- coding:utf-8 -*-
# ###########################
# File Name: deepfm.py
# Author: geekinglcq
# Mail: lcqgeek@live.com
# Created Time: 2020-12-31 11:09:13
# ###########################
"""
DeepFM
################################################
Reference:
    Huifeng Guo et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." in IJCAI 2017.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from .base import ContextModel
from .layers import BaseFactorizationMachine, MLPLayers


class DeepFM(ContextModel):
    """DeepFM is a DNN enhanced FM which both use a DNN and a FM to calculate feature interaction.
    Also DeepFM can be seen as a combination of FNN and FM.

    """
    def __init__(self, config, dataset):
        super(DeepFM, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']

        # define layers and loss
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        size_list = [
            self.embedding_size * len(self.token_field_names) +
            len(self.float_field_names)
        ] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.deep_predict_layer = nn.Linear(
            self.mlp_hidden_size[-1], 1)  # Linear product to the final score
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, interaction):
        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        sparse_embedding, dense_embedding = self.embed_input_fields(
            interaction)
        all_embeddings = []
        batch_size = sparse_embedding.shape[0]
        if sparse_embedding is not None:
            all_embeddings.append(sparse_embedding.view(batch_size, -1))
        if dense_embedding is not None and len(dense_embedding.shape) == 3:
            all_embeddings.append(dense_embedding.view(batch_size, -1))
        # import pdb
        # pdb.set_trace()
        deepfm_all_embeddings = torch.cat(
            all_embeddings, dim=1)  # [batch_size, num_field, embed_dim]
        y_fm = self.first_order_linear(interaction) + self.fm(sparse_embedding)

        y_deep = self.deep_predict_layer(
            self.mlp_layers(deepfm_all_embeddings.view(batch_size, -1)))
        y = self.sigmoid(y_fm + y_deep)
        return y.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label.float())

    def predict(self, interaction):
        return self.forward(interaction)
