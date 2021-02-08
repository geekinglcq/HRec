# -*- coding:utf-8 -*-

import abc

import torch
import numpy as np
import torch.nn as nn

from .utils import ModelType
from .layers import FMEmbedding, FMFirstOrderLinear
from ..datasets import FeatureSource


class BaseModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def predict(self, interaction):
        """ predict the score between users and items"""

    @abc.abstractmethod
    def forward(self):
        """ forward"""

    @abc.abstractmethod
    def calculate_loss(self, interaction):
        """ Calculate the loss """


class GeneralModel(BaseModel):
    model_type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.config.update(dataset.config)
        self.USER_ID = dataset.config['USER_ID_FIELD']
        self.ITEM_ID = dataset.config['ITEM_ID_FIELD']
        self.LABEL = dataset.config["LABEL_FIELD"]
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.device = config.get('device', 'cpu')


class ContextModel(BaseModel):
    model_type = ModelType.CONTEXT

    def __init__(self, config, dataset):
        super().__init__()
        self.USER_ID = dataset.config['USER_ID_FIELD']
        self.ITEM_ID = dataset.config['ITEM_ID_FIELD']
        self.LABEL = dataset.config["LABEL_FIELD"]
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.device = config.get('device', 'cpu')
        self.double_tower = config.get('double_tower', False)
        self.embedding_size = config.get('embedding_size', 64)

        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []

        self.num_feature_field = 0

        self.field_names = dataset.fields()

        if self.double_tower:
            self.user_field_names = []
            self.item_field_names = []
            for field_name in self.field_names:
                if dataset.field2source[field_name] in {
                        FeatureSource.USER, FeatureSource.USER_ID
                }:
                    self.user_field_names.append(field_name)
                elif dataset.field2source[field_name] in {
                        FeatureSource.ITEM, FeatureSource.ITEM_ID
                }:
                    self.item_field_names.append(field_name)
            self.field_names = self.user_field_names + self.item_field_names
            self.user_token_field_num = 0
            self.user_float_field_num = 0
            self.user_token_seq_field_num = 0
            for field_name in self.user_field_names:
                if dataset.field2type[field_name] == 'token':
                    self.user_token_field_num += 1
                elif dataset.field2type[field_name] == 'token_seq':
                    self.user_token_seq_field_num += 1
                else:
                    self.user_float_field_num += dataset.num(field_name)
            self.item_token_field_num = 0
            self.item_float_field_num = 0
            self.item_token_seq_field_num = 0
            for field_name in self.item_field_names:
                if dataset.field2type[field_name] == 'token':
                    self.item_token_field_num += 1
                elif dataset.field2type[field_name] == 'token_seq':
                    self.item_token_seq_field_num += 1
                else:
                    self.item_float_field_num += dataset.num(field_name)

        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue
            if dataset.field2type[field_name] == 'token':
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == 'token_seq':
                self.token_seq_field_names.append(field_name)
                self.token_seq_field_dims.append(dataset.num(field_name))
            else:
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))
            self.num_feature_field += 1

        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array(
                (0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long)
            self.token_embedding_table = FMEmbedding(self.token_field_dims,
                                                     self.token_field_offsets,
                                                     self.embedding_size)
        if len(self.float_field_dims) > 0:
            self.float_embedding_table = nn.Embedding(
                np.sum(self.float_field_dims, dtype=np.int32),
                self.embedding_size)
        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(
                    nn.Embedding(token_seq_field_dim, self.embedding_size))

        self.first_order_linear = FMFirstOrderLinear(config,
                                                     dataset,
                                                     embed=False)

    def embed_float_fields(self, float_fields, embed=False):
        """Embed the float feature columns

        Args:
            float_fields (torch.FloatTensor): The input dense tensor. shape of [batch_size, num_float_field]
            embed (bool): Return the embedding of columns or just the columns itself. default=True

        Returns:
            torch.FloatTensor: The result embedding tensor of float columns.
        """
        # input Tensor shape : [batch_size, num_float_field]
        if float_fields is None:
            return float_fields
        if not embed:
            if float_fields.dim() == 2:
                return float_fields.unsqueeze(1)
            else:
                return float_fields

        num_float_field = float_fields.shape[1]
        # [batch_size, num_float_field]
        index = torch.arange(
            0, num_float_field).unsqueeze(0).expand_as(float_fields).long().to(
                self.device)

        # [batch_size, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table(index)
        float_embedding = torch.mul(float_embedding, float_fields.unsqueeze(2))

        return float_embedding

    def embed_token_fields(self, token_fields):
        """Embed the token feature columns

        Args:
            token_fields (torch.LongTensor): The input tensor. shape of [batch_size, num_token_field]

        Returns:
            torch.FloatTensor: The result embedding tensor of token columns.
        """
        # input Tensor shape : [batch_size, num_token_field]
        if token_fields is None:
            return None
        # [batch_size, num_token_field, embed_dim]
        token_embedding = self.token_embedding_table(token_fields)

        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields, mode='mean'):
        """Embed the token feature columns

        Args:
            token_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of token sequence columns.
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[i]
            mask = token_seq_field != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            token_seq_embedding = embedding_table(
                token_seq_field)  # [batch_size, seq_len, embed_dim]

            mask = mask.unsqueeze(2).expand_as(
                token_seq_embedding)  # [batch_size, seq_len, embed_dim]
            if mode == 'max':
                masked_token_seq_embedding = token_seq_embedding - (
                    1 - mask) * 1e9  # [batch_size, seq_len, embed_dim]
                result = torch.max(masked_token_seq_embedding,
                                   dim=1,
                                   keepdim=True)  # [batch_size, 1, embed_dim]
            elif mode == 'sum':
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding,
                                   dim=1,
                                   keepdim=True)  # [batch_size, 1, embed_dim]
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding,
                                   dim=1)  # [batch_size, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result,
                                   value_cnt + eps)  # [batch_size, embed_dim]
                result = result.unsqueeze(1)  # [batch_size, 1, embed_dim]
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(
                fields_result,
                dim=1)  # [batch_size, num_token_seq_field, embed_dim]

    def double_tower_embed_input_fields(self, interaction):
        """Embed the whole feature columns in a double tower way.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of token sequence columns in the second part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the second part.

        """
        if not self.double_tower:
            raise RuntimeError(
                'Please check your model hyper parameters and set \'double tower\' as True'
            )
        sparse_embedding, dense_embedding = self.embed_input_fields(
            interaction)
        if dense_embedding is not None:
            first_dense_embedding, second_dense_embedding = \
                torch.split(dense_embedding, [self.user_float_field_num, self.item_float_field_num], dim=1)
        else:
            first_dense_embedding, second_dense_embedding = None, None

        if sparse_embedding is not None:
            sizes = [
                self.user_token_seq_field_num, self.item_token_seq_field_num,
                self.user_token_field_num, self.item_token_field_num
            ]
            first_token_seq_embedding, second_token_seq_embedding, first_token_embedding, second_token_embedding = \
                torch.split(sparse_embedding, sizes, dim=1)
            first_sparse_embedding = torch.cat(
                [first_token_seq_embedding, first_token_embedding], dim=1)
            second_sparse_embedding = torch.cat(
                [second_token_seq_embedding, second_token_embedding], dim=1)
        else:
            first_sparse_embedding, second_sparse_embedding = None, None

        return first_sparse_embedding, first_dense_embedding, second_sparse_embedding, second_dense_embedding

    def embed_input_fields(self, interaction):
        """Embed the whole feature columns.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns.
            torch.FloatTensor: The embedding tensor of float sequence columns.
        """
        float_fields = []
        for field_name in self.float_field_names:
            float_fields.append(interaction[field_name].float(
            ) if len(interaction[field_name].shape
                     ) == 2 else interaction[field_name].float().unsqueeze(1))
        if len(float_fields) > 0:
            float_fields = torch.cat(float_fields,
                                     dim=1)  # [batch_size, num_float_field]
        else:
            float_fields = None
        # [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        float_fields_embedding = self.embed_float_fields(float_fields)

        token_fields = []
        for field_name in self.token_field_names:
            token_fields.append(interaction[field_name].unsqueeze(1))
        if len(token_fields) > 0:
            token_fields = torch.cat(token_fields,
                                     dim=1)  # [batch_size, num_token_field]
        else:
            token_fields = None
        # [batch_size, num_token_field, embed_dim] or None
        token_fields_embedding = self.embed_token_fields(token_fields)

        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            token_seq_fields.append(interaction[field_name])
        # [batch_size, num_token_seq_field, embed_dim] or None
        token_seq_fields_embedding = self.embed_token_seq_fields(
            token_seq_fields)

        if token_fields_embedding is None:
            sparse_embedding = token_seq_fields_embedding
        else:
            if token_seq_fields_embedding is None:
                sparse_embedding = token_fields_embedding
            else:
                sparse_embedding = torch.cat(
                    [token_fields_embedding, token_seq_fields_embedding],
                    dim=1)

        dense_embedding = float_fields_embedding

        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding


class HModel(ContextModel):
    model_type = ModelType.HETERO

    def __init__(self, config, dataset):
        BaseModel.__init__(self)
        self.USER_ID = dataset.config['USER_ID_FIELD']
        self.ITEM_ID = dataset.config['ITEM_ID_FIELD']
        self.LABEL = dataset.config["LABEL_FIELD"]
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.device = config.get('device', 'cpu')
        self.double_tower = config.get('double_tower', False)
        self.embedding_size = config.get('embedding_size', 64)
        self.token_emb_size = config.get('token_emb_size', 32)

        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []

        self.num_feature_field = 0

        self.field_names = dataset.fields()

        if self.double_tower:
            self.user_field_names = []
            self.item_field_names = []
            for field_name in self.field_names:
                if dataset.field2source[field_name] in {
                        FeatureSource.USER, FeatureSource.USER_ID
                }:
                    self.user_field_names.append(field_name)
                elif dataset.field2source[field_name] in {
                        FeatureSource.ITEM, FeatureSource.ITEM_ID
                }:
                    self.item_field_names.append(field_name)
            self.field_names = self.user_field_names + self.item_field_names
            self.user_token_field_num = 0
            self.user_float_field_num = 0
            self.user_token_seq_field_num = 0
            for field_name in self.user_field_names:
                if dataset.field2type[field_name] == 'token':
                    self.user_token_field_num += 1
                elif dataset.field2type[field_name] == 'token_seq':
                    self.user_token_seq_field_num += 1
                else:
                    self.user_float_field_num += dataset.num(field_name)
            self.item_token_field_num = 0
            self.item_float_field_num = 0
            self.item_token_seq_field_num = 0
            for field_name in self.item_field_names:
                if dataset.field2type[field_name] == 'token':
                    self.item_token_field_num += 1
                elif dataset.field2type[field_name] == 'token_seq':
                    self.item_token_seq_field_num += 1
                else:
                    self.item_float_field_num += dataset.num(field_name)

        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue
            if dataset.field2type[field_name] == 'token':
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == 'token_seq':
                self.token_seq_field_names.append(field_name)
                self.token_seq_field_dims.append(dataset.num(field_name))
            else:
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))
            self.num_feature_field += 1

        self.token_embedding_table = {}
        for field_name, field_dim in zip(self.token_field_names,
                                         self.token_field_dims):
            self.token_embedding_table[field_name] = nn.Embedding(
                field_dim, self.token_emb_size).to(self.device)

        if len(self.float_field_dims) > 0:
            self.float_embedding_table = nn.Embedding(
                np.sum(self.float_field_dims, dtype=np.int32),
                self.embedding_size)
        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(
                    nn.Embedding(token_seq_field_dim, self.embedding_size))

        self.first_order_linear = FMFirstOrderLinear(config,
                                                     dataset,
                                                     embed=False)


if __name__ == '__main__':
    pass
