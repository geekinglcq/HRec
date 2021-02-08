# -*- coding:utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as fn

from torch.nn.init import normal_


class MLPLayers(nn.Module):
    r""" MLPLayers
    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'
                      candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'
    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """
    def __init__(self,
                 layers,
                 dropout=0,
                 activation='relu',
                 bn=False,
                 init_method=None):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
                zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == 'norm':
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def activation_layer(activation_name='relu', emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_name.lower() == 'tanh':
            activation = nn.Tanh()
        elif activation_name.lower() == 'relu':
            activation = nn.ReLU()
        elif activation_name.lower() == 'leakyrelu':
            activation = nn.LeakyReLU()
        elif activation_name.lower() == 'none':
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(
                activation_name))
    return activation


class BaseFactorizationMachine(nn.Module):
    r"""Calculate FM result over the embeddings

    Args:
        reduce_sum: bool, whether to sum the result, default is True.

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.

    Output
        output: tensor, A 3D tensor with shape: ``(batch_size,1)`` or ``(batch_size, embed_dim)``.
    """
    def __init__(self, reduce_sum=True):
        super(BaseFactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, input_x):
        square_of_sum = torch.sum(input_x, dim=1)**2
        sum_of_square = torch.sum(input_x**2, dim=1)
        output = square_of_sum - sum_of_square
        if self.reduce_sum:
            output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        return output


class FMEmbedding(nn.Module):
    r""" Embedding for token fields.

    Args:
        field_dims: list, the number of tokens in each token fields
        offsets: list, the dimension offset of each token field
        embed_dim: int, the dimension of output embedding vectors

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size)``.

    Return:
        output: tensor,  A 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.
    """
    def __init__(self, field_dims, offsets, embed_dim):
        super(FMEmbedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = offsets

    def forward(self, input_x):
        input_x = input_x + input_x.new_tensor(self.offsets).unsqueeze(0)
        output = self.embedding(input_x)
        return output


class FMFirstOrderLinear(nn.Module):
    """Calculate the first order score of the input features.
    This class is a member of ContextRecommender, you can call it easily when inherit ContextRecommender.

    """
    def __init__(self, config, dataset, output_dim=1, embed=True):

        super(FMFirstOrderLinear, self).__init__()
        self.field_names = dataset.fields()
        self.LABEL = dataset.config['LABEL_FIELD']
        self.device = config['device']
        self.embed = embed
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []
        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue
            if dataset.field2type[field_name] == "token":
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == "token_seq":
                self.token_seq_field_names.append(field_name)
                self.token_seq_field_dims.append(dataset.num(field_name))
            else:
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))
        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array(
                (0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long)
            self.token_embedding_table = FMEmbedding(self.token_field_dims,
                                                     self.token_field_offsets,
                                                     output_dim)
        if len(self.float_field_dims) > 0:
            self.float_embedding_table = nn.Embedding(
                np.sum(self.float_field_dims, dtype=np.int32), output_dim)
        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(
                    nn.Embedding(token_seq_field_dim, output_dim))

        self.bias = nn.Parameter(torch.zeros((output_dim, )),
                                 requires_grad=True)

    def embed_float_fields(self, float_fields, embed=True):
        """Calculate the first order score of float feature columns

        Args:
            float_fields (torch.FloatTensor): The input tensor. shape of [batch_size, num_float_field]

        Returns:
            torch.FloatTensor: The first order score of float feature columns
        """
        # input Tensor shape : [batch_size, num_float_field]
        if float_fields is None:
            return float_fields
        if not embed:
            if float_fields.dim() == 2:
                return float_fields.unsqueeze(1)

        num_float_field = float_fields.shape[1]
        # [batch_size, num_float_field]
        index = torch.arange(
            0, num_float_field).unsqueeze(0).expand_as(float_fields).long().to(
                self.device)

        # [batch_size, num_float_field, output_dim]
        float_embedding = self.float_embedding_table(index)
        float_embedding = torch.mul(float_embedding, float_fields.unsqueeze(2))

        # [batch_size, 1, output_dim]
        float_embedding = torch.sum(float_embedding, dim=1, keepdim=True)

        return float_embedding

    def embed_token_fields(self, token_fields):
        """Calculate the first order score of token feature columns

        Args:
            token_fields (torch.LongTensor): The input tensor. shape of [batch_size, num_token_field]

        Returns:
            torch.FloatTensor: The first order score of token feature columns
        """
        # input Tensor shape : [batch_size, num_token_field]
        if token_fields is None:
            return None
        # [batch_size, num_token_field, embed_dim]
        token_embedding = self.token_embedding_table(token_fields)
        # [batch_size, 1, output_dim]
        token_embedding = torch.sum(token_embedding, dim=1, keepdim=True)

        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields):
        """Calculate the first order score of token sequence feature columns

        Args:
            token_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]

        Returns:
            torch.FloatTensor: The first order score of token sequence feature columns
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[i]
            mask = token_seq_field != 0  # [batch_size, seq_len]
            mask = mask.float()
            # value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            token_seq_embedding = embedding_table(
                token_seq_field)  # [batch_size, seq_len, output_dim]

            mask = mask.unsqueeze(2).expand_as(
                token_seq_embedding)  # [batch_size, seq_len, output_dim]
            pdb.set_trace()
            masked_token_seq_embedding = token_seq_embedding * mask.float()
            result = torch.sum(masked_token_seq_embedding, dim=1,
                               keepdim=True)  # [batch_size, 1, output_dim]

            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.sum(torch.cat(fields_result, dim=1),
                             dim=1,
                             keepdim=True)  # [batch_size, 1, output_dim]

    def forward(self, interaction):
        total_fields_embedding = []
        float_fields = []
        for field_name in self.float_field_names:
            float_fields.append(interaction[field_name]
                                if len(interaction[field_name].shape) ==
                                2 else interaction[field_name].unsqueeze(1))

        if len(float_fields) > 0:
            float_fields = torch.cat(float_fields,
                                     dim=1)  # [batch_size, num_float_field]
        else:
            float_fields = None

        # [batch_size, 1, output_dim] or None
        float_fields_embedding = self.embed_float_fields(float_fields,
                                                         embed=self.embed)

        if float_fields_embedding is not None:
            total_fields_embedding.append(float_fields_embedding.float())

        token_fields = []
        for field_name in self.token_field_names:
            token_fields.append(interaction[field_name].unsqueeze(1))
        if len(token_fields) > 0:
            token_fields = torch.cat(token_fields,
                                     dim=1)  # [batch_size, num_token_field]
        else:
            token_fields = None
        # [batch_size, 1, output_dim] or None
        token_fields_embedding = self.embed_token_fields(token_fields)
        if token_fields_embedding is not None:
            total_fields_embedding.append(token_fields_embedding)

        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            token_seq_fields.append(interaction[field_name])
        # [batch_size, 1, output_dim] or None
        token_seq_fields_embedding = self.embed_token_seq_fields(
            token_seq_fields)
        if token_seq_fields_embedding is not None:
            total_fields_embedding.append(token_seq_fields_embedding)

        if self.embed:
            return torch.sum(torch.cat(total_fields_embedding, dim=1),
                             dim=1) + self.bias  # [batch_size, output_dim]
        else:
            return torch.sum(torch.cat(total_fields_embedding, dim=2),
                             dim=2) + self.bias


class AttLayer(nn.Module):
    """Calculate the attention signal(weight) according the input tensor.

    Args:
        infeatures (torch.FloatTensor): A 3D input tensor with shape of[batch_size, M, embed_dim].

    Returns:
        torch.FloatTensor: Attention weight of input. shape of [batch_size, M].
    """
    def __init__(self, in_dim, att_dim):
        super(AttLayer, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.w = torch.nn.Linear(in_features=in_dim,
                                 out_features=att_dim,
                                 bias=False)
        self.h = nn.Parameter(torch.randn(att_dim), requires_grad=True)

    def forward(self, infeatures):
        att_singal = self.w(infeatures)  # [batch_size, M, att_dim]
        att_singal = fn.relu(att_singal)  # [batch_size, M, att_dim]

        att_singal = torch.mul(att_singal, self.h)  # [batch_size, M, att_dim]
        att_singal = torch.sum(att_singal, dim=2)  # [batch_size, M]
        att_singal = fn.softmax(att_singal, dim=1)  # [batch_size, M]

        return att_singal


class RegLoss(nn.Module):
    """ RegLoss, L2 regularization on model parameters

    """
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


def meshgrid(x, y=None):
    if y is None:
        y = x
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    m, n = x.size(0), y.size(0)
    grid_x = x[None].expand(n, m)
    grid_y = y[:, None].expand(n, m)
    return grid_x, grid_y


def get_all_combination(x, dim=0, r=2, device='cpu'):
    """
    Get all combination of given x.
    Input:
        x: tensor
        dim:
        r: the number of elements to combine
    """
    xs = torch.arange(x.shape[dim], device=device)
    idx = torch.combinations(xs, r=r)
    a = x.index_select(dim, idx[:, 0])
    b = x.index_select(dim, idx[:, 1])
    return a, b


def combinations(x, y, dim, all=True, n=None):
    """
    Given the tensor x and y, return a list of pair of tensor where the first tensor sampled
    from x and the second tensor sampled from y.
    Input:
        x,y: tensors shared the same dim
        dim: sample from which dimension
        TODO all: if True, return all of possible combinations
        n: the num of samples should return. If all is True, will ignore n
    """
    xs = torch.arange(x.shape[dim])
    ys = torch.arange(y.shape[dim])
    grid_x, grid_y = meshgrid(xs, ys)
    # select n random elements from the
    # cartesian product
    sampled = torch.randperm(grid_x.numel())[:n]
    indices_x = grid_x.take(sampled)
    indices_y = grid_y.take(sampled)
    # get from the indices
    return x.index_select(dim, indices_x), y.index_select(dim, indices_y)


def set_kernel_layer(name):

    if name == 'gaussian':
        return gaussian_rbf_layer


def gaussian_rbf_layer(x, y):
    dist = torch.pairwise_distance(x, y, 2)
    return torch.exp(-0.5 * dist.pow(2))
