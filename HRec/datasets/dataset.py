# -*- coding:utf-8 -*-

import copy
import numpy as np
import pandas as pd

import os
import torch
import logging
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from .enum_type import FeatureSource as FS
from .enum_type import item_type_dict
from scipy.sparse import coo_matrix


class DataSet(object):
    def __init__(self, config, restore_path=None):

        self.config = config
        self._init_setting()
        if restore_path is None:
            self._load_feats()
        else:
            self._restore_saved_dataset(restore_path)
        self._preprocessing()

    def _preprocessing(self):
        self._normalize()
        self._reID(self.uid_field)
        self._reID(self.iid_field)

    def _load_feats(self):
        """Load features from raw features"""
        self.user_feat = self._load_meta_feats(self.config["user_feat_path"],
                                               FS.USER, "user_id")
        self.item_feat = self._load_meta_feats(self.config["item_feat_path"],
                                               FS.ITEM, "item_id")
        self.inter_feat = pd.read_csv(self.config["inter_feat_path"]).sample(
            frac=1, random_state=28)
        if self.single:
            self._prepare_single_data()

        self.inter_feat = pd.merge(self.inter_feat,
                                   self.item_feat,
                                   on=self.iid_field,
                                   how='left')
        self.user_num = len(self.user_feat)
        self.item_num = len(self.item_feat)

    def _prepare_single_data(self):
        self.use_type = self.config.get('single_type', 0)
        self.item_feat = self.item_feat[self.item_feat[self.itype_field] ==
                                        self.use_type]
        mask = self.inter_feat[self.iid_field].isin(
            self.item_feat[self.iid_field])
        self.inter_feat = self.inter_feat[mask]

    def _normalize(self):
        """
        Normalized the field in config
        """

        if self.config.get('normalize_field', None):
            fields = self.config['normalize_field']
            for field in fields:
                ftype = self.field2type[field]
                if ftype != 'float':
                    raise ValueError(
                        'Field [{}] doesn\'t need to normalize'.format(field))
        else:
            return
        for feat in self.feat_list:
            for field in feat:
                if field not in fields:
                    continue
                ftype = self.field2type[field]
                if ftype == "float":
                    lst = feat[field].values
                    mx, mn = max(lst), min(lst)
                    if mx == mn:
                        raise ValueError(
                            'All the same value in [{}] from [{}_feat]'.format(
                                field, feat))
                    feat[field] = (lst - mn) / (mx - mn)

    def _reID(self, field):
        """
        Re-ID the token-type feature, save the id map in self.field2token_id
        """
        self.logger.info(f'ReID field {field}.')
        ftype = self.field2type.get(field)
        assert ftype == 'token'
        source = self.field2source.get(field)
        if source == 'item' or source is FS.ITEM_ID:
            dataframe = self.item_feat
        elif source == 'user' or source is FS.USER_ID:
            dataframe = self.user_feat
        else:
            dataframe = self.inter_feat
        id_map = {v: k for k, v in enumerate(dataframe[field].unique())}
        self.field2token_id[field].update(id_map)
        dataframe[field] = dataframe[field].map(id_map)
        if source in ['item', 'user', FS.ITEM_ID, FS.USER_ID]:
            if field in self.inter_feat:
                self.inter_feat[field] = self.inter_feat[field].map(id_map)

    def _init_setting(self):
        self.logger = logging.getLogger()
        self.name = self.config['name']
        print(self.config)
        self.uid_field = self.config["USER_ID_FIELD"]
        self.iid_field = self.config["ITEM_ID_FIELD"]
        self.label_field = self.config["LABEL_FIELD"]
        self.itype_field = self.config["TYPE_FIELD"]
        self.field2type = {}
        self.field2source = {}
        self.field2id_token = defaultdict(dict)
        self.field2token_id = defaultdict(dict)
        self.user_feat_fields = []
        self.item_feat_fields = []
        self.pin_mem = self.config.get('pin_mem', False)
        self.single = self.config.get('single', False)

        for feat_name, feat_value in self.config['feat'].items():
            source = feat_value['source']
            self.field2type[feat_name] = feat_value['type']
            self.field2source[feat_name] = feat_value['source']
            if source == 'user' and feat_name != self.uid_field:
                self.user_feat_fields.append(feat_name)
            if source.startswith('item') and feat_name != self.iid_field:
                self.item_feat_fields.append(feat_name)

    def num(self, field):
        if field == self.uid_field:
            return self.user_num
        if field == self.iid_field:
            return self.item_num
        if field not in self.field2type:
            raise ValueError('field {} not in dataset'.format(field))
        # if field not in self.field2token_id:
        # raise ValueError('field {} is not token type'.format(field))
        if len(self.field2token_id[field]) == 0:
            if field in self.user_feat_fields:
                return len(self.user_feat[field].unique())
            elif field in self.item_feat_fields:
                return len(self.item_feat[field].unique())

        return len(self.field2token_id[field])

    def init_data_loader(self, batch_size=256, num_workers=2):
        if hasattr(self, 'train_inter_subset'):
            self.train_data_loader = DataLoader(self.train_inter_subset,
                                                batch_size=batch_size,
                                                pin_memory=self.pin_mem,
                                                num_workers=num_workers)
            self.val_data_loader = DataLoader(self.val_inter_subset,
                                              pin_memory=self.pin_mem,
                                              batch_size=batch_size,
                                              num_workers=num_workers)
            self.test_data_loader = DataLoader(self.test_inter_subset,
                                               pin_memory=self.pin_mem,
                                               batch_size=batch_size,
                                               num_workers=num_workers)

    def _load_meta_feats(self, path, source, field_name):

        if os.path.isfile(path):
            feat = pd.read_csv(path)
            if field_name in self.field2source:
                self.field2source[field_name] = FS(source.value + '_id')
        else:
            raise ValueError("Dataset file {} not found.".format(path))
        return feat

    def join(self, df):
        """Given interaction feature, join user feature into it.

        Args:
            df (pandas.DataFrame): Interaction feature to be joint.

        Returns:
            pandas.DataFrame: Interaction feature after joining operation.
        """
        if self.user_feat is not None and self.uid_field in df:
            df = pd.merge(df,
                          self.user_feat,
                          on=self.uid_field,
                          how='left',
                          suffixes=('_inter', '_user'))

        return df

    def join_interaction(self):
        self.inter_feat = self.join(self.inter_feat)

    def train_val_test_split(self,
                             ratios=[0.7, 0.2, 0.1],
                             group_by=None,
                             context=False):
        assert len(ratios) == 3
        train, val, test = self.split_by_ratio(ratios,
                                               group_by=group_by,
                                               create_new_dataset=False)
        if context:
            user_fs = self.user_feat_fields
            item_fs = self.item_feat_fields
        else:
            user_fs, item_fs = None, None
        self.train_inter_subset = SubSet(train, self.uid_field, self.iid_field,
                                         self.itype_field, self.label_field,
                                         user_fs, item_fs)
        self.val_inter_subset = SubSet(val, self.uid_field, self.iid_field,
                                       self.itype_field, self.label_field,
                                       user_fs, item_fs)
        self.test_inter_subset = SubSet(test, self.uid_field, self.iid_field,
                                        self.itype_field, self.label_field,
                                        user_fs, item_fs)
        self.all_inter_feat = self.inter_feat
        self.logger.info(
            "Replace interaction features with train interaction fatures.")
        self.logger.info(
            "Interaction features are stored in self.all_inter_feat")
        self.inter_feat = train

    def split_by_ratio(self,
                       ratios,
                       group_by=None,
                       create_new_dataset=False,
                       df=None):
        """
        Args:
            ratios: a list of ratio, e.g. [0.5, 0.3, 0.2]
            group_by: field name that used to group data before splitting
            create_new_dataset: bool value,
                True, will return a list of new Dataset class
                False, will return a list of feature dataframe
        Returns:
            dataset: a list of datasets/dataframes
        """
        ratio_sum = sum(ratios)
        ratios = [ratio / ratio_sum for ratio in ratios]

        if df is None:
            df = self.inter_feat
        if group_by is None:
            total_len = df.shape[0]
            cnts = [int(ratio * total_len) for ratio in ratios]
            cnts[0] = total_len - sum(cnts[1:])
            split_ids = np.cumsum(cnts)[:-1].tolist()
            new_index = [
                range(st, ed)
                for st, ed in zip([0] + split_ids, split_ids + [total_len])
            ]
        else:
            # TODO group by key
            raise NotImplementedError
            return
        new_df = [df.loc[index].reset_index(drop=True) for index in new_index]
        if create_new_dataset:
            return [self.copy(subset) for subset in new_df]
        else:
            return new_df

    def split_by_ratio_sampled(self, ratios, create_new_dataset=False):

        sample_ratio = self.config['sample']
        sampled = []
        for kind in self.types:
            r = sample_ratio.get(kind, 1.0)
            kind_id = item_type_dict[kind]
            new_ratio = [r * ratios[0], ratios[1], ratios[2]]
            df = self.inter_feat[self.inter_feat[self.itype_field] ==
                                 kind_id].reset_index(drop=True).copy()
            sampled.append(self.split_by_ratio(new_ratio, df=df))
        train_val_test = []
        for i in range(3):
            new_df = pd.concat([k[i] for k in sampled], ignore_index=True)
            new_df = new_df.sample(frac=1.).reset_index(drop=True)
            train_val_test.append(new_df)
        return train_val_test

    def copy(self, inter_feat):
        """ copy the dataset itself but replace the interaction features"""
        nxt = copy.copy(self)
        nxt.inter_feat = inter_feat
        return nxt

    def __len__(self):
        return len(self.inter_feat)

    def _check_attrs(self, *attr_names):
        for attr_name in attr_names:
            if getattr(self, attr_name, None) is None:
                raise ValueError(f'{attr_name} is not in this model.')

    def _history_matrix(self, row, value_field=None):
        """Get dense matrix describe user/item's history interaction records.
        `history_matrix[i]` represents `i`'s history interacted item_id.

        `history_value[i]` represents `i`'s history interaction records' values
            `0` if `value_field = None`.
        `history_len[i]` represents number of `i`'s history interaction records
        `0` is used as padding.
        Args:
            row (str): `user` or `item`.
            value_field (str, optional): Data of matrix,
                which should exist in `self.inter_feat`.
                Defaults to `None`.

        Returns:
        tuple:
            - History matrix (torch.Tensor)
            - History values matrix (torch.Tensor)
            - History length matrix (torch.Tensor)
        """
        self._check_attrs('uid_field', 'iid_field')

        user_ids, item_ids = self.inter_feat[
            self.uid_field].values, self.inter_feat[self.iid_field].values
        if value_field is None:
            values = np.ones(len(self.inter_feat))
        else:
            if value_field not in self.inter_feat.columns:
                raise ValueError(
                    'value_field {} should be one of `inter_feat`\'s features.'
                    .format(value_field))
            values = self.inter_feat[value_field].values

        if row == 'user':
            row_num = self.user_num
            row_ids, col_ids = user_ids, item_ids
        else:
            row_num = self.item_num
            row_ids, col_ids = item_ids, user_ids

        history_len = np.zeros(row_num, dtype=np.int64)
        for row_id in row_ids:
            history_len[row_id] += 1

        col_num = np.max(history_len)

        history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
        history_value = np.zeros((row_num, col_num))
        history_len[:] = 0
        for row_id, value, col_id in zip(row_ids, values, col_ids):
            history_matrix[row_id, history_len[row_id]] = col_id
            history_value[row_id, history_len[row_id]] = value
            history_len[row_id] += 1

        return torch.LongTensor(history_matrix), torch.FloatTensor(
            history_value), torch.LongTensor(history_len)

    def history_item_matrix(self, value_field=None):
        """Get dense matrix describe user's history interaction records.
        """
        return self._history_matrix(row='user', value_field=value_field)

    def history_user_matrix(self, value_field=None):
        """Get dense matrix describe item's history interaction records.
        """
        return self._history_matrix(row='item', value_field=value_field)

    def inter_matrix(self, form='coo', value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.
        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.
        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.uid_field or not self.iid_field:
            raise ValueError(
                'dataset doesn\'t exist uid/iid, thus can not converted to sparse matrix'
            )
        return self._create_sparse_matrix(self.inter_feat, self.uid_field,
                                          self.iid_field, form, value_field)

    def _create_sparse_matrix(self,
                              df_feat,
                              source_field,
                              target_field,
                              form='coo',
                              value_field=None):
        """Get sparse matrix that describe relations between two fields.
        Args:
            df_feat (pandas.DataFrame): Feature where src and tgt exist.
            source_field (str): Source field
            target_field (str): Target field
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        src = df_feat[source_field].values
        tgt = df_feat[target_field].values
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError(
                    'value_field [{}] should be one of `df_feat`\'s features.'.
                    format(value_field))
            data = df_feat[value_field].values
        mat = coo_matrix(
            (data, (src, tgt)),
            shape=(self.num(source_field), self.num(target_field)))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError(
                'sparse matrix format [{}] has not been implemented.'.format(
                    form))

    def fields(self):
        """
            Return a list of  field names
        """
        return list(self.field2type.keys())


class SubSet(Dataset):
    """SubSet used to generate dataloader"""
    def __init__(self,
                 dataframe,
                 uid_field,
                 iid_field,
                 label_field,
                 itype_field=None,
                 u_feat_fields=None,
                 i_feat_fields=None):
        self.df = dataframe
        self.uid = uid_field
        self.iid = iid_field
        self.label = label_field
        self.itype = itype_field
        self.u_feat = u_feat_fields
        self.i_feat = i_feat_fields

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        res = {}
        for i in [self.uid, self.iid, self.label, self.itype]:
            if i is not None:
                res[i] = self.df.iloc[idx][i]

        feat_fields = []
        for i in [self.u_feat, self.i_feat]:
            if i is not None:
                feat_fields.extend(i)
        for i in feat_fields:
            res[i] = self.df.iloc[idx][i]
        return res

    # def to(self, device):
    #     """
    #     Convert data to given device.
    #     """
    #     for k in
