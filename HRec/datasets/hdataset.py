# -*- coding:utf-8 -*-
# ###########################
# File Name: hdataset.py
# Author: geekinglcq
# Mail: lcqgeek@live.com
# Created Time: 2020-12-28 20:17:47
# ###########################

import pandas as pd

import os
import logging
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from .enum_type import FeatureSource as FS
from .enum_type import item_type_dict
from .dataset import DataSet, SubSet


class HDataSet(DataSet):
    """
    Dataset used for heterogenous items
    """
    def __init__(self, config, restore_path=None):
        self.config = config
        self._init_setting()
        if restore_path is None:
            self._load_feats()
        else:
            # TODO
            pass
        self._preprocessing()

    def _load_feats(self):
        self.user_feat = self._load_meta_feats(self.config["user_feat_path"],
                                               FS.USER, "user_id")
        self.item_feat = self._load_item_feats(self.config["item_feat_path"],
                                               FS.ITEM)
        self.inter_feat = pd.read_csv(self.config["inter_feat_path"]).sample(
            frac=1, random_state=28)
        mask = None
        if len(self.types) < 3:
            for item_type, item_feat in self.item_feat.items():
                new_mask = self.inter_feat[self.iid_field].isin(
                    item_feat[self.iid_field])
                if mask is not None:
                    mask = mask | new_mask
                else:
                    mask = new_mask
            self.inter_feat = self.inter_feat[mask]
        self.h_inter_feat = {}
        self.user_num = len(self.user_feat)
        self.item_num = sum([len(i) for i in self.item_feat.values()])
        self.item_nums = {k: len(v) for k, v in self.item_feat.items()}
        print(f'user num: {self.user_num}')
        print(f'item num: {self.item_num}')
        print(f'item nums: {self.item_nums}')

    def _preprocessing(self):
        self._normalize()
        if len(self.types) < 3:
            self._reID(self.iid_field)
            self._reID(self.uid_field)

    def _load_item_feats(self, paths, source):
        item_feat = {}
        for item_type, item_path in paths.items():
            if item_type not in self.types:
                continue
            if os.path.isfile(item_path):
                feat = pd.read_csv(item_path)
                item_feat[item_type] = feat
            else:
                raise ValueError("Dataset file not fountd.")
        return item_feat

    def _init_setting(self):
        self.logger = logging.getLogger()
        self.name = self.config['name']
        print(self.config)
        self.uid_field = self.config["USER_ID_FIELD"]
        self.iid_field = self.config["ITEM_ID_FIELD"]
        self.label_field = self.config["LABEL_FIELD"]
        self.itype_field = self.config["TYPE_FIELD"]
        self.types = self.config["type"]
        self.field2type = {}
        self.field2source = {}
        self.field2id_token = defaultdict(dict)
        self.field2token_id = defaultdict(dict)
        self.user_feat_fields = []
        self.item_feat_fields = defaultdict(list)

        for feat_name, feat_value in self.config['feat'].items():
            source = feat_value['source']
            self.field2type[feat_name] = feat_value['type']
            self.field2source[feat_name] = feat_value['source']
            if source == 'user' and feat_name != self.uid_field:
                self.user_feat_fields.append(feat_name)
            if source.startswith('item') and feat_name != self.iid_field:
                item_type = source.split("_")[1]
                if item_type in self.types:
                    self.item_feat_fields[item_type].append(feat_name)

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
            else:
                for item_type, item_feat_fields in self.item_feat_fields.items(
                ):
                    if field in item_feat_fields:
                        return len(self.item_feat[item_type][field].unique())
        return len(self.field2token_id[field])

    def _reID(self, field):
        """
        Re-ID the token-type feature, save the id map in self.field2token_id
        """
        self.logger.info(f'ReID field {field}.')
        ftype = self.field2type.get(field)
        assert ftype == 'token'
        source = self.field2source.get(field)
        if type(source) is str and source.startswith("item_"):
            item_type = source.split("_")[1]
            dataframe = self.item_feat[item_type]
        elif source is FS.ITEM_ID or source == "item":
            dataframe = pd.concat(list(self.item_feat.values()), join='inner')
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
            for item_type, item_feat in self.item_feat.items():
                if field in item_feat:
                    item_feat[field] = item_feat[field].map(id_map)

    def join(self, df):
        """
        Join user/item features to interactions.
        """
        if self.user_feat is not None and self.uid_field in df:
            df = pd.merge(df,
                          self.user_feat,
                          on=self.uid_field,
                          how='left',
                          suffixes=('_inter', '_user'))
        if self.item_feat is not None and self.iid_field in df:
            for item_type, item_feat in self.item_feat.items():
                df = pd.merge(df,
                              item_feat,
                              on=self.iid_field,
                              how='left',
                              suffixes=(f'_{item_type}', '_inter'))
            type_c = [i for i in df.columns if i.startswith(self.itype_field)]
            df[self.itype_field] = df[type_c].agg(sum, axis=1)
        return df

    def join_interaction(self):
        self.inter_feat = self.join(self.inter_feat)
        if 'sample' in self.config:
            sample_ratio = self.config['sample']
            sampled = []
            for kind in self.types:
                ratio = sample_ratio.get(kind, 1.0)
                kind_id = item_type_dict[kind]
                # preverse the data for val & test
                new_df = self.inter_feat[self.inter_feat['type'] ==
                                         kind_id].sample(frac=ratio * 0.7 +
                                                         0.3,
                                                         random_state=16)
                print(kind, kind_id, ratio, new_df.shape)
                sampled.append(new_df)
            self.inter_feat = pd.concat(sampled, ignore_index=True)
            self.inter_feat = self.inter_feat.sample(frac=1.).reset_index(
                drop=True)

    def train_val_test_split(self,
                             ratios=[0.7, 0.2, 0.1],
                             group_by=None,
                             **kwargs):
        assert len(ratios) == 3
        if 'sample' in self.config:
            train, val, test = self.split_by_ratio_sampled(
                ratios, create_new_dataset=False)
        else:
            train, val, test = self.split_by_ratio(ratios,
                                                   group_by=group_by,
                                                   create_new_dataset=False)
        user_fs = self.user_feat_fields
        item_fs = self.item_feat_fields
        type_field = self.itype_field
        self.train_inter_subset = {}
        self.val_inter_subset = {}
        self.test_inter_subset = {}
        for item_type in self.types:
            item_type_id = item_type_dict[item_type]
            self.train_inter_subset[item_type] = SubSet(
                train[train[type_field] == item_type_id], self.uid_field,
                self.iid_field, self.itype_field, self.label_field, user_fs,
                item_fs[item_type])
            self.val_inter_subset[item_type] = SubSet(
                val[val[type_field] == item_type_id], self.uid_field,
                self.iid_field, self.itype_field, self.label_field, user_fs,
                item_fs[item_type])
            self.test_inter_subset[item_type] = SubSet(
                test[test[type_field] == item_type_id], self.uid_field,
                self.iid_field, self.itype_field, self.label_field, user_fs,
                item_fs[item_type])
        self.all_inter_feat = self.inter_feat
        self.logger.info(
            "Replace interaction features with train interaction fatures.")
        self.logger.info(
            "Interaction features are stored in self.all_inter_feat")
        self.inter_feat = train

    def init_data_loader(self, batch_size=256, num_workers=1):
        self.train_data_loader = {}
        self.val_data_loader = {}
        self.test_data_loader = {}
        for item_type in self.types:
            self.train_data_loader[item_type] = DataLoader(
                self.train_inter_subset[item_type],
                batch_size=batch_size,
                # pin_memory=True,
                num_workers=num_workers)
            self.val_data_loader[item_type] = DataLoader(
                self.val_inter_subset[item_type],
                batch_size=batch_size,
                num_workers=num_workers)
            self.test_data_loader[item_type] = DataLoader(
                self.test_inter_subset[item_type],
                batch_size=batch_size,
                num_workers=num_workers)


class HSubSet(Dataset):
    def __init__(self, dataframes, uid_field, iid_field, label_field,
                 u_feat_fields, i_feat_fields):
        self.types = dataframes.keys()
        self.dfs = dataframes
        self.uid = uid_field
        self.iid = iid_field
        self.label = label_field

    def __len__(self):
        return min([len(df.index) for df in self.dfs])
