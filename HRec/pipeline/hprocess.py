# -*- coding:utf-8 -*-
# ###########################
# File Name: hprocess
# Author: geekinglcq
# Mail: lcqgeek@live.com
# Created Time: 2021-01-06 16:24:08
# ###########################

from tqdm import tqdm
import numpy as np
import torch.nn as nn
from ..datasets import HDataSet, SubSet
from .process import Process
from collections import defaultdict
from torch.utils.data import DataLoader


class HProcess(Process):
    """
    Process for Heterogeneous Recommendation
    """
    def __init__(self, config):
        self.config = config
        self._path_config(config['path'])
        self._logger_config()
        self._set_device(config)
        self._prepare_data(config['data'])
        self._prepare_model(config['model'])
        self._prepare_optimizer(config['opt'], self.model.parameters())
        self._prepare_evaluator(config)

    def train_one_epoch(self, data_loader=None):
        """Train one epoch using given data"""
        if data_loader is None:
            data_loader = self.dataset.train_data_loader
        max_len = max([len(dl) for dl in data_loader.values()])
        m = self.model
        loss_fn = self.model.calculate_loss

        m.train()
        losses = None
        losses_dict = defaultdict(float)
        opt = self.opt.opt
        if hasattr(self.opt, 'scheduler'):
            pass
            # TODO: scheduler
            # sch = self.opt.scheduler
        else:
            pass
            # sch = None
        iters = {}
        hdata = {}
        for idx in tqdm(range(max_len), total=max_len):

            for item_type, dl in data_loader.items():
                try:
                    data_iter = iters.get(item_type, None)
                    if data_iter is None:
                        iters[item_type] = iter(dl)
                        data_iter = iters[item_type]
                    data = next(data_iter)
                except:
                    iters[item_type] = iter(dl)
                    data = next(iters[item_type])

                # self.modelsize(self.model, data)
                if type(data) is dict:
                    for key, value in data.items():
                        data[key] = value.to(self.device)
                hdata[item_type] = data

            opt.zero_grad()
            loss, loss_dict = loss_fn(hdata)
            loss.backward()
            opt.step()

            losses = loss.item() if losses is None else losses + loss.item()
            for k, v in loss_dict.items():
                losses_dict[k] += v
        losses_dict['total'] = losses

        return losses_dict

    def get_item_embeddings(self, item_kind):
        item_feat = self.dataset.item_feat[item_kind]
        item_set = SubSet(item_feat, None, self.dataset.iid_field,
                          self.dataset.itype_field, None, None,
                          self.dataset.item_feat_fields[item_kind])
        dl = DataLoader(item_set)
        id2mapemb = {}
        id2emb = {}
        id2rawemb = {}
        for data in dl:
            if type(data) is dict:
                for k, v in data.items():
                    data[k] = v.to(self.device)
                rawembs, mapembs, embs = self.model.get_item_embedding(
                    item_kind, data)
                ids = data['item_id'].cpu().detach().numpy()
                embs = embs.cpu().detach().numpy()
                mapembs = mapembs.cpu().detach().numpy()
                rawembs = rawembs.cpu().detach().numpy()
                for idx, mapemb, emb, rawemb in zip(ids, mapembs, embs,
                                                    rawembs):
                    id2mapemb[idx] = mapemb
                    id2emb[idx] = emb
                    id2rawemb[idx] = rawemb

        return id2emb, id2mapemb, id2rawemb

    def validate(self, data_loader=None):
        """
        Run model in validation dataset and calculate the
        score using evaluator.
        Return:
            result: a dict store metrics name-value pair.
        """
        if data_loader is None:
            data_loader = self.dataset.val_data_loader
        m = self.model
        m.eval()

        lens = [len(dl) for dl in data_loader.values()]
        batch_matrix_list = []
        with tqdm(total=sum(lens)) as pbar:
            for item_type, dl in data_loader.items():
                for data in dl:

                    if type(data) is dict:
                        for key, value in data.items():
                            data[key] = value.to(self.device)
                    pred = m.predict(item_type, data)
                    batch_matrix = self.evaluator.collect(data, pred)
                    batch_matrix_list.append(batch_matrix)
                    pbar.update(1)

        result = self.evaluator.evaluate(batch_matrix_list, groupby=True)
        return result

    def test(self, data_loader=None):
        """
        Test
        """
        if data_loader is None:
            data_loader = self.dataset.test_data_loader
        return self.validate(data_loader=data_loader)

    def _prepare_data(self, data_config):
        self.dataset = HDataSet(data_config)
        self.LABEL = self.dataset.config['LABEL_FIELD']
        self.types = self.dataset.types

    def modelsize(self, model, input, type_size=4):
        para = sum([np.prod(list(p.size())) for p in model.parameters()])
        print('Model {} : params: {:4f}M'.format(
            model._get_name(), para * type_size / 1000 / 1000))
        input_ = input
        input_.requires_grad_(requires_grad=False)
        mods = list(model.modules())
        out_sizes = []

        for i in range(1, len(mods)):
            m = mods[i]
            if isinstance(m, nn.ReLU):
                if m.inplace:
                    continue
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

        total_nums = 0
        for i in range(len(out_sizes)):
            s = out_sizes[i]
            nums = np.prod(np.array(s))
            total_nums += nums
        print('Model {} : intermedite variables: {:3f} M (without backward)'.
              format(model._get_name(), total_nums * type_size / 1000 / 1000))
        print(
            'Model {} : intermedite variables: {:3f} M (with backward)'.format(
                model._get_name(), total_nums * type_size * 2 / 1000 / 1000))
