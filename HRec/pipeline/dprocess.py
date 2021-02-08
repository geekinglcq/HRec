# -*- coding:utf-8 -*-
# ###########################
# File Name: dprocess.py
# Author: geekinglcq
# Mail: lcqgeek@live.com
# Created Time: 2021-02-03 02:14:47
# ###########################

import os
from tqdm import tqdm
from .optimizer import Optimizer
import torch
from ..models import model_map
from .hprocess import HProcess
from collections import defaultdict
from torch.autograd import Variable


class DProcess(HProcess):
    """
    Process for the DDTCDR model
    """
    def __init__(self, config):
        self.config = config
        self._path_config(config['path'])
        self._logger_config()
        self._set_device(config)
        self._prepare_data(config['data'])
        self._prepare_model(config['model'])
        self._prepare_optimizer(config['opt'])
        self._prepare_evaluator(config)

    def _prepare_model(self, model_config):
        self.models = {}
        name = model_config['name']
        for item_type in self.types:
            model = model_map[name](model_config, self.dataset, item_type)
            self.models[item_type] = model.to(self.device)
        self.model = self.models[item_type]
        self.crit = torch.nn.BCELoss()
        self.alpha = model_config['alpha']

    def _prepare_optimizer(self, opt_config):
        self.opts = {}
        for item_type in self.types:
            opt = Optimizer(opt_config, self.models[item_type].parameters())
            self.opts[item_type] = opt
        self.start_epoch = 0
        self.best_val_score = -1
        self.epochs = opt_config['epochs']
        self.eval_step = opt_config['eval_step']
        self.save_step = opt_config['save_step']
        self.train_loss_dict = {}
        self.val_loss_dict = {}
        if 'early_stop' in opt_config.keys():
            self.early_stop = True
            config = opt_config['early_stop']
            self.eval_metric = config.get('metric', 'auc')
            self.eval_mode = config.get('mode', 'max')
            self.stop_step = config.get('stop_step', 5)
        else:
            self.early_stop = False

    def train_one_batch(self, hdata):

        for opt in self.opts.values():
            opt.zero_grad()

        preds = defaultdict(dict)
        losses = defaultdict(dict)

        for item_model_type in self.types:
            for item_type in self.types:
                if item_type == item_model_type:
                    preds[item_model_type][item_type] = self.models[
                        item_model_type](item_type, hdata[item_type])
                else:
                    preds[item_model_type][item_type] = self.models[
                        item_model_type](item_type,
                                         hdata[item_type],
                                         dual=True)
                label = hdata[item_type][self.LABEL].reshape((-1, 1)).float()
                losses[item_model_type][item_type] = self.crit(
                    preds[item_model_type][item_type], label)

        # wighted loss
        w_loss = defaultdict(list)
        for item_type in self.types:
            for item_model_type in self.types:
                if item_type == item_model_type:
                    loss = (1 -
                            self.alpha) * losses[item_model_type][item_type]
                else:
                    # change variable to Tensor if error
                    loss = self.alpha * Variable(
                        losses[item_model_type][item_type].data,
                        requires_grad=False)
                w_loss[item_type].append(loss)
        t_loss = {}
        for k, v in w_loss.items():
            t_loss[k] = torch.sum(torch.stack(w_loss[k], dim=0))
            t_loss[k].backward(retain_graph=True)
        orth_loss = {}
        for item_type in self.types:
            orth_loss[item_type] = torch.zeros(1, device=self.device)
        reg = 1e-6

        for item_type, model in self.models.items():
            for name, param in model.bridge.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0], device=self.device)
                    orth_loss[item_type] += reg * sym.abs().sum()
            orth_loss[item_type].backward()
        for item_type in self.types:
            self.opts[item_type].step()

        r_loss = 0
        for _, loss in t_loss.items():
            r_loss += loss.data.cpu().item()
        for _, l in orth_loss.items():
            r_loss += loss.data.cpu().item()

        return r_loss

    def train_one_epoch(self, data_loader=None):
        """Train one epoch using given data"""
        if data_loader is None:
            data_loader = self.dataset.train_data_loader
        max_len = max([len(dl) for dl in data_loader.values()])
        m = self.model

        m.train()
        losses_dict = defaultdict(float)

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

            r_loss = self.train_one_batch(hdata)

        losses_dict['total'] = r_loss

        return losses_dict

    def validate(self, data_loader=None):
        """
        Run model in validation dataset and calculate the
        score using evaluator.
        Return:
            result: a dict store metrics name-value pair.
        """
        if data_loader is None:
            data_loader = self.dataset.val_data_loader
        ms = self.models
        for m in ms.values():
            m.eval()

        lens = [len(dl) for dl in data_loader.values()]
        batch_matrix_list = []
        with tqdm(total=sum(lens)) as pbar:
            for item_type, dl in data_loader.items():
                for data in dl:

                    if type(data) is dict:
                        for key, value in data.items():
                            data[key] = value.to(self.device)
                    pred = ms[item_type].predict(item_type, data).reshape(
                        (-1, ))
                    batch_matrix = self.evaluator.collect(data, pred)
                    batch_matrix_list.append(batch_matrix)
                    pbar.update(1)

        result = self.evaluator.evaluate(batch_matrix_list, groupby=True)
        return result

    def save_checkpoint(self, epoch, name='last', path=None):
        if path is None:
            path = self.ckp_path
        model_dict = {}
        opt_dict = {}
        for item_type in self.models:
            model_dict[item_type] = self.models[item_type].state_dict()
            opt_dict[item_type] = self.opts[item_type].opt.state_dict()
        state = {
            'epoch': epoch,
            'state_dict': model_dict,
            'optimizer': opt_dict
        }
        if name == 'last':
            file_name = os.path.join(self.ckp_path,
                                     f'{name}-{epoch}-model.pth')
        else:
            file_name = os.path.join(self.ckp_path, f'{name}-model.pth')
        self.last_model_path = file_name
        torch.save(state, file_name)
        if name == 'best':
            self.best_ckp_path = file_name
        elif name == 'last':
            self.last_ckp_path = file_name

    def load_checkpoint(self, file_name=None, mode=None):
        if file_name is None:
            if mode == 'last':
                file_name = self.last_model_path
            elif mode == 'best':
                file_name = self.best_model_path
            else:
                raise ValueError("No checkpoint path provided.")
        ckp = torch.load(file_name)
        self.start_epoch = ckp['epoch'] + 1
        for item_type in self.models:
            self.models[item_type].load_state_dict(
                ckp['state_dict'][item_type])
            self.opts[item_type].opt.load_state_dict(
                ckp['optimizer'][item_type])
        self.logger.info(f"Load ckp from {file_name}.")
