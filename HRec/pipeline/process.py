# -*- coding:utf-8 -*-
import json
import os
import time
import logging
from datetime import datetime
from collections import defaultdict

import torch
from tqdm import tqdm

from ..datasets import DataSet
from ..models import model_map, ModelType
from .evaluator import Evaluator
from .optimizer import Optimizer
from .utils import get_free_gpu, EarlyStopping


class Process(object):
    def __init__(self, config):
        self.config = config
        self._path_config(config['path'])
        self._logger_config()
        self._set_device(config)
        self._prepare_data(config['data'])
        self._prepare_model(config['model'])
        self._prepare_optimizer(config['opt'], self.model.parameters())
        self._prepare_evaluator(config)

    def _set_device(self, config):
        device = config.get('device', None)
        if device is not None:
            self.device = device
        else:
            device_list = get_free_gpu(mode="memory", memory_need=5000)
            if len(device_list) < 1:
                raise ValueError("No GPU available now.")
            else:
                self.device = f'cuda:{device_list[0]}'
                self.logger.info(f'Use device {self.device}')
        config['data']['device'] = self.device
        config['model']['device'] = self.device

    def _logger_config(self):
        """
        Set the logger
        """
        model_name = self.config['model']['name']
        data_name = self.config['data']['name']
        if self.config['data'].get('single', False):
            single_type = self.config['data'].get('single_type')
            data_name = f'{data_name}_{single_type}'
        logfile_name = os.path.join(self.log_path,
                                    f'{model_name}-{data_name}.log')
        fmt = "%(asctime)-15s %(levelname)s %(message)s"
        filedatefmt = "%a %d %b %Y %H:%M:%S"
        fileformatter = logging.Formatter(fmt, filedatefmt)

        sdatefmt = "%d %b %H:%M"
        sformatter = logging.Formatter(fmt, sdatefmt)

        fh = logging.FileHandler(logfile_name)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fileformatter)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(sformatter)
        logging.basicConfig(level=logging.INFO, handlers=[fh, sh])
        self.logger = logging.getLogger()

    def _prepare_model(self, model_config):
        name = model_config['name']
        self.model = model_map[name](model_config, self.dataset)
        self.model.to(self.device)

    def _prepare_data(self, data_config):
        self.dataset = DataSet(data_config)
        self.LABEL = self.dataset.config['LABEL_FIELD']
        self.single = self.dataset.single

    def _prepare_optimizer(self, opt_config, params):
        self.opt = Optimizer(opt_config, params)
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

    def _prepare_evaluator(self, config):
        self.evaluator = Evaluator(config)

    def train_one_epoch(self, data_loader=None):
        """Train one epoch using given data"""
        if data_loader is None:
            data_loader = self.dataset.train_data_loader

        m = self.model
        loss_fn = self.model.calculate_loss

        m.train()
        losses = None
        opt = self.opt.opt
        if hasattr(self.opt, 'scheduler'):
            pass
            # TODO: scheduler
            # sch = self.opt.scheduler
        else:
            pass
            # sch = None
        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            if type(data) is dict:
                for key, value in data.items():
                    data[key] = value.to(self.device)
            opt.zero_grad()
            loss = loss_fn(data)
            loss.backward()
            opt.step()
            losses = loss.item() if losses is None else losses + loss.item()
        return losses

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

        batch_matrix_list = []
        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            if type(data) is dict:
                for key, value in data.items():
                    data[key] = value.to(self.device)
            pred = m.predict(data)
            batch_matrix = self.evaluator.collect(data, pred)
            batch_matrix_list.append(batch_matrix)

        if self.single:
            result = self.evaluator.evaluate(batch_matrix_list, groupby=False)
        else:
            result = self.evaluator.evaluate(batch_matrix_list, groupby=True)
        return result

    def test(self, data_loader=None):

        if data_loader is None:
            data_loader = self.dataset.test_data_loader
        return self.validate(data_loader=data_loader)

    def save_checkpoint(self, epoch, name='last', path=None):
        if path is None:
            path = self.ckp_path
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.opt.opt.state_dict()
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
        self.model.load_state_dict(ckp['state_dict'])
        self.opt.opt.load_state_dict(ckp['optimizer'])
        self.logger.info(f"Load ckp from {file_name}.")

    def fit(self,
            train_data=None,
            val_data=None,
            test_data=None,
            verbose=True):

        if self.model.model_type == ModelType.CONTEXT:
            self.dataset.join_interaction()
            self.dataset.train_val_test_split(context=True)
        elif self.model.model_type == ModelType.HETERO:
            self.dataset.join_interaction()
            self.dataset.train_val_test_split()
        else:
            self.dataset.train_val_test_split(context=False)

        batch_size = self.config['opt'].get('batch_size', 256)
        num_workers = self.dataset.config.get('num_workers', 2)
        self.dataset.init_data_loader(batch_size=batch_size,
                                      num_workers=num_workers)
        for epoch_idx in range(self.start_epoch, self.epochs):

            # Train
            st = time.time()
            train_loss = self.train_one_epoch(train_data)
            self.train_loss_dict[epoch_idx] = train_loss
            ed = time.time()

            if verbose:
                if type(train_loss) is dict or type(train_loss) is defaultdict:
                    train_loss = '\t'.join(
                        [f'{k}: {v}' for k, v in train_loss.items()])
                self.logger.info(
                    f'[TRAIN] Epoch: {epoch_idx} cost time: {ed - st:.1f}, train loss: {train_loss}'
                )

            # Eval
            if not ((epoch_idx + 1) % self.eval_step):
                st = time.time()
                result = self.validate(val_data)
                ed = time.time()
                self.logger.info(
                    f'[EVAL] Epoch: {epoch_idx} cost time: {ed - st:.1f}')
                result_str = '[EVAL] ' + '\t'.join(
                    [f'{k}: {v} ' for k, v in result.items()])
                stop_flag, better = EarlyStopping.update(
                    result, epoch_idx, self.eval_metric, self.eval_mode,
                    self.stop_step)
                self.logger.info(result_str)

                # Save the best model
                if better:
                    self.save_checkpoint(epoch_idx, 'best')
                if self.early_stop and stop_flag:
                    self.logger.info(f'Early Stop in {epoch_idx} epoch. ')
                    break

            if not ((epoch_idx + 1) % self.save_step):
                self.save_checkpoint(epoch_idx, 'last')

        # Test
        self.logger.info(
            'Finish training. Start to evaluate in the test set using the best model in val set.'
        )
        if hasattr(self, 'best_ckp_path'):
            self.load_checkpoint(self.best_ckp_path)
        result = self.test(test_data)
        result_str = '[TEST] ' + '\t'.join(
            [f'{k}: {v:.3f} ' for k, v in result.items()])
        self.logger.info(result_str)
        self.config['result'] = result
        # Save the result to config file
        json.dump(self.config,
                  open(os.path.join(self.output_path, "config.json"), "w"),
                  indent='\t')

    def _path_config(self, config):
        now = str(datetime.now()).replace(" ", "_").split(".")[0]
        model_name = self.config['model']['name']
        data_name = self.config['data']['name']
        output_path = os.path.join(config["output"],
                                   f'{model_name}-{data_name}-{now}')
        self.output_path = output_path
        if os.path.isdir(output_path):
            raise ValueError("Output dir already exist")
        else:
            os.makedirs(output_path)
        # Save config files
        json.dump(self.config,
                  open(os.path.join(output_path, "config.json"), "w"),
                  indent='\t')
        print(f"Config is saved in {output_path}.")
        for sub_dir in ["log", "ckp"]:
            path = os.path.join(output_path, sub_dir)
            if not os.path.exists(path):
                os.mkdir(path)
            setattr(self, f'{sub_dir}_path', path)
            print(f'{sub_dir} is saved in {path}')
