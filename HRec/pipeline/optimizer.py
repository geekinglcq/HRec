# -*- coding:utf-8 -*-
# ###########################
# File Name: optimizer.py
# Author: geekinglcq
# Mail: lcqgeek@live.com
# Created Time: 2020-12-18 11:49:59
# ###########################

import torch.optim as optim

opt_map = {
    "Adam": optim.Adam,
    "Adadelta": optim.Adadelta,
    "Adagrad": optim.Adagrad,
    "AdamW": optim.AdamW,
    "RMSprop": optim.RMSprop,
    "SGD": optim.SGD
}


class Optimizer(object):
    def __init__(self, config, params):
        opt_fn = opt_map[config["name"]]
        try:
            self.opt = opt_fn(params, **config["hyper_params"])
        except TypeError:
            print("Unexcepted key error in optimizer")
        self.adjust_lr = config.get("adjust_lr", False)
        if self.adjust_lr:
            self.scheduler = self.get_scheduler(config.get("scheduler"))

    def get_scheduler(self, config):
        if config["name"] == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.opt, **config["hyper_params"])
        else:
            # TODO: Other schedulers
            raise NotImplementedError

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        self.opt.step()
