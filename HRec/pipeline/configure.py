# -*- coding:utf-8 -*-

import json
class Config(object):

    """Config class that control all config in the experiment.
    """
    def __init__(self, config_path):

        self.dict = json.load(open(config_path))
        for key in ['data', 'model', 'opt']:
            self.dict.update(self.dict[key])

    
    def __getitem__(self, arg):

        if arg in self.dict:
            return self.get(arg)
        else:
            raise ValueError(f'No [{arg}] value in this config.')


