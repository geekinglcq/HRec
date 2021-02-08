# -*- coding:utf-8 -*-
# ###########################
# File Name:
# Author: geekinglcq
# Mail: lcqgeek@live.com
# Created Time: 2020-12-18 15:34:30
# ###########################

import numpy as np
import torch
from .metrics import metrics_dict

# These metrics are typical in loss recommendations
loss_metrics = {
    metric.lower(): metric
    for metric in ['AUC', 'RMSE', 'MAE', 'LOGLOSS']
}


class Evaluator(object):
    r"""Loss Evaluator is mainly used in rating prediction and click through rate prediction. Now, we support four
    loss metrics which contain `'AUC', 'RMSE', 'MAE', 'LOGLOSS'`.
    """
    def __init__(self, config):
        super().__init__()

        self.metrics = config['metrics']

        self.label_field = config['data']['LABEL_FIELD']
        self.type_field = config['data']['TYPE_FIELD']
        self._check_args()

    def collect(self, interaction, pred_scores):
        """collect the loss intermediate result of one batch, this function mainly
        implements concatenating preds and trues. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            pred_scores (tensor): the tensor of model output with a size of `(N, )`

        Returns:
            tensor : a batch of socres with a size of `(N, 2)`

        """
        true_scores = interaction[self.label_field].to(pred_scores.device)
        types = interaction[self.type_field].to(pred_scores.device).float()
        assert len(true_scores) == len(pred_scores)
        return torch.stack((true_scores.float(), pred_scores.detach(), types),
                           dim=1)

    def evaluate(self, batch_matrix_list, groupby=False, *args):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches

        Returns:
            dict: such as {'AUC': 0.83}

        """
        concat = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        metric_dict = {}
        if groupby:
            types = concat[:, 2]
            for t in np.unique(types):
                trues = concat[types == t][:, 0]
                preds = concat[types == t][:, 1]
                result_list = self._calculate_metrics(trues, preds)
                for metric, value in zip(self.metrics, result_list):
                    key = str(t) + "-" + str(metric)
                    metric_dict[key] = round(value, 4)

        trues = concat[:, 0]
        preds = concat[:, 1]
        # get metrics
        result_list = self._calculate_metrics(trues, preds)
        for metric, value in zip(self.metrics, result_list):
            key = str(metric)
            metric_dict[key] = round(value, 4)

        return metric_dict

    def _check_args(self):

        # Check metrics
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError('metrics must be str or list')

        # Convert metric to lowercase
        for m in self.metrics:
            if m.lower() not in loss_metrics:
                raise ValueError("There is no loss metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]

    def metrics_info(self, trues, preds):
        """get metrics result

        Args:
            trues (np.ndarray): the true scores' list
            preds (np.ndarray): the predict scores' list

        Returns:
            list: a list of metrics result

        """
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(trues, preds)
            result_list.append(result)
        return result_list

    def _calculate_metrics(self, trues, preds):
        return self.metrics_info(trues, preds)

    def __str__(self):
        mesg = 'The Loss Evaluator Info:\n' + '\tMetrics:[' + ', '.join(
            [loss_metrics[metric.lower()] for metric in self.metrics]) + ']'
        return mesg
