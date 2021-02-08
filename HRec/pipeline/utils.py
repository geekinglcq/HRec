# -*- coding:utf-8 -*-
# ###########################
# File Name: utils.py
# Author: geekinglcq
# Mail: lcqgeek@live.com
# Created Time: 2020-12-20 21:32:39
# ###########################

from gpustat import GPUStatCollection


def get_free_gpu(mode="memory", memory_need=10000) -> list:
    r"""Get free gpu according to mode (process-free or memory-free).
    Args:
        mode (str, optional): memory-free or process-free. Defaults to "memory".
        memory_need (int): The memory you need, used if mode=='memory'. Defaults to 10000.
    Returns:
        list: free gpu ids sorting by free memory
    """
    assert mode in ["memory", "process"], "mode must be 'memory' or 'process'"
    if mode == "memory":
        assert memory_need is not None, \
            "'memory_need' if None, 'memory' mode must give the free memory you want to apply for"
        memory_need = int(memory_need)
        assert memory_need > 0, "'memory_need' you want must be positive"
    gpu_stats = GPUStatCollection.new_query()
    gpu_free_id_list = []

    for idx, gpu_stat in enumerate(gpu_stats):
        if gpu_check_condition(gpu_stat, mode, memory_need):
            gpu_free_id_list.append([idx, gpu_stat.memory_free])
            print("gpu[{}]: {}MB".format(idx, gpu_stat.memory_free))

    if gpu_free_id_list:
        gpu_free_id_list = sorted(gpu_free_id_list,
                                  key=lambda x: x[1],
                                  reverse=True)
        gpu_free_id_list = [i[0] for i in gpu_free_id_list]
    return gpu_free_id_list


def gpu_check_condition(gpu_stat, mode, memory_need) -> bool:
    r"""Check gpu is free or not.
    Args:
        gpu_stat (gpustat.core): gpustat to check
        mode (str): memory-free or process-free.
        memory_need (int): The memory you need, used if mode=='memory'
    Returns:
        bool: gpu is free or not
    """
    if mode == "memory":
        return gpu_stat.memory_free > memory_need
    elif mode == "process":
        for process in gpu_stat.processes:
            if process["command"] == "python":
                return False
        return True
    else:
        return False


class EarlyStopping(object):
    """
    The class control the info to decide whether to do earlystopping
    """
    best_score = None
    best_epoch = None
    steps = 0

    @classmethod
    def update(self, scores, epoch, metric='auc', mode='max', stop_step=5):
        """
        Update current suitation after each epoch.
        Args:
            scores: a dict store metrice name-value pair
            metric: which metric to use in earlystopping
            mode: 'max' or 'min'
            stop_step: if after the given num of epochs, the
                model does not improve then stop the training
        Return:
            stop_flag: if or not stop
            better: if current version is the best
        """
        assert metric in scores.keys()
        assert mode in ['max', 'min']

        def _set_best(score, epoch):
            EarlyStopping.best_score = score
            EarlyStopping.best_epoch = epoch
            EarlyStopping.steps = 0

        def _compare(score, mode):
            comp_fn = {"max": lambda a, b: a > b, "min": lambda a, b: a < b}
            return comp_fn[mode](score, EarlyStopping.best_score)

        score = scores[metric]
        better = False
        if EarlyStopping.best_score is None:
            _set_best(score, epoch)
            better = True
        elif _compare(score, mode):
            _set_best(score, epoch)
            better = True
        else:
            EarlyStopping.steps += 1
        if EarlyStopping.steps >= stop_step:
            return True, better
        return False, better


if __name__ == '__main__':
    get_free_gpu()
