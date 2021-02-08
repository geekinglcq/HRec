# -*- coding:utf-8 -*-

import gc
import torch
from enum import Enum


class ModelType(Enum):
    """Type of models:
        - GENERAL: General model, treat different type item as the same.
        - HETERO: Heterogenous model
    """
    GENERAL = 1
    CONTEXT = 2
    HETERO = 3


## MEM utils ##
def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''
    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type

        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)

        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' % (mem_type))
        print('-' * LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            import pdb
            pdb.set_trace()
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel * element_size / 1024 / 1024  # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (element_type, size, mem))
        print('-' * LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' %
              (total_numel, total_mem))
        print('-' * LEN)

    LEN = 65
    print('=' * LEN)
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' % ('Element type', 'Size', 'Used MEM(MBytes)'))
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('=' * LEN)
