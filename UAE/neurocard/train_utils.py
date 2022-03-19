"""Utility functions for training."""

import numpy as np
import os
import psutil
import torch
import torch.nn as nn

import made

# from prettytable import PrettyTable
from pynvml import *
nvmlInit()

def get_device(core = None):
#     if core is None:
    return 'cuda' if torch.cuda.is_available() else 'cpu'
#     else:
#         assert isinstance(core,int)
#         print(f'USE cuda:{core}')
#         return f'cuda:{core}' if torch.cuda.is_available() else 'cpu'




def weight_init(m):
    if type(m) == made.MaskedLinear or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
#     print(model)
    return mb


# Forked from https://github.com/google-research/google-research/blob/master/opt_list/opt_list/torch_opt_list.py.
def get_cosine_learning_rate_fn(training_steps, learning_rate,
                                min_learning_rate_mult, constant_fraction,
                                warmup_fraction):
    """Get a function that does cosine learning rate decay with warmup.

    The learning rate starts at zero, is "warmed up" linearly over
    `warmup_fraction * training_steps` iterations to achieve a final value of
    `learning_rate`. A constant learning rate of `learning_rate` is held up
    until `training_steps*constant_fraction` at which point a cosine decay is
    started to a final learning rate of `min_learning_rate_mult *
    learning_rate`.  The cosine decay sets the learning rate using a
    monotomically decreasing section of the cosine function from 0 to pi/2. It
    has been proven to be useful in large large language modeling (gpt,
    megatron-lm) and image classification. See https://arxiv.org/abs/1608.03983
    for more information on the cosine decay.

    Args:
      training_steps: number of training steps the schedule should be run for.
      learning_rate: base learning rate. This is the learning rate used just
        after warmup and where the decay starts from.
      min_learning_rate_mult: a multiplicative factor to control how low the
        learning rate should be decayed to.
      constant_fraction: the fraction of training steps number of steps to take
        before starting the decay. This includes the time spent warming up the
        learning rate.
      warmup_fraction: the fraction of training steps to use for a learning
        rate warmup.
    Returns:
      A function that takes as input a training iteration and returns the
      learning rate from the specified schedule.
    """

    def ff(x):
        return torch.tensor(x, dtype=torch.float32)

    def fn(global_step):
        """Returns a learning rate given the current training iteration."""

        float_training_steps = ff(training_steps)
        global_step = ff(global_step)

        # ensure we don't train longer than training steps
        global_step = torch.min(global_step, float_training_steps)

        constant_steps = float_training_steps * constant_fraction
        x = torch.max(ff(global_step), ff(constant_steps))

        min_learning_rate = min_learning_rate_mult * learning_rate

        if warmup_fraction:
            min_warmup_fraction = max(warmup_fraction, constant_fraction)
            warmup_steps = float_training_steps * min_warmup_fraction
            is_warmup = ff(ff(warmup_steps) > ff(global_step))
            warmup_lr = (global_step / warmup_steps) * learning_rate
        else:
            warmup_lr = learning_rate
            is_warmup = 0.0

        step = x - constant_steps

        constant_and_decay = (learning_rate - min_learning_rate) * (
            torch.cos(step * np.pi /
                      (float_training_steps - constant_steps)) / 2.0 +
            0.5) + min_learning_rate

        new_learning_rate = constant_and_decay * (
            1.0 - is_warmup) + is_warmup * (warmup_lr)
        return new_learning_rate

    return fn


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f', display_average=True):
        self.name = name
        self.fmt = fmt
        self.display_average = display_average
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if not isinstance(self.fmt, str):
            return '{} {}'.format(self.name, self.fmt(self.val))

        if self.display_average:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        else:
            fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def print_tensor_mem(var,step,tag=''):
    print_step = 20
    if step%print_step != 0:
        return
    mem_size = var.element_size() * var.nelement()/(1024)
    gpu_usage = torch.cuda.memory_allocated(0) / (1024*1024)
    print(f"[MEM] {tag}:{mem_size:.2f}KB in {gpu_usage}\t({var.shape})")

def get_current_mem():
    pid = os.getpid()
    current_process = psutil.Process(pid)
    current_process_memory_usage_as_MB = current_process.memory_info()[0] / 2.**30
    memory_usage_dict = dict(psutil.virtual_memory()._asdict())
    memory_usage_percent = memory_usage_dict['percent']
    return current_process_memory_usage_as_MB,memory_usage_percent

def cur_mem():
    mem_usage, mem_usage_percent = get_current_mem()
    return f"\nmemory usage : {mem_usage:2.f}MB\nmemory usage persent : {mem_usage_percent:.2f}%"

def print_gpu_info(log='',num=0,step=0):
#     return None
    print_step = 100
    if step % print_step != 0:
        return
    h = nvmlDeviceGetHandleByIndex(num)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f"{log}")
    print(f'\ttotal    : {info.total/(1024*1024):.4f}MB')
    print(f'\tfree     : {info.free/(1024*1024):.4f}MB')
    print(f'\tused     : {info.used/(1024*1024):.4f}MB\tstep:{step}')