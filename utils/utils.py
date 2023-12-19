import torch
import torch.nn as nn
import numpy as np
import random
import os
import logging
import colorlog
import math
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math


class LR_Manage:
    def __init__(self,
                 base_lr=1e-4,
                 p=0.9,
                 max_itr=15000,
                 is_cosine_decay=False,
                 min_lr=1e-5,
                 ann_lr_ratio=0.1,
                 warm_up_ratio=0.05):
        self.base_lr = base_lr
        self.p = p
        self.max_itr = max_itr
        self.warm_up_steps = warm_up_ratio * self.max_itr
        self.is_cosine_decay = is_cosine_decay
        self.min_lr = min_lr
        self.ann_lr_ratio = ann_lr_ratio

    def adjust_learning_rate(self, optimizer, itr):
        if itr < self.warm_up_steps:
            now_lr = self.min_lr + (self.base_lr - self.min_lr) * itr / self.warm_up_steps
        else:
            itr = itr - self.warm_up_steps
            max_itr = self.max_itr - self.warm_up_steps
            if self.is_cosine_decay:
                now_lr = self.min_lr + (self.base_lr - self.min_lr) * (math.cos(math.pi * itr /
                                                                                (max_itr + 1)) + 1.) * 0.5
            else:
                now_lr = self.min_lr + (self.base_lr - self.min_lr) * (1 - itr / (max_itr + 1))**self.p

        for param_group in optimizer.param_groups:
            if self.ann_lr_ratio != 1.0 and "ann" in param_group["names"]:
                param_group['lr'] = (now_lr - self.min_lr) * self.ann_lr_ratio + self.min_lr
            else:
                param_group['lr'] = now_lr

        return now_lr


class AverageMeter(object):
    def __init__(self, window=-1):
        self.window = window
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.max = -np.Inf

        if self.window > 0:
            self.val_arr = np.zeros(self.window)
            self.arr_idx = 0

    def update(self, val, n=1):
        if math.isnan(val):
            return
        self.val = val
        self.cnt += n
        self.max = max(self.max, val)

        if self.window > 0:
            self.val_arr[self.arr_idx] = val
            self.arr_idx = (self.arr_idx + 1) % self.window
            self.avg = self.val_arr.mean()
        else:
            self.sum += val * n
            self.avg = self.sum / self.cnt


def setup_seed(seed=1024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logging(filename, resume=False):
    root_logger = logging.getLogger()

    ch = logging.StreamHandler()
    fh = logging.FileHandler(filename=filename, mode='a')

    root_logger.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)

    color_formatter = colorlog.ColoredFormatter('%(log_color)s %(asctime)s %(name)s %(levelname)s %(message)s',
                                                log_colors={
                                                    'DEBUG': 'cyan',
                                                    'INFO': 'green',
                                                    'WARNING': 'yellow',
                                                    'ERROR': 'red',
                                                    'CRITICAL': 'red,bg_white',
                                                })

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    ch.setFormatter(color_formatter)
    fh.setFormatter(formatter)

    root_logger.addHandler(ch)
    root_logger.addHandler(fh)
