from config import cfg
import time
import torch
from .RST import build_RST

def build_model(args):
    model_type = cfg.MODEL.TYPE
    model = None
    if model_type == 'RST':
        model = build_RST(args, cfg)
    else:
        raise ValueError("Unsupport model type")
    return model